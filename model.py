import math
import random
import functools
import operator

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function

from op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d, conv2d_gradfix


class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer("kernel", kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer("kernel", kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = conv2d_gradfix.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})"
        )


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        fused=True,
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size)
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate
        self.fused = fused

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, "
            f"upsample={self.upsample}, downsample={self.downsample})"
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        if not self.fused:
            weight = self.scale * self.weight.squeeze(0)
            style = self.modulation(style)

            if self.demodulate:
                w = weight.unsqueeze(0) * style.view(batch, 1, in_channel, 1, 1)
                dcoefs = (w.square().sum((2, 3, 4)) + 1e-8).rsqrt()

            input = input * style.reshape(batch, in_channel, 1, 1)

            if self.upsample:
                weight = weight.transpose(0, 1)
                out = conv2d_gradfix.conv_transpose2d(
                    input, weight, padding=0, stride=2
                )
                out = self.blur(out)

            elif self.downsample:
                input = self.blur(input)
                out = conv2d_gradfix.conv2d(input, weight, padding=0, stride=2)

            else:
                out = conv2d_gradfix.conv2d(input, weight, padding=self.padding)

            if self.demodulate:
                out = out * dcoefs.view(batch, -1, 1, 1)

            return out

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = conv2d_gradfix.conv_transpose2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=0, stride=2, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            input = input.view(1, batch * in_channel, height, width)
            out = conv2d_gradfix.conv2d(
                input, weight, padding=self.padding, groups=batch
            )
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input, shift_h=0, shift_w=0, transform=None):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)
        # shift. details in appendix
        out = torch.roll(out, int(shift_h), 2) * (1 - shift_h + int(shift_h)) + torch.roll(out, int(shift_h) + 1, 2) * (
                    shift_h - int(shift_h))
        out = torch.roll(out, int(shift_w), 3) * (1 - shift_w + int(shift_w)) + torch.roll(out, int(shift_w) + 1, 3) * (
                    shift_w - int(shift_w))

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class Generator(nn.Module):
    def __init__(
        self,
        size,
        style_dim,
        n_mlp,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        position='none',
        kernel_size=3,
        affine=False,
        scale=1.0,
        device="cuda",
    ):
        super().__init__()

        self.size = size

        self.style_dim = style_dim

        self.position = position
        self.affine = affine
        self.device = device

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation="fused_lrelu"
                )
            )

        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }
        if 'pe' in self.position:
            self.input = PE2dStart(512, 4, 4, scale=scale, device=device)
        else:
            self.input = ConstantInput(self.channels[4])
        self.shift_h_dict = {4: 0}
        self.shift_w_dict = {4: 0}

        if self.affine:
            self.affine_fourier = EqualLinear(style_dim, 4)
            self.affine_fourier.weight.detach().zero_()
            self.affine_fourier.bias.detach().copy_(
                torch.tensor([1, 0, 0, 0], dtype=torch.float32)
            )

        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], kernel_size, style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()
        if 'mspe' in self.position:
            self.pes = nn.ModuleList()
        for i in range(3, self.log_size + 1):
            self.shift_h_dict[2 ** i] = 0
            self.shift_w_dict[2 ** i] = 0

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f"noise_{layer_idx}", torch.randn(*shape))

        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    kernel_size,
                    style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )
            ###### positional encoding ######
            if 'mspe' in self.position:
                self.pes.append(PE2d(out_channel, 2 ** i, 2 ** i, scale, device=device))

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, kernel_size, style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def make_noise(self):
        device = self.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
        shift_h=0,
        shift_w=0,
        transform=None,
        pe_transform = None
    ):
        ##### positional encoding #####
        # continuous roll
        if shift_h:
            for i in range(2, self.log_size + 1):
                self.shift_h_dict[2 ** i] = shift_h / (self.size // (2 ** i))
        if shift_w:
            for i in range(2, self.log_size + 1):
                self.shift_w_dict[2 ** i] = shift_w / (self.size // (2 ** i))

        if not input_is_latent:
            styles = [self.style(s) for s in styles]

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers
            else:
                noise = [
                    getattr(self.noises, f"noise_{i}") for i in range(self.num_layers)
                ]

        if truncation < 1:
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2:
            inject_index = self.n_latent

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)

            else:
                latent = styles[0]

        else:
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1)

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        if self.affine and transform is None:
            transform = self.affine_fourier(latent)

        #if 'pe' in self.position:
        out = self.input(latent, self.shift_h_dict[4], self.shift_w_dict[4], pe_transform=pe_transform)
        #else:
        #    out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])

        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)

            if 'mspe' in self.position:
                res = 8 * (2 ** ((i - 1) // 2))
                out = self.pes[(i - 1) // 2](out, shift_h=self.shift_h_dict[res], shift_w=self.shift_w_dict[res],pe_transform=pe_transform)

            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)

            i += 2

        image = skip

        if return_latents:
            return image, latent

        else:
            return image, None


class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            layers.append(FusedLeakyReLU(out_channel, bias=bias))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=2, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, blur_kernel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.stddev_group = 4
        self.stddev_feat = 1

        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)

        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out

########## POSITONAL ENCODING #############
class PE2d(nn.Module):
    def __init__(self, channel, height, width, scale=1.0, device="cuda"):
        super().__init__()
        if channel % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                                "odd dimension (got dim={:d})".format(channel))
        height = int(height * scale)
        width = int(width * scale)
        self.pe = torch.zeros(channel, height, width, device=device)

        # Each dimension use half of d_model
        self.d_model = int(channel / 2)
        self.half_d = int(self.d_model/2)
        self.div_term = torch.exp(torch.arange(0., self.d_model, 2) * -(math.log(10000.0) / self.d_model)) / scale
        self.pos_h = torch.arange(0., height).unsqueeze(1)
        self.pos_w = torch.arange(0., width).unsqueeze(1)

        self.gamma = nn.Parameter(torch.ones(1))

    def forward(self, x, shift_h=0, shift_w=0, pe_transform=None):
        height, width = self.pos_h.shape[0], self.pos_w.shape[0]

        # Index based positional encoding
        initial_pe = torch.zeros(2,height,width)
        initial_pe[0,:,:]= self.pos_w.transpose(0,1).unsqueeze(1).repeat(1,height,1)
        initial_pe[1,:,:]= self.pos_h.transpose(0,1).unsqueeze(2).repeat(1,1,width)

        # Affine Transformation of the encoding torus
        if pe_transform is not None:
          initial_pe[0,:,:] -= int(width/2)
          initial_pe[1,:,:] -= int(height/2)
          initial_pe=torch.tensordot(pe_transform,initial_pe,1)
          initial_pe[0,:,:] += int(width/2)
          initial_pe[1,:,:] += int(height/2)
        initial_pe[0,:,:] += shift_w
        initial_pe[1,:,:] += shift_h
        initial_pe[0,:,:] = torch.remainder(initial_pe[0], width)
        initial_pe[1,:,:] = torch.remainder(initial_pe[1], height)

        # Convert to quadtree encoding
        self.pe[0:self.d_model:2,:,:] = torch.sin(torch.reshape(torch.outer(torch.reshape(initial_pe[0],(1,height*width)).squeeze(),self.div_term),(height,width,self.half_d))).permute(2,0,1)
        self.pe[1:self.d_model:2,:,:] = torch.cos(torch.reshape(torch.outer(torch.reshape(initial_pe[0],(1,height*width)).squeeze(),self.div_term),(height,width,self.half_d))).permute(2,0,1)
        self.pe[self.d_model::2,:,:] = torch.sin(torch.reshape(torch.outer(torch.reshape(initial_pe[1],(1,height*width)).squeeze(),self.div_term),(height,width,self.half_d))).permute(2,0,1)
        self.pe[self.d_model+1::2,:,:] = torch.cos(torch.reshape(torch.outer(torch.reshape(initial_pe[1],(1,height*width)).squeeze(),self.div_term),(height,width,self.half_d))).permute(2,0,1)

        return x + self.gamma * self.pe.unsqueeze(0)

class PE2dStart(nn.Module):
    def __init__(self, channel, height, width, scale=1.0, device="cuda"):
        super().__init__()
        if channel % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                                "odd dimension (got dim={:d})".format(channel))
        height = int(height * scale)
        width = int(width * scale)
        self.pe = torch.zeros(channel, height, width, device=device)

        # Each dimension use half of d_model
        self.d_model = int(channel / 2)
        self.half_d = int(channel / 4)
        self.div_term = torch.exp(torch.arange(0., self.d_model, 2) * -(math.log(10000.0) / self.d_model)) / scale
        self.pos_h = torch.arange(0., height).unsqueeze(1)
        self.pos_w = torch.arange(0., width).unsqueeze(1)

    def forward(self, x, shift_h=0, shift_w=0, pe_transform=None):
        height, width = self.pos_h.shape[0], self.pos_w.shape[0]

        # Index based positional encoding
        initial_pe = torch.zeros(2,height,width)
        initial_pe[0,:,:] = self.pos_w.transpose(0,1).unsqueeze(1).repeat(1,height,1)
        initial_pe[1,:,:] = self.pos_h.transpose(0,1).unsqueeze(2).repeat(1,1,width)

        # Affine Transformation of the encoding torus
        if pe_transform is not None:
          initial_pe[0,:,:] -= int(width/2)
          initial_pe[1,:,:] -= int(height/2)
          initial_pe=torch.tensordot(pe_transform,initial_pe,1)
          initial_pe[0,:,:] += int(width/2)
          initial_pe[1,:,:] += int(height/2)
        initial_pe[0,:,:] += shift_w
        initial_pe[1,:,:] += shift_h
        initial_pe[0,:,:] = torch.remainder(initial_pe[0], width)
        initial_pe[1,:,:] = torch.remainder(initial_pe[1], height)

        # Convert to quadtree encoding
        self.pe[0:self.d_model:2,:,:] = torch.sin(torch.reshape(torch.outer(torch.reshape(initial_pe[0],(1,height*width)).squeeze(),self.div_term),(height,width,self.half_d))).permute(2,0,1)
        self.pe[1:self.d_model:2,:,:] = torch.cos(torch.reshape(torch.outer(torch.reshape(initial_pe[0],(1,height*width)).squeeze(),self.div_term),(height,width,self.half_d))).permute(2,0,1)
        self.pe[self.d_model::2,:,:] = torch.sin(torch.reshape(torch.outer(torch.reshape(initial_pe[1],(1,height*width)).squeeze(),self.div_term),(height,width,self.half_d))).permute(2,0,1)
        self.pe[self.d_model+1::2,:,:] = torch.cos(torch.reshape(torch.outer(torch.reshape(initial_pe[1],(1,height*width)).squeeze(),self.div_term),(height,width,self.half_d))).permute(2,0,1)

        return self.pe.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
