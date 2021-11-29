import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import lpips
from model import Generator
from torchvision import utils
import skimage.transform as sk
from skimage import img_as_ubyte
import math
import numpy as np

petransforms = {
    "rotation": {
    "method": lambda theta: torch.tensor([[math.cos(theta),math.sin(theta)],[-math.sin(theta),math.cos(theta)]]),
    "skmethod": lambda theta: np.array([[math.cos(theta), math.sin(theta),0.], [-math.sin(theta),math.cos(theta),0.],[0.,0.,1.]]),
    "min": 0.,
    "max": 2*math.pi
    },
    "horiz_shear": {
      "method": lambda alpha: torch.tensor([[1.,alpha],[0.,1.]]),
      "skmethod": lambda alpha: np.array([[1.,alpha,0.],[0.,1.,0.],[0.,0.,1.]]),
      "min": -1.,
      "max": 1.
    }
}

def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    parser.add_argument(
        "--ckpt", type=str, default="550000.pt", help="path to the model checkpoint"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--noise", type=float, default=0.05, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument("--step", type=int, default=1000, help="optimize iterations")
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=1e3,
        help="weight of the noise regularization",
    )
    parser.add_argument("--mse", type=float, default=0, help="weight of the mse loss")
    parser.add_argument(
        "--w_plus",
        action="store_true",
        help="allow to use distinct latent codes to each layers",
    )
    parser.add_argument(
        "files", metavar="FILES", nargs="+", help="path to image files to be projected"
    )
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--position', type=str, default='mspe', help='pe options (none | sspe | mspe | rand_mspe)')
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument("--pics", type=int, default=100, help="number of images to be generated")
    parser.add_argument("--do_translations",action="store_true")

    args = parser.parse_args()

    n_mean_latent = 10000
    radius = 32

    resize = min(args.size, 256)

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    g_ema = Generator(args.size, 512, 8, position=args.position, kernel_size=args.kernel_size, scale=1.0, device=device)
    g_ema.load_state_dict(torch.load(f'checkpoint/{args.name}/{args.ckpt}')["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5
        for tname in petransforms.keys():
          range_min = petransforms[tname]["min"]
          range_max = petransforms[tname]["max"]
          step = (range_max-range_min) / args.pics
          petransforms[tname]["range"] = np.arange(range_min, range_max, step)
          if not os.path.exists(f"inverse/{args.name}/{tname}"):
            os.makedirs(f"inverse/{args.name}/{tname}")

    raw_imgs = []
    imgs = []
    for imgfile in args.files:
        raw_img = Image.open(imgfile).convert("RGB")
        raw_imgs.append(np.array(raw_img))
        img = transform(raw_img)
        imgs.append(img)

    imgs = torch.stack(imgs, 0).to(device)


    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )
    center=128
    pre = sk.SimilarityTransform(translation=-center)
    post = sk.SimilarityTransform(translation=center)

    for tname, pe_transform_wrapper in petransforms.items():
      for k in range(args.pics):
        param = pe_transform_wrapper["range"][k]
        pe_transform = pe_transform_wrapper["method"](param).float()
        sk_transform = pe_transform_wrapper["skmethod"](param)
        imgs_ = []
        for i in range(len(imgs)):
          affine = sk.SimilarityTransform(sk_transform)
          full_transform = pre + affine + post
          imgs_.append(transform(Image.fromarray(img_as_ubyte(sk.warp(raw_imgs[i],full_transform.params,mode="wrap"))).convert("RGB")))
        imgs_ = torch.stack(imgs_,0).to(device)

        #### initialize w and noise
        noises_single = g_ema.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

        if args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

        latent_in.requires_grad = True

        for noise in noises:
            noise.requires_grad = True

        optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

        pbar = tqdm(range(args.step))
        latent_path = []
        ### optimize
        for i in pbar:
            t = i / args.step
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr
            noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())

            img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises, pe_transform=pe_transform)

            batch, channel, height, width = img_gen.shape

            if height > 256:
                factor = height // 256

                img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img_gen = img_gen.mean([3, 5])

            p_loss = percept(img_gen, imgs_).sum()
            n_loss = noise_regularize(noises)
            mse_loss = F.mse_loss(img_gen, imgs_)

            loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize_(noises)

            if (i + 1) % 100 == 0:
                latent_path.append(latent_in.detach().clone())

            pbar.set_description(
                (
                    f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                    f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                )
            )

        img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises, pe_transform=pe_transform)

        utils.save_image(
            imgs_,
            f"inverse/{args.name}/{tname}/target_{args.files[0].split('/')[-1].split('.')[0]}_{str(k).zfill(3)}.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )
        utils.save_image(
            img_gen,
            f"inverse/{args.name}/{tname}/sample_{args.files[0].split('/')[-1].split('.')[0]}_{str(k).zfill(3)}.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )
        #### initialize w and noise
        noises_single = g_ema.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

        if args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

        latent_in.requires_grad = True

        for noise in noises:
            noise.requires_grad = True

        optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

        pbar = tqdm(range(args.step))
        latent_path = []
        ### optimize
        for i in pbar:
            t = i / args.step
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr
            noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())

            img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

            batch, channel, height, width = img_gen.shape

            if height > 256:
                factor = height // 256

                img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img_gen = img_gen.mean([3, 5])

            p_loss = percept(img_gen, imgs_).sum()
            n_loss = noise_regularize(noises)
            mse_loss = F.mse_loss(img_gen, imgs_)

            loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize_(noises)

            if (i + 1) % 10 == 0:
                latent_path.append(latent_in.detach().clone())

            pbar.set_description(
                (
                    f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                    f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                )
            )

        img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises, pe_transform=pe_transform)
        utils.save_image(
            img_gen,
            f"inverse/{args.name}/{tname}/OPEsample_{args.files[0].split('/')[-1].split('.')[0]}_{str(k).zfill(3)}.png",
            nrow=1,
            normalize=True,
            range=(-1, 1),
        )

    if args.do_translations:
      if not os.path.exists(f"inverse/{args.name}/translation"):
        os.makedirs(f"inverse/{args.name}/translation")
      ####### shift in circle ######
      for k in range(args.pics):
          dh = int(math.sin(2 * math.pi * (k / args.pics)) * radius)
          dw = int(math.cos(2 * math.pi * (k / args.pics)) * radius)
          imgs_ = torch.roll(imgs, dh, 2)
          imgs_ = torch.roll(imgs, dw, 3)

          #### initialize w and noise
          noises_single = g_ema.make_noise()
          noises = []
          for noise in noises_single:
              noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

          latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

          if args.w_plus:
              latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

          latent_in.requires_grad = True

          for noise in noises:
              noise.requires_grad = True

          optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

          pbar = tqdm(range(args.step))
          latent_path = []

          ### optimize
          for i in pbar:
              t = i / args.step
              lr = get_lr(t, args.lr)
              optimizer.param_groups[0]["lr"] = lr
              noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
              latent_n = latent_noise(latent_in, noise_strength.item())

              img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises, shift_h=dh, shift_w=dw)

              batch, channel, height, width = img_gen.shape

              if height > 256:
                  factor = height // 256

                  img_gen = img_gen.reshape(
                      batch, channel, height // factor, factor, width // factor, factor
                  )
                  img_gen = img_gen.mean([3, 5])

              p_loss = percept(img_gen, imgs_).sum()
              n_loss = noise_regularize(noises)
              mse_loss = F.mse_loss(img_gen, imgs_)

              loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

              noise_normalize_(noises)

              if (i + 1) % 100 == 0:
                  latent_path.append(latent_in.detach().clone())

              pbar.set_description(
                  (
                      f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                      f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                  )
              )

          img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises, shift_h=dh, shift_w=dw)

          utils.save_image(
              imgs_,
              f"inverse/{args.name}/translation/target_{args.files[0].split('/')[-1].split('.')[0]}_{str(k).zfill(3)}.png",
              #f"inverse/{args.name}/target_{str(k).zfill(4)}.png",
              nrow=1,
              normalize=True,
              range=(-1, 1),
          )
          utils.save_image(
              img_gen,
              f"inverse/{args.name}/translation/sample_{args.files[0].split('/')[-1].split('.')[0]}_{str(k).zfill(3)}.png",
              #f"inverse/{args.name}/sample_{str(k).zfill(4)}.png",
              nrow=1,
              normalize=True,
              range=(-1, 1),
          )
