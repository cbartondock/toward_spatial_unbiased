import argparse

import torch
from torchvision import utils
from model import Generator
from tqdm import tqdm
import math

import os
import subprocess
import numpy as np
transforms = {
    "rotation": {
    "method": lambda theta: torch.tensor([[math.cos(theta), math.sin(theta)], [-math.sin(theta),math.cos(theta)]]),
    "min": 0.,
    "max": 2*math.pi
    },
    "horiz_shear": {
      "method": lambda alpha: torch.tensor([[1.,alpha],[0.,1.]]),
      "min": -1.,
      "max": 1.
    },
    "vert_shear": {
      "method": lambda alpha: torch.tensor([[1.,0.],[alpha,1.]]),
      "min": -1.,
      "max": 1.
    },
    "horiz_stretch": {
      "method": lambda alpha: torch.tensor([[alpha,0.],[0.,1.]]),
      "min": -2.,
      "max": 2.
    },
    "vert_stretch": {
      "method": lambda alpha: torch.tensor([[1.,0.],[0.,alpha]]),
      "min": -2.,
      "max": 2.
    },
}

def gifs(args):
  if not os.path.exists(f"gif/{args.name}"):
    os.makedirs(f"gif/{args.name}")
  ops = list(transforms.keys())
  ops.append("translation")
  cwd = os.path.dirname(os.path.realpath(__file__))
  for tname in ops:
    print(f"Creating a .gif for {tname}.")
    command = f"convert -delay 8 -loop 0 generate/{args.name}/{tname}/*.png gif/{args.name}/{tname}.gif"
    subprocess.call(command, shell=True, cwd=cwd)

def generate(args, g_ema, device, mean_latent):
    radius = 32
    with torch.no_grad():
        g_ema.eval()
        for tname in transforms.keys():
          range_min = transforms[tname]["min"]
          range_max = transforms[tname]["max"]
          step = (range_max-range_min) / args.pics
          transforms[tname]["range"] = np.arange(range_min, range_max, step)
          if not os.path.exists(f"generate/{args.name}/{tname}"):
            os.makedirs(f"generate/{args.name}/{tname}")
        for tname, transform in transforms.items():
          sample_z = torch.randn(args.sample, args.latent, device=device)
          print(f"Generating {tname} images:")
          for i in tqdm(range(args.pics)):
            param = transform["range"][i]
            pe_transform = transform["method"](param).float()
            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent,pe_transform=pe_transform)
            for j in range(args.sample):
                utils.save_image(
                    sample[j].unsqueeze(0),
                    f"generate/{args.name}/{tname}/{str(j).zfill(3)}_{str(i).zfill(4)}.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )
        if not os.path.exists(f"generate/{args.name}/translation"):
          os.makedirs(f"generate/{args.name}/translation")
        sample_z = torch.randn(args.sample, args.latent, device=device)

        print("Generating translation images:")
        for i in tqdm(range(args.pics)):
            dh = math.sin(2 * math.pi * (i / args.pics)) * radius
            dw = math.cos(2 * math.pi * (i / args.pics)) * radius
            sample, _ = g_ema(
                [sample_z], truncation=args.truncation, truncation_latent=mean_latent, shift_h=dh, shift_w=dw)

            for j in range(args.sample):
                utils.save_image(
                    sample[j].unsqueeze(0),
                    f"generate/{args.name}/translation/{str(j).zfill(3)}_{str(i).zfill(4)}.png",
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="Generate samples from the generator")

    parser.add_argument(
        "--size", type=int, default=256, help="output image size of the generator"
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=4,
        help="number of samples to be generated for each image",
    )
    parser.add_argument(
        "--pics", type=int, default=100, help="number of images to be generated"
    )
    parser.add_argument("--truncation", type=float, default=1, help="truncation ratio")
    parser.add_argument(
        "--truncation_mean",
        type=int,
        default=4096,
        help="number of vectors to calculate mean for the truncation",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="550000.pt",
        help="path to the model checkpoint",
    )
    parser.add_argument(
        "--channel_multiplier",
        type=int,
        default=2,
        help="channel multiplier of the generator. config-f = 2, else = 1",
    )
    parser.add_argument('--name', type=str, default='test')
    parser.add_argument('--position', type=str, default='mspe', help='pe options (none | sspe | mspe | rand_mspe)')
    parser.add_argument('--kernel_size', type=int, default=3)
    parser.add_argument('--gifs', action='store_true')
    args = parser.parse_args()


    args.latent = 512
    args.n_mlp = 8

    g_ema = Generator(
        args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier, position=args.position, kernel_size=args.kernel_size, scale=1.0, device=device,
    ).to(device)
    checkpoint = torch.load(f'checkpoint/{args.name}/{args.ckpt}')

    g_ema.load_state_dict(checkpoint["g_ema"], strict=False)

    if args.truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(args.truncation_mean)
    else:
        mean_latent = None

    generate(args, g_ema, device, mean_latent)

    if args.gifs:
      gifs(args)
