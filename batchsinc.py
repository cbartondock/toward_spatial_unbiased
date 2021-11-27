import numpy as np
import scipy
import scipy.misc
import matplotlib.pyplot as plt
from scipy import ndimage
from PIL import Image
import pathlib as path
import os
import argparse

parser = argparse.ArgumentParser(description='Project whole directory')
parser.add_argument('indir',type=path.Path, help="Directory of images you want to sinc smooth")
parser.add_argument('outdir',type=path.Path,help="Destination directory for sinc smoothed images")
args = parser.parse_args()
files = os.listdir(args.indir)
for fi in files:
  img = Image.open(os.path.join(args.indir,fi)).convert('RGB')
  r, g, b = img.split()

  # take rgb channels and filter high frequencies independently, then recombine to get filtered colored image
  for i in range(1,4):
      if i == 1:
          img = r
      elif i == 2:
          img = g
      else:
          img = b
      f = np.fft.fft2(img)
      fshift = np.fft.fftshift(f) ## shift for centering 0.0 (x,y)
      magnitude_spectrum = 20*np.log(np.abs(fshift))

      # plt.subplot(121),plt.imshow(img)
      # plt.title('Input Image'), plt.xticks([]), plt.yticks([])
      # plt.subplot(122),plt.imshow(magnitude_spectrum, cmap = 'gray')
      # plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
      # plt.show()

      ## removing low frequency contents by applying a 60x60 rectangle window (for masking)
      rows = np.size(img, 0) #taking the size of the image
      cols = np.size(img, 1)
      crow, ccol = int(rows/2), int(cols/2)

      # larger w => higher threshold for filtering => less deviation from original image
      w = 60

      original = np.zeros_like(fshift)
      original[crow-w:crow+w, ccol-w:ccol+w] = fshift[crow-w:crow+w, ccol-w:ccol+w]
      f_ishift= np.fft.ifftshift(original)

      img_back = np.fft.ifft2(f_ishift) ## shift for centering 0.0 (x,y)
      img_back = np.abs(img_back)

      if i == 1:
          r_back = Image.fromarray(img_back).convert("L")
      elif i == 2:
          g_back = Image.fromarray(img_back).convert("L")
      else:
          b_back = Image.fromarray(img_back).convert("L")

  img_back = Image.merge('RGB', (r_back, g_back, b_back))
  img = Image.merge('RGB', (r, g, b))

# plt.subplot(131),plt.imshow(img, cmap = 'gray')
# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
# plt.subplot(132),plt.imshow(img_back, cmap = 'gray')
# plt.title('Removed High Freq'), plt.xticks([]), plt.yticks([])
# plt.show()
  img_back.save(os.path.join(args.outdir,fi.split('.')[0]+'_sinced.png'))
