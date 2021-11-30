import argparse
import os
import pathlib as path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import seaborn as sns
import math

parser = argparse.ArgumentParser(description='Histogram l2 norm differences')
parser.add_argument('indir',type=path.Path, help="Directory containing sample and target images")
parser.add_argument('--name', default="dist",help="Name of output (without extension)")
args = parser.parse_args()

transforms = {
    "rotation": {
      "prettyName": "Rotation",
      "min": 0.,
      "max": 2*math.pi
      },
    "horiz_shear": {
      "prettyName": "Horizontal shear",
      "min": -1.,
      "max": 1.
      }
}
transformdirs = [os.path.join(args.indir,x) for x in os.listdir(args.indir) if os.path.isdir(os.path.join(args.indir,x))]

for transformdir in transformdirs:
  tname = os.path.basename(os.path.normpath(transformdir))
  transform = transforms[tname]
  print(f"Computing relative L2 errors for {transform['prettyName']}.")

  samples = [os.path.join(transformdir,x) for x in os.listdir(transformdir) if x.split("_")[0]=="sample"]
  targets = [os.path.join(transformdir,x) for x in os.listdir(transformdir) if x.split("_")[0]=="target"]
  OPEsamples = [os.path.join(transformdir,x) for x in os.listdir(transformdir) if x.split("_")[0]=="OPEsample"]
  samples.sort()
  targets.sort()
  OPEsamples.sort()
  if not (len(samples) == len(targets) == len(OPEsamples)):
    print("There should be the same number of samples and OPE samples as there are targets")
    print(f"But there are {len(samples)} samples and {len(targets)} targets")
    exit()

  diffs, fracs = [], []
  OPEdiffs, OPEfracs = [], []
  for i in range(0,len(samples)):
    sample = np.array(image.imread(samples[i]))
    OPEsample = np.array(image.imread(OPEsamples[i]))
    target= np.array(image.imread(targets[i]))
    diffnorm = np.linalg.norm(np.ndarray.flatten(sample - target), 2)
    OPEdiffnorm = np.linalg.norm(np.ndarray.flatten(OPEsample - target), 2)
    targetnorm = np.linalg.norm(np.ndarray.flatten(target), 2)
    diffs.append(diffnorm)
    OPEdiffs.append(OPEdiffnorm)
    fracs.append(diffnorm / targetnorm)
    OPEfracs.append(OPEdiffnorm / targetnorm)

  sns_plot = sns.distplot(fracs, hist=True,norm_hist=True, kde=True, bins = 20, color='darkblue',hist_kws={'edgecolor':'black',"range":(0,1.5)},kde_kws={'clip':(0,1000),'linewidth': 4}, label="Affine PE")
  sns.distplot(OPEfracs, hist=True,norm_hist=True, kde=True, bins = 20, color='darkorange',hist_kws={'edgecolor':'black',"range":(0,1.5)},kde_kws={'clip':(0,1000),'linewidth': 4}, label="Original PE")


  plt.title(f"Histogram of relative L2 error\n{transform['prettyName']}")
  plt.xlabel("Relative L2 error")
  plt.ylabel("Density")
  plt.legend()
  prename = args.indir.name.lower().replace(" ","") if args.name=="dist" else args.name
  sns_plot.figure.savefig(prename + "_" + tname + ".pdf")

  plt.clf()



