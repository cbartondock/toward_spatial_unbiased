import argparse
import os
import pathlib as path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import matplotlib.patches as mpatches
import seaborn as sns
import math
import pandas as pd

parser = argparse.ArgumentParser(description='Histogram l2 norm differences')
parser.add_argument('indir',type=path.Path, help="Directory containing sample and target images")
parser.add_argument('--name', default="dist",help="Name of output (without extension)")
args = parser.parse_args()

opacity=.5
ticks=8
#ticks=3
transforms = {
    "rotation": {
      "prettyName": "Rotation",
      "parameter": "Theta (Radians)",
      "xticks": ['-\u03c0','-3\u03c0/4','-\u03c0/2','-\u03c0/4','0','\u03c0/4','\u03c0/2','3\u03c0/4','\u03c0'],
      #"xticks": ['0','2\u03c0/3','4\u03c0/3'],
      #"shift": 0
      "shift": 10
      },
    "horiz_shear": {
      "prettyName": "Horizontal shear",
      "parameter": "Alpha",
      #"xticks": ['-1','-1/3','1/3'],
      "xticks": ['-1','-.75','-.5','-.25','0','.25','.5','.75','1'],
      "shift": 0
      }
}

cycle = lambda shift: lambda l: l[-shift:]+l[:-shift]

transformdirs = [os.path.join(args.indir,x) for x in os.listdir(args.indir) if os.path.isdir(os.path.join(args.indir,x))]

for transformdir in transformdirs:
  tname = os.path.basename(os.path.normpath(transformdir))
  transform = transforms[tname]
  print(f"Computing relative l2 errors for {transform['prettyName']}.")

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

  fracs = {}
  OPEfracs = {}
  for i in range(len(samples)):
    #  print(f"Sample: {samples[i]}")
    #  print(f"OPESample: {OPEsamples[i]}")
    #  print(f"Target: {targets[i]}")
    #  print("\n")
    k = str(int(os.path.basename(samples[i]).split("_")[-1].split('.')[0]))
    if not k in fracs:
      fracs[k] = []
    if not k in OPEfracs:
      OPEfracs[k] = []
    sample = np.array(image.imread(samples[i]))
    OPEsample = np.array(image.imread(OPEsamples[i]))
    target= np.array(image.imread(targets[i]))
    diffnorm = np.linalg.norm(np.ndarray.flatten(sample - target), 2)
    OPEdiffnorm = np.linalg.norm(np.ndarray.flatten(OPEsample - target), 2)
    targetnorm = np.linalg.norm(np.ndarray.flatten(target),2)
    fracs[k].append(diffnorm / targetnorm)
    OPEfracs[k].append( OPEdiffnorm / targetnorm )
  shift = transforms[tname]["shift"]
  OPEfracs_data = cycle(shift)([np.array(OPEfracs[(str(k))]) for k in range(len(fracs))])
  fracs_data = cycle(shift)([np.array(fracs[str(k)]) for k in range(len(fracs))])
  sns_plot = sns.pointplot(data=fracs_data,color='darkblue', plot_kws=dict(alpha=opacity))
  plt.setp(sns_plot.collections,alpha=opacity)
  plt.setp(sns_plot.lines,alpha=opacity)
  sns_plot2 = sns.pointplot(data=OPEfracs_data,color='darkorange', plot_kws=dict(alpha=opacity))
  plt.setp(sns_plot2.collections,alpha=opacity)
  plt.setp(sns_plot2.lines,alpha=opacity)

  plt.title(f"Test\n {transform['prettyName']}")
  plt.xlabel(transforms[tname]["parameter"])
  plt.xticks(np.arange(0,len(fracs)+1,(len(fracs))/ticks), transforms[tname]["xticks"])
  #plt.xticks([0,1,2],transforms[tname]["xticks"])
  plt.ylabel("Relative L2 Error")
  bluepatch = mpatches.Patch(color='darkblue',label='Affine PE')
  orangepatch = mpatches.Patch(color='darkorange',label='Original PE')
  plt.legend(handles=[bluepatch,orangepatch])
  sns_plot.figure.savefig("test_"+tname+".pdf")
  plt.clf()
