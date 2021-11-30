import argparse
import os
import pathlib as path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
import seaborn as sns

parser = argparse.ArgumentParser(description='Histogram l2 norm differences')
parser.add_argument('indir',type=path.Path, help="Directory containing sample and target images")
parser.add_argument('--name', default="dist",help="Name of output (without extension)")
args = parser.parse_args()

samples = [os.path.join(args.indir,x) for x in os.listdir(args.indir) if x.split("_")[0]=="sample"]
targets = [os.path.join(args.indir,x) for x in os.listdir(args.indir) if x.split("_")[0]=="target"]
samples.sort()
targets.sort()
if len(samples) != len(targets):
  print("There should be the same number of samples as there are targets")
  print(f"But there are {len(samples)} samples and {len(targets)} targets")
  exit()

diffs, fracs = [], []
for i in range(0,len(samples)):
  sample= np.array(image.imread(samples[i]))
  target= np.array(image.imread(targets[i]))
  diffnorm = np.linalg.norm(np.ndarray.flatten(sample-target),2)
  frac = diffnorm / np.linalg.norm(np.ndarray.flatten(target),2)
  diffs.append(diffnorm)
  fracs.append(frac)

#  fig, ax = plt.subplots(1,1)
#  ax.hist(fracs,bins=20)
#  ax.set_title("Histogram of relative l2 error\nClean Training Sinced Test")
#  plt.xlim([0,1])
#  plt.savefig("hist.pdf")

sns_plot = sns.distplot(fracs, hist=True,norm_hist=True, kde=True, bins = 20, color='darkblue',hist_kws={'edgecolor':'black',"range":(0,3)},kde_kws={'clip':(0,1000),'linewidth': 4})
plt.title(f"Histogram of relative l2 error\n{args.indir.name}")
plt.xlabel("Relative l2 error")
plt.ylabel("Density")
sns_plot.figure.savefig((args.indir.name.lower().replace(" ","") if args.name=="dist" else args.name) + ".pdf")




