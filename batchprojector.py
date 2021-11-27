import pathlib as path
import os
import argparse
parser = argparse.ArgumentParser(description='Project whole directory')
parser.add_argument('indir',type=path.Path, help="Directory of images you want to project")
parser.add_argument('--batchsize',type=int,default=1,help=("Number of images to pass to projector at once (lower to avoid gpu memory issues)"))
parser.add_argument('--exp_name',default="downloaded",help="Experiment name. Images will be saved in inverse/exp_name and specified checkpoint should be in checkpoints/exp_name")
args = parser.parse_args()
files = [os.path.join(args.indir, f) for f in os.listdir(args.indir)]
files = [f for f in files if not os.path.isdir(f)]

chunk = lambda l, n: [l[i:i+n] for i in range(0,len(l),n)]
for batch in chunk(files, args.batchsize):
  joinedfiles = ' '.join(batch)
  os.system('python projector.py --name' + ' ' + args.exp_name + ' ' + '--w_plus --pics 1 --ckpt 750000.pt --position mspe' + ' ' + joinedfiles)

# For UMIACS
#os.system(' '.join('srun python projector.py --name downloaded --w_plus --ckpt 750000.pt --position mspe',files))
