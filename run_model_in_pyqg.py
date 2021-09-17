import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('.')
from models import *
from pyqg_subgrid_dataset import generate_parameterized_dataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="/scratch/zanna/data/pyqg/pyqg_runs")
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--inputs', type=str, default="u,v,q")
parser.add_argument('--target', type=str, default="q_forcing_advection")
parser.add_argument('--model', type=str, default="fully_cnn")
parser.add_argument('--n_runs', type=int, default=16)
args = parser.parse_args()

if args.save_dir == '':
    save_dir = f"{args.data_dir}/{args.model}_{args.target}"
else:
    save_dir = args.save_dir

if args.model == 'basic_cnn':
    cnn0 = BasicCNN((len(args.inputs.split(",")),64,64), (1,64,64))
    cnn1 = BasicCNN((len(args.inputs.split(",")),64,64), (1,64,64))
elif args.model == 'fully_cnn':
    cnn0 = FullyCNN(len(args.inputs.split(",")), 1)
    cnn1 = FullyCNN(len(args.inputs.split(",")), 1)
else:
    assert False

cnn0.load(f"{save_dir}/model_z0")
cnn1.load(f"{save_dir}/model_z1")

run_dir = os.path.join(save_dir, "parameterized_pyqg_runs")
os.system(f"mkdir -p {run_dir}")

for i in range(args.n_runs):
    run = generate_parameterized_dataset(cnn0, cnn1, args.inputs)
    complex_vars = [k for k,v in run.variables.items() if v.dtype == np.complex128]
    run.drop(complex_vars).to_netcdf(os.path.join(run_dir, f"{i}.nc"))
