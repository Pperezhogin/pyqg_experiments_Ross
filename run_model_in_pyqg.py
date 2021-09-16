import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('.')
from models import *
from pyqg_subgrid_dataset import generate_parameterized_dataset

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="/scratch/zanna/data/pyqg/pyqg_runs")
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--inputs', type=str, default="q")
parser.add_argument('--model', type=str, default="basic_cnn")
parser.add_argument('--n_runs', type=int, default=16)
args = parser.parse_args()

if args.save_dir == '':
    save_dir = f"{args.data_dir}/{args.model}"
else:
    save_dir = args.save_dir

if args.model == 'basic_cnn':
    cnn0 = BasicCNN((len(args.inputs.split(",")),64,64), (1,64,64))
    cnn1 = BasicCNN((len(args.inputs.split(",")),64,64), (1,64,64))
else:
    assert False

cnn0.load(f"{save_dir}/model_z0")
cnn1.load(f"{save_dir}/model_z1")

runs =  xr.concat(
    [generate_parameterized_dataset(cnn0, cnn1, args.inputs) for _ in range(args.n_runs)], "run")

runs.to_netcdf(os.path.join(save_dir, "parameterized_pyqg_runs.nc"))
