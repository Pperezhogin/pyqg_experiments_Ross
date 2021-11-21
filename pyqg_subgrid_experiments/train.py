import xarray as xr
import numpy as np
from scipy.stats import pearsonr
import os
import sys
import json
import glob

import pyqg_subgrid_experiments as pse
from pyqg_subgrid_experiments.models import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train_set', type=str, default="/scratch/zanna/data/pyqg/data/train/*.nc")
parser.add_argument('--test_set', type=str, default="/scratch/zanna/data/pyqg/data/test/*.nc")
parser.add_argument('--transfer_set', type=str, default="/scratch/zanna/data/pyqg/data/transfer/*.nc")
parser.add_argument('--save_dir', type=str)
parser.add_argument('--inputs', type=str, default="u,v,q")
parser.add_argument('--target', type=str, default="q_forcing_advection")
parser.add_argument('--zero_mean', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--scaler', type=str, default='basic')
parser.add_argument('--layerwise_inputs', type=int, default=0)
parser.add_argument('--layerwise_targets', type=int, default=0)
args = parser.parse_args()

save_dir = args.save_dir
os.system(f"mkdir -p {save_dir}") 

with open(f"{save_dir}/model_config.json", 'w') as f:
    f.write(json.dumps(args.__dict__))

train = pse.Dataset(args.train_set)

param = pse.CNNParameterization.train_on(train, save_dir,
            layerwise_inputs=args.layerwise_inputs,
            layerwise_targets=args.layerwise_targets,
            zero_mean=args.zero_mean,
            num_epochs=args.num_epochs)

test = pse.Dataset(args.test_set)
xfer = pse.Dataset(args.transfer_set)

param.test_offline(test).to_netcdf(os.path.join(save_dir, "test.nc"))
param.test_offline(xfer).to_netcdf(os.path.join(save_dir, "transfer.nc"))

test_dir = os.path.join(save_dir, "test")
xfer_dir = os.path.join(save_dir, "transfer")

for ds, ddir in [(test, test_dir), (xfer, xfer_dir)]:
    os.system(f"mkdir -p {ddir}") 
    for i in range(25):
        run = param.run_online(**ds.pyqg_params)
        run.to_netcdf(os.path.join(ddir, f"{i}.nc"))
