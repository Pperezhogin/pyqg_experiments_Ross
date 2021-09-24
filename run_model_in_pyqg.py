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
parser.add_argument('--save_dir', type=str)
parser.add_argument('--divide_by_dt', action='store_true', default=False)
parser.add_argument('--inputs', type=str, default="u,v,q")
parser.add_argument('--run_idx', type=int)
args = parser.parse_args()

save_dir = args.save_dir

cnn0 = FullyCNN(len(args.inputs.split(",")), 1)
cnn1 = FullyCNN(len(args.inputs.split(",")), 1)

cnn0.load(f"{save_dir}/model_z0")
cnn1.load(f"{save_dir}/model_z1")

run_dir = os.path.join(save_dir, "parameterized_pyqg_runs")
os.system(f"mkdir -p {run_dir}")

i = args.run_idx
run = generate_parameterized_dataset(cnn0, cnn1, args.inputs, divide_by_dt=args.divide_by_dt)
complex_vars = [k for k,v in run.variables.items() if v.dtype == np.complex128]
run.drop(complex_vars).to_netcdf(os.path.join(run_dir, f"{i}.nc"))
