import xarray as xr
import numpy as np

import sys
sys.path.append('.')
from models import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="/scratch/zanna/data/pyqg/pyqg_runs")
parser.add_argument('--inputs', type=str, default="u,v")
parser.add_argument('--target', type=str, default="q_forcing_advection")
parser.add_argument('--model', type=str, default="basic_cnn")
parser.add_argument('--save_path', type=str, default='')
args = parser.parse_args()

if args.save_path == '':
    save_path = f"{args.data_dir}/{args.model}"
else:
    save_path = args.save_path

ds = xr.open_mfdataset(f"{args.data_dir}/*/lores.nc", combine="nested", concat_dim="run")

for z in range(2):
    print(f"z={z}")
    X = np.vstack([
      np.swapaxes(np.array([
        ds.isel(run=i,lev=z)[inp].data
        for inp in args.inputs.split(",")
      ]),0,1)  
      for i in range(len(ds.coords['run']))
    ])

    Y = np.vstack([
      [ds.isel(run=i,lev=z)[args.target].data]
      for i in range(len(ds.coords['run']))
    ]).reshape(-1,1,64,64)

    run_length = len(ds.isel(run=0).time)
    run_count = len(ds.run)
    cutoff = int(run_count * (3/4)) * run_length
    X_train = X[:cutoff]
    Y_train = Y[:cutoff]
    X_test = X[cutoff:]
    Y_test = Y[cutoff:]

    X_scale = BasicScaler(
        mu=np.array([
            X_train[:,i].mean() for i,_ in enumerate(args.inputs.split(","))
        ])[np.newaxis,:,np.newaxis,np.newaxis],

        sd=np.array([
            X_train[:,i].std() for i,_ in enumerate(args.inputs.split(","))
        ])[np.newaxis,:,np.newaxis,np.newaxis]
    )

    Y_scale = BasicScaler(Y_train.mean(), Y_train.std())

    if args.model == 'basic_cnn':
        model = BasicCNN(X_train.shape[1:], Y_train.shape[1:])
    else:
        assert False

    model_path = f"{save_path}_z{z}"
    model.set_scales(X_scale, Y_scale)
    model.fit(X_train, Y_train, num_epochs=100)
    model.save(model_path)

    mse = model.mse(X_test, Y_test)
    print(f"MSE: {mse}")
    with open(f"{model_path}_mse.txt", 'w') as f:
        f.write(mse)
