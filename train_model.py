import xarray as xr
import numpy as np

import os
import sys
sys.path.append('.')
from models import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default="/scratch/zanna/data/pyqg/pyqg_runs")
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--inputs', type=str, default="u,v,q")
parser.add_argument('--target', type=str, default="q_forcing_advection")
parser.add_argument('--model', type=str, default="fully_cnn")
args = parser.parse_args()

if len(args.save_dir):
    save_dir = args.save_dir
else:
    save_dir = f"{args.data_dir}/{args.model}_{args.target}"

os.system(f"mkdir -p {save_dir}") 

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
    elif args.model == 'fully_cnn':
        model = FullyCNN(X_train.shape[1], Y_train.shape[1])
    else:
        assert False

    model.set_scales(X_scale, Y_scale)
    model.fit(X_train, Y_train, num_epochs=25)
    print("Finished fitting")

    model_path = f"{save_dir}/model_z{z}"
    model.save(model_path)
    print("Finished saving")

    mse = model.mse(X_test, Y_test)
    print(f"MSE: {mse}")
    with open(f"{model_path}_mse.txt", 'w') as f:
        f.write(str(mse))
