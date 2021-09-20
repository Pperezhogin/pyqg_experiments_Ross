import xarray as xr
import numpy as np
from scipy.stats import pearsonr
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
parser.add_argument('--zero_mean', type=int, default=1)
args = parser.parse_args()

if len(args.save_dir):
    save_dir = args.save_dir
else:
    save_dir = f"{args.data_dir}/{args.model}_{args.target}"

os.system(f"mkdir -p {save_dir}") 

ds = xr.open_mfdataset(f"{args.data_dir}/*/lores.nc", combine="nested", concat_dim="run")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models = []

run_length = len(ds.isel(run=0).time)
run_count = len(ds.run)
cutoff = int(run_count * (3/4)) * run_length

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

    if args.zero_mean:
        Y_scale = BasicScaler(0, Y_train.std())
    else:
        Y_scale = BasicScaler(Y_train.mean(), Y_train.std())

    if args.model == 'basic_cnn':
        model = BasicCNN(X_train.shape[1:], Y_train.shape[1:])
    elif args.model == 'fully_cnn':
        model = FullyCNN(X_train.shape[1], Y_train.shape[1])
    else:
        assert False

    model.to(device)

    if args.zero_mean:
        model.set_zero_mean(True)
    model.set_scales(X_scale, Y_scale)
    model.fit(X_train, Y_train, num_epochs=100, device=device)
    print("Finished fitting")

    model_path = f"{save_dir}/model_z{z}"
    model.cpu()
    model.save(model_path)
    model.to(device)
    print("Finished saving")

    models.append(model)

preds = []
corrs = []
mses = []
for i in range(len(ds.run)):
    z_preds = []
    z_corrs = []
    z_mses = []
    for z in range(2):
        x = np.swapaxes(np.array([
            ds.isel(run=i,lev=z)[inp].data
            for inp in args.inputs.split(",")
        ]),0,1)  
        y = np.array(ds.isel(run=i,lev=z)[args.target].data)[:,np.newaxis,:,:]
        yhat = models[z].predict(x, device=device)
        z_preds.append(yhat)
        z_corrs.append([pearsonr(yi.reshape(-1), yhi.reshape(-1))[0] for yi, yhi in zip(y, yhat)])
        z_mses.append([np.sum((yi-yhi)**2) for yi, yhi in zip(y, yhat)])
    preds.append(z_preds)
    corrs.append(z_corrs)
    mses.append(z_mses)
preds = np.array(preds)
corrs = np.array(corrs)
mses = np.array(mses)

dims = ['run','lev','time','y','x']
coords = {}
for d in dims:
    coords[d] = ds[d]

preds_ds = xr.Dataset(data_vars=dict(
    predictions=xr.DataArray(preds[:,:,:,0,:,:], coords=coords, dims=dims, attrs=dict(target=args.target)),
    correlation=xr.DataArray(corrs, coords=dict(run=ds.run, lev=ds.lev, time=ds.time), dims=['run','lev','time']),
    mean_sq_err=xr.DataArray(mses, coords=dict(run=ds.run, lev=ds.lev, time=ds.time), dims=['run','lev','time']),
))

preds_ds.to_netcdf(f"{save_dir}/predictions.nc")
