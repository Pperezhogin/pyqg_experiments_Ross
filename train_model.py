import xarray as xr
import numpy as np
from scipy.stats import pearsonr
import os
import sys
import json
import glob
sys.path.append('.')
from models import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default="/scratch/zanna/data/pyqg/64_256/train")
parser.add_argument('--test_dir', type=str, default="/scratch/zanna/data/pyqg/64_256/test")
parser.add_argument('--save_dir', type=str)
parser.add_argument('--inputs', type=str, default="u,v,q")
parser.add_argument('--target', type=str, default="q_forcing_advection")
parser.add_argument('--normalize_loss', type=int, default=0)
parser.add_argument('--zero_mean', type=int, default=1)
parser.add_argument('--l1_grads', type=float, default=0)
parser.add_argument('--mask_grads', type=int, default=0)
parser.add_argument('--grad_radius', type=int, default=6)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--scaler', type=str, default='basic')
args = parser.parse_args()

save_dir = args.save_dir
os.system(f"mkdir -p {save_dir}") 

with open(f"{save_dir}/model_config.json", 'w') as f:
    f.write(json.dumps(args.__dict__))

train_files = sorted(glob.glob(f"{args.train_dir}/*/lores.nc"))
train_datasets = [xr.open_dataset(f) for f in train_files]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

models = []

def extract_vars(qtys, datasets, z):
    return np.vstack([
      np.swapaxes(np.array([
        ds.isel(lev=z)[qty].data
        for qty in qtys
      ]),0,1)  
      for ds in datasets
    ])

def extract_xy(datasets, z):
    x = extract_vars(args.inputs.split(","), datasets, z)
    y = extract_vars(args.target.split(","), datasets, z)
    return x, y

for z in range(2):
    print(f"z={z}")
    
    X_train, Y_train = extract_xy(train_datasets, z)

    print("Extracted datasets")

    model = FullyCNN(X_train.shape[1], Y_train.shape[1])
    model_path = f"{save_dir}/model_z{z}"

    if os.path.exists(model_path):
        model.load(model_path)
        model.to(device)
    else:
        model.to(device)

        if args.scaler = 'logpow':
            X_scale = MultivariateLogPowScaler(X_train)
            Y_scale = MultivariateLogPowScaler(Y_train)
        else:
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

        print("Computed scales")

        if args.zero_mean:
            model.set_zero_mean(True)
        model.set_scales(X_scale, Y_scale)

        print("Starting fitting")
        model.fit(X_train, Y_train,
                num_epochs=args.num_epochs,
                device=device,
                mask_grads=args.mask_grads,
                grad_radius=args.grad_radius,
                #normalize_loss=args.normalize_loss,
                l1_grads=args.l1_grads)
        print("Finished fitting")

        model.cpu()
        model.save(model_path)
        model.to(device)
        print("Finished saving")

    models.append(model)

for f in glob.glob(f"{args.test_dir}/*/lores.nc"):
    ds = xr.open_dataset(f)
    run_idx = f.split("/")[-2]

    preds = []
    corrs = []
    mses = []
    for z in range(2):
        x, y = extract_xy([ds], z)
        yhat = models[z].predict(x, device=device)
        preds.append(yhat)
        corrs.append([pearsonr(yi.reshape(-1), yhi.reshape(-1))[0] for yi, yhi in zip(y, yhat)])
        mses.append([np.sum((yi-yhi)**2) for yi, yhi in zip(y, yhat)])
    preds = np.array(preds)
    corrs = np.array(corrs)
    mses = np.array(mses)

    def coord_kwargs(*dims):
        coords = {}
        for d in dims:
            coords[d] = ds[d]
        return dict(coords=coords, dims=dims)
    
    os.system(f"mkdir -p {save_dir}/test/{run_idx}")

    xr.Dataset(data_vars=dict(
        predictions=xr.DataArray(preds[:,:,0,:,:],
            attrs=dict(target=args.target),
            **coord_kwargs('lev','time','y','x')),
        correlation=xr.DataArray(corrs, **coord_kwargs('lev','time')),
        mean_sq_err=xr.DataArray(mses, **coord_kwargs('lev','time')),
    )).to_netcdf(f"{save_dir}/test/{run_idx}/preds.nc")

