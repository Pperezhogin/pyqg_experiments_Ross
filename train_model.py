import xarray as xr
import numpy as np
from scipy.stats import pearsonr
import os
import sys
import json
import glob
dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dirname)
from models import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--train_dir', type=str, default="/scratch/zanna/data/pyqg/datasets/train")
parser.add_argument('--test_dir', type=str, default="/scratch/zanna/data/pyqg/datasets/test")
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
parser.add_argument('--skip_datasets', type=int, default=0)
parser.add_argument('--use_both_layers_as_input', type=int, default=0)
args = parser.parse_args()

save_dir = args.save_dir
os.system(f"mkdir -p {save_dir}") 

with open(f"{save_dir}/model_config.json", 'w') as f:
    f.write(json.dumps(args.__dict__))

train = xr.open_mfdataset(f"{args.train_dir}/*/lores.nc", combine="nested", concat_dim="run")
num_datasets = len(train.run)
train = train.isel(run=slice(args.skip_datasets, None))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
models = []

for z in range(2):
    print(f"z={z}")

    targets = [(feat, z) for feat in args.target.split(",")]
    if args.use_both_layers_as_input:
        inputs = [(feat, zi) for feat in args.inputs.split(",") for zi in range(2)]
    else:
        inputs = [(feat, z) for feat in args.inputs.split(",")]

    model = FullyCNN(inputs, targets)
    X_train = model.extract_inputs_from_netcdf(train)
    Y_train = model.extract_targets_from_netcdf(train)

    print("Extracted datasets")

    model_path = f"{save_dir}/model_z{z}"

    if os.path.exists(model_path):
        model.load(model_path)
        model.to(device)
    else:
        model.to(device)

        if args.scaler == 'logpow':
            X_scale = MultivariateLogPowScaler(X_train)
            Y_scale = MultivariateLogPowScaler(Y_train)
        else:
            X_scale = BasicScaler(
                mu=np.array([
                    X_train[:,i].mean() for i in range(X_train.shape[1])
                ])[np.newaxis,:,np.newaxis,np.newaxis],

                sd=np.array([
                    X_train[:,i].std() for i in range(X_train.shape[1])
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

        num_epochs = args.num_epochs
        if args.skip_datasets > 0:
            num_epochs = int(num_epochs * (num_datasets / args.skip_datasets))

        print("Starting fitting")
        model.fit(X_train, Y_train,
                num_epochs=num_epochs,
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
    run_idx = f.split("/")[-2]
    if os.path.exists(f"{save_dir}/test/{run_idx}/preds.nc"): continue

    ds = xr.open_dataset(f).expand_dims('run')
    preds = []
    corrs = []
    mses = []
    for z in range(2):
        x = models[z].extract_inputs_from_netcdf(ds)
        y = models[z].extract_targets_from_netcdf(ds)
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

from generate_dataset import *

year = 24*60*60*360.

paramsets = [
    dict(),
    dict(rek=7.000000e-08, delta=0.1, beta=1.0e-11)
]

cnn0, cnn1 = models

for j, params in enumerate(paramsets):
    run_dir = os.path.join(save_dir, "pyqg_runs/paramsets", str(j))
    os.system(f"mkdir -p {run_dir}")

    with open(f"{run_dir}/pyqg_params.json", 'w') as f:
        f.write(json.dumps(params))

    for i in range(5):
        run = generate_parameterized_dataset(cnn0, cnn1, **params)
        complex_vars = [k for k,v in run.variables.items() if v.dtype == np.complex128]
        run.drop(complex_vars).to_netcdf(os.path.join(run_dir, f"{i}.nc"))

    m1 = initialize_pyqg_model(nx=256, **params)
    m1.run()

    m2 = initialize_parameterized_model(cnn0, cnn1, **params)

    corrs, steps = time_until_uncorrelated(m1, m2)

    with open(f"{run_dir}/decorrelation_timesteps", 'w') as f:
        f.write(str(steps))
