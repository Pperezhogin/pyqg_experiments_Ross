# PYQG Subgrid Forcing Experiments

![pyqg simulation animation](./eddy.gif)

## Summary

This repository contains code that uses [`pyqg`](https://pyqg.readthedocs.io/en/latest/) (a quasi-geostrophic ocean simulation library) to test out machine learning parameterizations as well as generate datasets for training them. It also contains associated code for training such ML parameterizations and evaluating their offline and online performance.

## Loading simulation data

Simulation data from repeatedly running [`generate_dataset.py`](./generate_dataset.py) is currently stored at `/scratch/zanna/data/pyqg/64_256/{train,test,transfer}`.

The `train` set consists of 500 simulations from random initializations at the default configuration of `pyqg.QGModel` that results in uniformly chaotic eddies. Simulation snapshots are sampled irregularly according to an exponential interval distribution with an average of 1000 model timesteps; this timescale is chosen such that the [autocorrelation is small](./notebooks/pyqg_autocorrelation.ipynb) between samples. Within each simulation folder (e.g. `/scratch/zanna/data/pyqg/64_256/train/499/`), there are two netcdf files: `hires.nc` and `lores.nc`. Both netcdf files correspond to the same essential simulation but at different resolutions; `hires.nc` is run with 256 grid points (enough to almost fully resolve eddies) but `lores.nc` is run with only 64. Before each model timestep (or before computing diagnostics), we ensure the two simulations stay in sync by updating the initial conditions of the low-res model to the current state of the high-resolution model. After each timestep, we compute the subgrid forcing for the low resolution model, both in potential vorticity and velocity. To access training data, you can look at the potential vorticity `q`, the velocities `u` and `v`, and subgrid forcings, stored under `q_forcing_advection`, `u_forcing_advection`, and `v_forcing_advection`, all in `lores.nc`.

The `test` set is similar, but consists of only 50 simulations, this time regularly sampled (so it's possible to run e.g. `xr.load_mfdataset("/scratch/zanna/data/pyqg/64_256/test/*/lores.nc", combine="nested", concat_dim="run")`. The test set contains `lores.nc` and `hires.nc`, but also `control.nc`, which stores the results of running an unparameterized 64-grid point simulation, as well as `physical.nc`, which stores the results of running a 64-grid point simulation with a physical parameterization.

Finally, the `transfer` set is similar to test, but consists of simulations run at a very different configuration of pyqg that results in long-range jets.

To load the test and transfer datasets, the quickest oneliner would be:

```python
test = xr.load_mfdataset("/scratch/zanna/data/pyqg/64_256/test/*/lores.nc", combine="nested", concat_dim="run")
xfer = xr.load_mfdataset("/scratch/zanna/data/pyqg/64_256/transfer/*/lores.nc", combine="nested", concat_dim="run")
```

For the training set, this won't work because each simulation is sampled at different time intervals. Instead, you can use the following helpers:

```
from generate_dataset import PYQGSubgridDataset

train = PYQGSubgridDataset("/scratch/zanna/data/pyqg/64_256/train")

q1 = train.extract_variable("q", z=0)
q2 = train.extract_variable("q", z=1)

X_for_ml_model = train.extract_variables(["u","v","q"], z=0)
Y_for_ml_model = train.extract_variables(["q_forcing_advection"], z=0)
```

A demo of loading simulation data (as well as trained models) is available at [`notebooks/data-and-model-loading-demo.ipynb`](./notebooks/data-and-model-loading-demo.ipynb)
