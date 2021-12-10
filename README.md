# PYQG Subgrid Forcing Experiments

![pyqg simulation animation](./eddy.gif)

## Summary

This repository contains code that uses [`pyqg`](https://pyqg.readthedocs.io/en/latest/) (a quasi-geostrophic ocean simulation library) to test out machine learning parameterizations as well as generate datasets for training them. It also contains associated code for training such ML parameterizations and evaluating their offline and online performance.

## Installation

1. Ensure you've installed Python 3.x
1. Ensure you've installed the latest development version of `pyqg` using `pip install git+https://github.com/pyqg/pyqg.git`
1. Ensure you've installed the other [requirements](./requirements.txt) (`numpy`, `scipy`, `xarray`, and `torch` if you want neural network helpers / `matplotlib` if you want plot helpers)
1. Clone this repository
1. Run `pip install --editable .`

More detailed version requirements will be determined soon.

## Usage

**NOTE: this is still in draft form, and subject to change.**

```python
import pyqg_subgrid_experiments as pse

# Load a dataset (thin wrapper around xarray.Dataset)
forcing_eddy_train = pse.Dataset("/scratch/zanna/pyqg/data/train/*.nc")
forcing_eddy_test = pse.Dataset("/scratch/zanna/pyqg/data/test/*.nc")
forcing_jets_test = pse.Dataset("/scratch/zanna/pyqg/data/transfer/*.nc")

# Variables directly included
forcing_eddy_train.q_forcing_advection # shape = (n_runs, n_steps, 2, 64, 64)
forcing_eddy_train.u.shape
forcing_eddy_train.v.shape

# Differentiation in spectral space
forcing_eddy_train.ddx('u')
forcing_eddy_train.ddy('u')

# Arbitrary DSL for extracting features (useful for symbolic regression)
forcing_eddy_train.extract_feature('add(mul(ddx(u), q), mul(ddy(v), q))')

# Initialize a parameterization
param = pse.ZB2020Parameterization()

# Compute offline metrics
offline_metrics_eddy = param.test_offline(forcing_eddy_test)
print(offline_metrics_eddy.correlation)
print(offline_metrics_eddy.u_forcing_advection_temporal_correlation)
offline_metrics_jets = param.test_offline(forcing_jets_test)
print(offline_metrics_jets.correlation)
print(offline_metrics_jets.u_forcing_advection_temporal_correlation)

# Run online simulations
param_eddy = pse.Dataset(xr.concat(
    param.run_online(**forcing_eddy_test.pyqg_params) for _ in range(5),
    'run'
))
param_jets = pse.Dataset(xr.concat(
    param.run_online(**forcing_jets_test.pyqg_params) for _ in range(5),
    'run'
))

# Compare online performance to baselines
hires_eddy = pse.Dataset("/scratch/zanna/pyqg/data/test/hires/*.nc")
lores_eddy = pse.Dataset("/scratch/zanna/pyqg/data/test/lores/*.nc")
hires_jets = pse.Dataset("/scratch/zanna/pyqg/data/transfer/hires/*.nc")
lores_jets = pse.Dataset("/scratch/zanna/pyqg/data/transfer/lores/*.nc")

pse.plot_helpers.compare_simulations(
    lores_eddy.assign_attrs(label='Lo-res'),
    param_eddy.assign_attrs(label='Lo-res + param'),
    hires_eddy.assign_attrs(label='Hi-res'),
    title_suffix=', Eddy Config'
)

pse.plot_helpers.compare_simulations(
    lores_jets.assign_attrs(label='Lo-res'),
    param_jets.assign_attrs(label='Lo-res + param'),
    hires_jets.assign_attrs(label='Hi-res'),
    title_suffix=', Jet Config'
)
```

See [here](./examples) for more examples.

### Defining your own parameterizations

You can either mimic [this example in the pyqg documentation](https://pyqg.readthedocs.io/en/latest/examples/parameterization.html), or you can use the helpers in this library as follows:

```python
import pyqg_subgrid_experiments as pse

# Define a parameterization that predicts the subgrid forcing of potential vorticity
class QParameterization(pse.Parameterization):
    @property
    def targets(self):
        return ['q_forcing_advection']
        
    def predict(self, m):           
        dq = some_computation_involving(m)
        return dict(q_forcing_advection=dq)
        
# Define a parameterization that predicts the subgrid forcing of velocity
class UVParameterization(pse.Parameterization):
    @property
    def targets(self):
        return ['u_forcing_advection', 'v_forcing_advection']
        
    def predict(self, m):           
        du, dv = some_computation_involving(m)
        return dict(u_forcing_advection=du, v_forcing_advection=dv)
        
# Define a parameterization that predicts the subgrid forcing of PV, but as fluxes
class QFluxParameterization(pse.Parameterization):
    @property
    def targets(self):
        return ['uq_subgrid_flux', 'vq_subgrid_flux']
        
    def predict(self, m):           
        uq_flux, vq_flux = some_computation_involving(m)
        return dict(uq_subgrid_flux=uq_flux, vq_subgrid_flux=vq_flux)
        
param = QParameterization() # or another
arbitrary_simulation = param.run_online(**pyqg_params)

dataset = pse.Dataset('/path/to/dataset')
simulation_like_dataset = param.run_online(**dataset.pyqg_params)
```

## More Details

More detailed documentation is coming soon, but for now, see [this Overleaf](https://www.overleaf.com/read/jcfbxczmnptb) for work-in-progress definitions of the dataset generation process, included fields, and both offline and online metrics.
