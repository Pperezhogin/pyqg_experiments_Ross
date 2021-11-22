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

More detailed version requirements will hopefully be determined soon.

## Usage

**NOTE: this is still in draft form, and subject to change.**

```python
import pyqg_subgrid_experiments as pse

# Load a dataset (thin wrapper around xarray.Dataset)
dataset = pse.Dataset("/scratch/zanna/pyqg/data/train/*.nc")

# Variables directly included
dataset.q_forcing_advection # shape = (n_runs, n_steps, 2, 64, 64)
dataset.u.shape
dataset.v.shape

# Differentiation in spectral space
dataset.ddx('u')
dataset.ddy('u')

# Arbitrary DSL for extracting features (useful for symbolic regression)
dataset.extract_feature('ddx_u_times_q_plus_ddy_v_times_q')

# Initialize a parameterization
param = pse.ZB2020Parameterization()

# Test it against a dataset, both offline and online
offline_metrics, simulations, online_metrics = param.test_on(dataset)

# Examine offline metrics
print(offline_metrics.correlation)

# Examine online performance
pse.plot_helpers.compare_simulations(
    dataset.assign_attrs(label='Hi-res downscaled'),
    simulations.assign_attrs(label='Lo-res + ZB2020 param'))
```

## More Details

More detailed documentation is coming soon, but for now, see [this Overleaf](https://www.overleaf.com/read/jcfbxczmnptb) for work-in-progress definitions of the dataset generation process, included fields, and both offline and online metrics.
