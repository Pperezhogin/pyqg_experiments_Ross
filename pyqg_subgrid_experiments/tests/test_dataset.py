import pytest
import pyqg_subgrid_experiments as pse
import os
import xarray as xr
import numpy as np
import pyqg

dirname = os.path.dirname(os.path.realpath(__file__))

def load_dataset():
    return pse.Dataset(os.path.join(dirname, 'fixtures/train.nc'))

def test_extract_feature():
    ds = load_dataset()

    du_dx = ds.extract_feature('ddx(u)')
    div_u = ds.extract_feature('add(ddx(u),ddy(v))')
    div_u2 = ds.div('u','v')

    np.testing.assert_allclose(du_dx, ds.ddx('u'))
    np.testing.assert_allclose(div_u, ds.ddx('u') + ds.ddy('v'))
    np.testing.assert_allclose(div_u, div_u2)
