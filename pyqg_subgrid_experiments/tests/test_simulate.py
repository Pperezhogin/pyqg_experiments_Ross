import pytest
import pyqg_subgrid_experiments as pse
import os
import xarray as xr
import numpy as np
import pyqg

def test_downscaling():
    m1 = pyqg.QGModel(nx=256, log_level=0)
    m2 = pyqg.QGModel(nx=64, log_level=0)
    
    q_low1 = pse.spectral_filter_and_coarsen(m1.q, m1, m2)
    q_low2 = pse.spectral_filter_and_coarsen(m1.q, m1, m2, filtr=1)

    assert q_low1.shape == m2.q.shape
    assert q_low1.shape == q_low2.shape
   
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(q_low1, q_low2)

def test_run_forcing_simulation():
    dt = 3600.
    m1 = pyqg.QGModel(nx=256, log_level=0, dt=dt, tmax=dt*2)
    m2 = pyqg.QGModel(nx=64, log_level=0, dt=dt, tmax=dt*2)
    q0 = m1.q
    ds = pse.run_forcing_simulations(m1, m2, sampling_freq=1)
    assert len(ds.time) == 2
