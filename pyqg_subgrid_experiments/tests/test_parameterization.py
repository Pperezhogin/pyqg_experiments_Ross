import pytest
import pyqg_subgrid_experiments as pse
import os
import xarray as xr
import numpy as np
import pyqg

dirname = os.path.dirname(os.path.realpath(__file__))

def test_saving_and_loading():
    pdir = os.path.join(dirname, 'tmp')
    os.system(f"rm -rf {pdir}")

    ddir = os.path.join(dirname, 'fixtures/train.nc')

    dataset = pse.Dataset(
        xr.open_dataset(ddir).expand_dims('run')
    )

    param = pse.CNNParameterization.train_on(
            dataset,
            pdir,
            inputs=['q','u','v'],
            targets=['q_forcing_advection'],
            layerwise_inputs=True,
            layerwise_targets=True,
            num_epochs=0)

    assert len(param.models) == 2
    assert param.models[0].inputs == [('q',0),('u',0),('v',0)]
    assert param.models[1].inputs == [('q',1),('u',1),('v',1)]
    assert param.models[0].targets == [('q_forcing_advection',0)]
    assert param.models[1].targets == [('q_forcing_advection',1)]

    param2 = pse.CNNParameterization(pdir)

    assert param2.models[0].inputs == [('q',0),('u',0),('v',0)]
    assert param2.models[1].inputs == [('q',1),('u',1),('v',1)]
    assert param2.models[0].targets == [('q_forcing_advection',0)]
    assert param2.models[1].targets == [('q_forcing_advection',1)]

    m = pyqg.QGModel(nx=64)

    dq1 = param(m)
    dq2 = param2(m)
    dqd = param(dataset)

    np.testing.assert_allclose(dq1, dq2)

    assert dq1.shape == (2, 64, 64)
    assert dqd.shape == (len(dataset.time), 2, 64, 64)

    np.testing.assert_allclose(
        param.models[0].extract_inputs(dataset),
        np.moveaxis(np.array([
            dataset.isel(lev=0).q.data,
            dataset.isel(lev=0).u.data,
            dataset.isel(lev=0).v.data
        ]), 0, -3).reshape(-1, 3, 64, 64)
    )
    np.testing.assert_allclose(
        param.models[1].extract_inputs(dataset),
        np.moveaxis(np.array([
            dataset.isel(lev=1).q.data,
            dataset.isel(lev=1).u.data,
            dataset.isel(lev=1).v.data
        ]), 0, -3).reshape(-1, 3, 64, 64)
    )
    np.testing.assert_allclose(
        param.models[0].extract_targets(dataset),
        dataset.isel(lev=0).q_forcing_advection.data.reshape(-1, 1, 64, 64)
    )
    np.testing.assert_allclose(
        param.models[1].extract_targets(dataset),
        dataset.isel(lev=1).q_forcing_advection.data.reshape(-1, 1, 64, 64)
    )

def test_testing():
    pdir = os.path.join(dirname, 'tmp')
    os.system(f"rm -rf {pdir}")

    ddir = os.path.join(dirname, 'fixtures/hires_downscaled.nc')

    dataset = xr.open_dataset(ddir).expand_dims('run')

    param = pse.CNNParameterization.train_on(
            dataset,
            directory=pdir,
            inputs=['q','u','v'],
            targets=['q_forcing_advection'],
            layerwise_inputs=True,
            layerwise_targets=True,
            num_epochs=0)

    #perf = param.test_on(dataset)
