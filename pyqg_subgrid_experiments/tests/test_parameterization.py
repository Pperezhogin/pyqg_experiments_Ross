import pytest
import pyqg_subgrid_experiments as pse
import os
import xarray as xr
import numpy as np
import pyqg

dirname = os.path.dirname(os.path.realpath(__file__))

def load_dataset():
    return pse.Dataset(os.path.join(dirname, 'fixtures/train.nc'))

def test_saving_and_loading():
    dataset = load_dataset()

    pdir = os.path.join(dirname, 'tmp')
    os.system(f"rm -rf {pdir}")
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

    np.testing.assert_allclose(
        param.models[0].input_scale.sd,
        param2.models[0].input_scale.sd
    )
    np.testing.assert_allclose(
        param.models[0].output_scale.sd,
        param2.models[0].output_scale.sd
    )
    np.testing.assert_allclose(
        param.models[0].output_scale.sd,
        dataset.q_forcing_advection.isel(lev=0).std()
    )
    np.testing.assert_allclose(
        param.models[0].output_scale.mu,
        0
    )

    m = pyqg.QGModel(nx=64)

    dq1 = param(m)
    dq2 = param2(m)
    dqd = param(dataset)

    np.testing.assert_allclose(dq1, dq2)

    assert dq1.shape == (2, 64, 64)
    assert dqd.shape == dataset.q.shape

    np.testing.assert_allclose(np.mean(dq1), 0, atol=1e-18)

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

def test_running():
    dataset = load_dataset()

    pdir = os.path.join(dirname, 'tmp')
    os.system(f"rm -rf {pdir}")
    param = pse.CNNParameterization.train_on(
            dataset,
            pdir,
            inputs=['u','v'],
            targets=['u_forcing_advection','v_forcing_advection'],
            num_epochs=0)

    offline = param.test_offline(dataset)

    # This is a random prediction, so it should have low correlation
    assert np.abs(offline.correlation.mean()) < 0.05

    # Make sure we got the shape / matchup correct
    dataset2 = pse.Dataset(
        xr.concat([dataset.ds, dataset.ds, dataset.ds], 'run')
    )

    offline2 = param.test_offline(dataset2)

    pred_std = float(np.std(offline.u_forcing_advection_predictions.data))

    for i in range(3):
        np.testing.assert_allclose(
            offline2.u_forcing_advection_predictions.data[i],
            offline.u_forcing_advection_predictions.data[0],
            atol=pred_std/1000
        )

    pyqg_kwargs = dataset.pyqg_params
    pyqg_kwargs['tmax'] = pyqg_kwargs['dt'] * 1
    pyqg_kwargs['log_level'] = 0

    online = param.run_online(sampling_freq=1, **pyqg_kwargs)

    # Assert it ran successfully (for only one step)
    assert online.time.shape == (1,)

def test_zb2020_parameterization():
    dataset = load_dataset()
    param = pse.ZB2020Parameterization()
    offline = param.test_offline(dataset)
    assert offline.correlation.mean() > 0.9
    assert offline.skill.mean() > 0.9

def test_target_variants():
    dataset = load_dataset()

    def setup_model(**kw):
        pdir = os.path.join(dirname, 'tmp')
        os.system(f"rm -rf {pdir}")
        model = pse.CNNParameterization.train_on(dataset, pdir, num_epochs=0, **kw)
        return model

    param_uv_q = setup_model(
            inputs=['u','v'],
            targets=['q_forcing_advection'],
            layerwise_inputs=True,
            layerwise_targets=True)

    assert len(param_uv_q.models) == 2
    assert param_uv_q.targets == ['q_forcing_advection']
    assert param_uv_q.models[0].inputs == [('u', 0), ('v', 0)]
    assert param_uv_q.models[1].inputs == [('u', 1), ('v', 1)]
    assert param_uv_q.models[0].targets == [('q_forcing_advection', 0)]
    assert param_uv_q.models[1].targets == [('q_forcing_advection', 1)]
    assert param_uv_q(dataset).shape == dataset.q.shape
    assert param_uv_q.parameterization_type == 'q_parameterization'

    param_uv2_q = setup_model(
            inputs=['u','v'],
            targets=['q_forcing_advection'],
            layerwise_inputs=False,
            layerwise_targets=True)

    assert len(param_uv2_q.models) == 2
    assert param_uv2_q.targets == ['q_forcing_advection']
    assert param_uv2_q.models[0].inputs == [('u', 0), ('u', 1), ('v', 0), ('v', 1)]
    assert param_uv2_q.models[1].inputs == [('u', 0), ('u', 1), ('v', 0), ('v', 1)]
    assert param_uv2_q.models[0].targets == [('q_forcing_advection', 0)]
    assert param_uv2_q.models[1].targets == [('q_forcing_advection', 1)]
    assert param_uv2_q(dataset).shape == dataset.q.shape

    param_uv2_q2 = setup_model(
            inputs=['u','v'],
            targets=['q_forcing_advection'],
            layerwise_inputs=False,
            layerwise_targets=False)

    assert len(param_uv2_q2.models) == 1
    assert param_uv2_q2.targets == ['q_forcing_advection']
    assert param_uv2_q2.models[0].inputs == [('u', 0), ('u', 1), ('v', 0), ('v', 1)]
    assert param_uv2_q2.models[0].targets == [('q_forcing_advection', 0), ('q_forcing_advection', 1)]
    assert param_uv2_q2(dataset).shape == dataset.q.shape

    param_uv_uv = setup_model(
            inputs=['u','v'],
            targets=['u_forcing_advection', 'v_forcing_advection'],
            layerwise_inputs=True,
            layerwise_targets=True)

    assert len(param_uv_uv.models) == 2
    assert param_uv_uv.targets == ['u_forcing_advection', 'v_forcing_advection']
    assert param_uv_uv.models[0].inputs == [('u', 0), ('v', 0)]
    assert param_uv_uv.models[1].inputs == [('u', 1), ('v', 1)]
    assert param_uv_uv.models[0].targets == [('u_forcing_advection', 0), ('v_forcing_advection', 0)]
    assert param_uv_uv.models[1].targets == [('u_forcing_advection', 1), ('v_forcing_advection', 1)]
    du, dv = param_uv_uv(dataset)
    assert du.shape == dataset.u.shape
    assert dv.shape == dataset.v.shape
    assert param_uv_uv.parameterization_type == 'uv_parameterization'

    param_uv2_uv2 = setup_model(
            inputs=['u','v'],
            targets=['u_forcing_advection', 'v_forcing_advection'],
            layerwise_inputs=False,
            layerwise_targets=False)

    assert len(param_uv2_uv2.models) == 1
    assert param_uv2_uv2.targets == ['u_forcing_advection', 'v_forcing_advection']
    assert param_uv2_uv2.models[0].inputs == [('u', 0), ('u', 1), ('v', 0), ('v', 1)]
    assert param_uv2_uv2.models[0].targets == [('u_forcing_advection', 0), ('u_forcing_advection', 1),
                                               ('v_forcing_advection', 0), ('v_forcing_advection', 1)]
    du, dv = param_uv2_uv2(dataset)
    assert du.shape == dataset.u.shape
    assert dv.shape == dataset.v.shape

    param_uv_flux = setup_model(
            inputs=['u','v'],
            targets=['uu_subgrid_flux', 'uv_subgrid_flux', 'vv_subgrid_flux'],
            layerwise_inputs=False,
            layerwise_targets=False)
    du, dv = param_uv_flux(dataset)
    assert du.shape == dataset.u.shape
    assert dv.shape == dataset.v.shape
