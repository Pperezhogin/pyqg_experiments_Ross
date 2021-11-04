import os
import sys
import glob
import pyqg
import pickle
import numpy as np
import xarray as xr
import numpy.fft as npfft
import json
from pyqg.xarray_output import spatial_dims

YEAR = 24*60*60*360.

DEFAULT_PYQG_PARAMS = dict(nx=64, dt=3600., tmax=10*YEAR, tavestart=5*YEAR)

FORCING_ATTR_DATABASE = dict(
    uq_difference=dict(
        long_name="Pre-divergence x-component of advected PV subgrid forcing, $\overline{u}\,\overline{q} - \overline{uq}$",
        units="meters second ^-2",
    ),
    vq_difference=dict(
        long_name="Pre-divergence y-component of advected PV subgrid forcing, $\overline{v}\,\overline{q} - \overline{vq}$",
        units="meters second ^-2",
    ),
    q_forcing_advection=dict(
        long_name="Advected PV subgrid forcing, $\overline{(\mathbf{u} \cdot \\nabla)q} - (\overline{\mathbf{u}} \cdot \overline{\\nabla})\overline{q}$",
        units="second ^-2",
    ),
    u_forcing_advection=dict(
        long_name="Advected x-velocity subgrid forcing, $\overline{(\mathbf{u} \cdot \\nabla)u} - (\overline{\mathbf{u}} \cdot \overline{\\nabla})\overline{u}$",
        units="second ^-2",
    ),
    v_forcing_advection=dict(
        long_name="Advected y-velocity subgrid forcing, $\overline{(\mathbf{u} \cdot \\nabla)v} - (\overline{\mathbf{u}} \cdot \overline{\\nabla})\overline{v}$",
        units="second ^-2",
    ),
    dqdt_post=dict(
        long_name="Subsequent partial derivative of PV wrt. time",
        units="second ^-2",
    ),
    dqdt_pre_hires_downscaled=dict(
        long_name="Previous partial derivative of PV wrt. time in the hi-res model (downscaled)",
        units="second ^-2"
    ),
    dqdt_post_hires_downscaled=dict(
        long_name="Subsequent partial derivative of PV wrt. time in the hi-res model (downscaled)",
        units="second ^-2",
    ),
    q_forcing_empirical=dict(
        long_name="Empirical PV subgrid forcing, $\overline{\partial_t q} - \partial_t \overline{q}$",
        units="second ^-2"
    ),
)

def spatial_var(var, ds):
    return xr.DataArray(var, coords=dict([(d, ds.coords[d]) for d in spatial_dims]), dims=spatial_dims)

def concat_and_convert(datasets):
    # Concatenate datasets along the time dimension
    d = xr.concat(datasets, dim='time')
    
    # Diagnostics get dropped by this procedure since they're only present for
    # part of the timeseries; resolve this by saving the most recent
    # diagnostics (they're already time-averaged so this is ok)
    for k,v in datasets[-1].variables.items():
        if k not in d:
            d[k] = v.isel(time=-1)

    # To save on storage, reduce float64 -> float32
    for k,v in d.variables.items():
        if v.dtype == np.float64:
            d[k] = v.astype(np.float32)

    # Drop complex variables since they can't be saved in netcdf
    complex_vars = [k for k,v in d.variables.items() if v.dtype == np.complex128]

    return d.drop(complex_vars)

def advected(ds, quantity='q'):
    # Handle double periodicity by padding before differentiating / unpadding afterwards
    pad = dict(x=(3,3), y=(3,3))
    unpad = dict(x=slice(3,-3), y=slice(3,-3))
    # Use second-order diff., though padding might make it unnecessary
    q = ds[quantity].pad(pad, mode='wrap')
    dq_dx = q.differentiate('x', edge_order=2).isel(indexers=unpad)
    dq_dy = q.differentiate('y', edge_order=2).isel(indexers=unpad)
    # Return the advected quantity
    return ds.ufull * dq_dx +  ds.vfull * dq_dy

def initialize_pyqg_model(**kwargs):
    pyqg_kwargs = dict(DEFAULT_PYQG_PARAMS)
    pyqg_kwargs.update(**kwargs)
    return pyqg.QGModel(**pyqg_kwargs)


def zb2020_uv_parameterization(m, factor_upper=-62261027.5, factor_lower=-54970158.2):
    # Implements Equation 6 of
    # https://laurezanna.github.io/files/Zanna-Bolton-2020.pdf with default
    # factors tuned to (a particular configuration of) pyqg.

    ds = m.to_dataset().isel(time=-1)
    ds['relative_vorticity'] = ds.v.differentiate('x') - ds.u.differentiate('y')
    ds['shearing_deformation'] = ds.u.differentiate('y') + ds.v.differentiate('x')
    ds['stretching_deformation'] = ds.u.differentiate('x') - ds.v.differentiate('y')

    factors = np.array([factor_upper, factor_lower])

    du = factors[:,np.newaxis,np.newaxis] * (
        (-ds.relative_vorticity*ds.shearing_deformation).differentiate('x')
        + (ds.relative_vorticity*ds.stretching_deformation).differentiate('y')
        + 0.5*(
            ds.relative_vorticity**2
            + ds.shearing_deformation**2
            + ds.stretching_deformation**2
        ).differentiate('x')  
    )

    dv = factors[:,np.newaxis,np.newaxis] * (
          (ds.relative_vorticity*ds.shearing_deformation).differentiate('y')
        + (ds.relative_vorticity*ds.stretching_deformation).differentiate('x')
        + 0.5*(
            ds.relative_vorticity**2
            + ds.shearing_deformation**2
            + ds.stretching_deformation**2
        ).differentiate('y')  
    )
    
    return np.array(du.data), np.array(dv.data)

def generate_ag7531_parameterized_dataset(factor=1.0, **kwargs):
    import torch
    import sys
    sys.path.append('/scratch/zanna/code/ag7531')
    sys.path.append('/scratch/zanna/code/ag7531/pyqgparamexperiments')
    from subgrid.models.utils import load_model_cls
    from subgrid.models.transforms import SoftPlusTransform
    from parameterization import Parameterization
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_cls = load_model_cls('subgrid.models.models1', 'FullyCNN')
    net = model_cls(2, 4)
    net.final_transformation = SoftPlusTransform()
    net.final_transformation.indices = [1,3]
    net.load_state_dict(
        torch.load('/scratch/zanna/data/pyqg/models/ag7531/1/dc74cea68a7f4c7e98f9228649a97135/artifacts/models/trained_model.pth'),
    )
    net.to(device)
    param = Parameterization(net, device)
    def uv_parameterization(m):
        du, dv = param(m.ufull, m.vfull, m.t)
        return du*factor, dv*factor
    return generate_dataset(uv_parameterization=uv_parameterization, **kwargs)

def generate_physically_parameterized_dataset(factor_upper=-62261027.5, factor_lower=-54970158.2, **kwargs):
    uv_param = lambda m: zb2020_uv_parameterization(m, factor_upper=factor_upper, factor_lower=factor_lower)
    return generate_dataset(uv_parameterization=uv_param, **kwargs)

def initialize_parameterized_model(cnn0, cnn1, **kwargs):
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn0.to(device)
    cnn1.to(device)
    
    def eval_models(m):
        x0 = cnn0.extract_inputs_from_qgmodel(m)
        if cnn0.inputs == cnn1.inputs:
            x1 = x0
        else:
            x1 = cnn1.extract_inputs_from_qgmodel(m)

        y0 = cnn0.predict(x0, device=device)[0]
        y1 = cnn1.predict(x1, device=device)[0]
        return y0, y1

    if cnn0.targets == [('u_forcing_advection', 0), ('v_forcing_advection', 0)]:
        def uv_parameterization(m):
            y0, y1 = eval_models(m)
            du = np.array([y0[0], y1[0]]).astype(m.q.dtype)
            dv = np.array([y0[1], y1[1]]).astype(m.q.dtype)
            return du, dv
        kwargs['uv_parameterization'] = uv_parameterization
    else:
        def q_parameterization(m):
            y0, y1 = eval_models(m)
            if cnn0.targets == [('uq_difference', 0), ('vq_difference', 0)]:
                ds = m.to_dataset()
                ds['uq_difference'] = spatial_var(np.array([y0[0], y1[0]])[np.newaxis], ds)
                ds['vq_difference'] = spatial_var(np.array([y0[1], y1[1]])[np.newaxis], ds)
                ds['q_forcing_pred'] = (
                    ds.uq_difference.differentiate('x') +
                    ds.vq_difference.differentiate('y') 
                )
                dq = np.array(ds['q_forcing_pred'].data[0])
            else:
                dq = np.array([y0[0], y1[0]])

            return dq.astype(m.q.dtype)
        kwargs['q_parameterization'] = q_parameterization

    return initialize_pyqg_model(**kwargs)

def generate_dataset(sampling_freq=1000, sampling_dist='uniform', **kwargs):
    m = initialize_pyqg_model(**kwargs)
    return run_simulation(m, sampling_freq=sampling_freq, sampling_dist=sampling_dist)

def generate_parameterized_dataset(cnn0, cnn1, sampling_freq=1000, sampling_dist='uniform', **kwargs):
    m = initialize_parameterized_model(cnn0, cnn1, **kwargs)
    return run_simulation(m, sampling_freq=sampling_freq, sampling_dist=sampling_dist)

def generate_forcing_dataset(hires=256, lores=64, sampling_freq=1000, sampling_dist='uniform', filter=None, **pyqg_params):
    params1 = dict(DEFAULT_PYQG_PARAMS)
    params1.update(pyqg_params)
    params1['nx'] = hires

    params2 = dict(DEFAULT_PYQG_PARAMS)
    params2.update(pyqg_params)
    params2['nx'] = lores

    m1 = pyqg.QGModel(**params1)
    m2 = pyqg.QGModel(**params2)

    ds = run_forcing_simulations(m1, m2, sampling_freq=sampling_freq, sampling_dist=sampling_dist, filter=filter)
    return ds.assign_attrs(pyqg_params=json.dumps(params2))

def run_simulation(m, sampling_freq=1000, sampling_dist='uniform'):
    snapshots = []
    while m.t < m.tmax:
        if m.tc % sampling_freq == 0:
            snapshots.append(m.to_dataset().copy(deep=True))
        m._step_forward()
    return concat_and_convert(snapshots)

def run_forcing_simulations(m1, m2, sampling_freq=1000, sampling_dist='uniform', filter=None):
    scale = m1.nx/m2.nx
    assert scale == int(scale)
    assert scale > 1
    scale = int(scale)
    
    if filter is None:
        import gcm_filters
        filter = gcm_filters.Filter(filter_scale=scale, dx_min=1, grid_type=gcm_filters.GridType.REGULAR)

    def downscaled(ds):
        return filter.apply(ds, dims=['y','x']).coarsen(x=scale, y=scale).mean()

    ds1 = m1.to_dataset().copy(deep=True)

    def downscaled_hires_q():
        return downscaled(
            xr.DataArray(m1.q, coords=[ds1.coords[d] for d in ['lev','y','x']])
        ).data

    # Set the diagnostics of the coarse simulation to be those of the hi-res
    # simulation, but downscaled
    old_inc = m2._increment_diagnostics
    def new_inc():
        original_lores_q = np.array(m2.q)
        m2.set_q1q2(*downscaled_hires_q())
        m2._invert()
        old_inc()
        m2.set_q1q2(*original_lores_q)
        m2._invert()
    m2._increment_diagnostics = new_inc

    # Arrays to hold the datasets we'll sample over the course of the
    # simulation 
    snapshots = []

    # If we're sampling irregularly, pick the time index for the next sample
    # from an exponential distribution
    if sampling_dist == 'exponential':
        next_sample = int(np.random.exponential(sampling_freq))

    while m1.t < m1.tmax:
        if sampling_dist == 'exponential':
            # If we're sampling irregularly, check if we've hit the next
            # interval
            should_sample = m1.tc >= next_sample
            if should_sample:
                next_sample = m1.tc + int(np.random.exponential(sampling_freq))
        else:
            # If we're sampling regularly, check if we're at that fixed
            # interval
            should_sample = (m1.tc % sampling_freq == 0)

        if should_sample:
            # Convert the high-resolution model to an xarray dataset
            ds1 = m1.to_dataset().copy(deep=True)

            # Downscale the high-resolution dataset using our filter
            ds1_downscaled = downscaled(ds1)

            # Update the PV of the low-resolution model to match the downscaled
            # high-resolution PV
            downscaled_q = ds1_downscaled.q.isel(time=0).copy().data
            m2.set_q1q2(*downscaled_q)
            m2._invert() # recompute velocities

            # Convert the low-resolution model to an xarray dataset
            ds2 = m2.to_dataset().copy(deep=True)

            # Compute various versions of the subgrid forcing defined in terms
            # of the advection and downscaling operators
            ds2['advected_q'] = advected(ds2, 'q')
            ds2['advected_ufull'] = advected(ds2, 'ufull')
            ds2['advected_vfull'] = advected(ds2, 'vfull')
            ds2['q_forcing_advection'] = ds2['advected_q'] - downscaled(advected(ds1, 'q'))
            ds2['u_forcing_advection'] = ds2['advected_ufull'] - downscaled(advected(ds1, 'ufull'))
            ds2['v_forcing_advection'] = ds2['advected_vfull'] - downscaled(advected(ds1, 'vfull'))
            ds2['uq_difference'] = (ds1_downscaled.ufull * ds1_downscaled.q - downscaled(ds1.ufull * ds1.q))
            ds2['vq_difference'] = (ds1_downscaled.vfull * ds1_downscaled.q - downscaled(ds1.vfull * ds1.q))
            ds2['dqdt_pre_hires_downscaled'] = ds1_downscaled.dqdt

            # Now, step both models forward (which recomputes ∂q/∂t)
            m1._step_forward()
            m2._step_forward()

            # Store the resulting values of ∂q/∂t
            ds1['dqdt_post'] = xr.DataArray(
                npfft.irfftn(m1.dqhdt, axes=(-2,-1))[np.newaxis],
                coords=[ds1.coords[d] for d in spatial_dims])
            ds2['dqdt_post'] = xr.DataArray(
                npfft.irfftn(m2.dqhdt, axes=(-2,-1))[np.newaxis],
                coords=[ds2.coords[d] for d in spatial_dims])

            # On the lo-res dataset, also store the downscaled hi-res value of
            # ∂q/∂t
            ds2['dqdt_post_hires_downscaled'] = xr.DataArray(
                downscaled(ds1.dqdt_post).data,
                coords=[ds2.coords[d] for d in spatial_dims])

            # Finally, store the difference between those two quantities (which
            # serves as an alternate measure of subgrid forcing, that takes
            # into account other differences in the simulations beyond just
            # hi-res vs. lo-res advection)
            ds2['q_forcing_empirical'] = ds2['dqdt_post_hires_downscaled'] - ds2['dqdt_post']

            # Add attributes and units to the xarray dataset
            for key, attrs in FORCING_ATTR_DATABASE.items():
                if key in ds2:
                    ds2[key] = ds2[key].assign_attrs(attrs)

            # Save the datasets
            snapshots.append(ds2)
        else:
            # If we aren't sampling at this index, just step both models
            # forward (with the second model continuing to evolve in lock-step)
            m2.set_q1q2(*downscaled_hires_q())
            m1._step_forward()
            m2._step_forward()

    # Concatenate the datasets along the time dimension
    return concat_and_convert(snapshots).assign_attrs(downscaled_by=scale)

def generate_dataset(ag7531=0, control=0, physical=0, transfer_test=0, **kwargs):
    if physical:
        ds = generate_physically_parameterized_dataset(**kwargs)
    elif ag7531:
        ds = generate_ag7531_parameterized_dataset(**kwargs)
    elif control:
        ds = generate_dataset(**kwargs)
    else:
        ds = generate_forcing_dataset(**kwargs)
    return ds

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_to', type=str)
    parser.add_argument('--transfer_test', type=int, default=0)
    args, extra = parser.parse_known_args()

    # Setup parameters for dataset generation functions
    kwargs = dict()
    for param in extra:
        key, val = param.split('=')
        kwargs[key.replace('--', '')] = float(val)
    if args.transfer_test:
        kwargs.update(dict(rek=7.000000e-08, delta=0.1, beta=1.0e-11))

    os.system(f"mkdir -p {os.path.dirname(os.path.realpath(args.save_to))}")

    ds = generate_dataset(**kwargs)
    ds.to_netcdf(args.save_to)
