import os
import sys
import glob
import pyqg
import pickle
import numpy as np
import xarray as xr
import json
from scipy.stats import pearsonr
from pyqg.xarray_output import spatial_dims
import pyqg_subgrid_experiments as pse

YEAR = 24*60*60*360.

DEFAULT_PYQG_PARAMS = dict(nx=64, dt=3600., tmax=10*YEAR, tavestart=5*YEAR)

FORCING_ATTR_DATABASE = dict(
    uq_subgrid_flux=dict(
        long_name=r"x-component of advected PV subgrid flux, $\overline{u}\,\overline{q} - \overline{uq}$",
        units="meters second ^-2",
    ),
    vq_subgrid_flux=dict(
        long_name=r"y-component of advected PV subgrid flux, $\overline{v}\,\overline{q} - \overline{vq}$",
        units="meters second ^-2",
    ),
    uu_subgrid_flux=dict(
        long_name=r"xx-component of advected velocity subgrid flux, $\overline{u}^2 - \overline{u^2}$",
        units="meters second ^-2",
    ),
    uv_subgrid_flux=dict(
        long_name=r"xy-component of advected velocity subgrid flux, $\overline{u}\,\overline{v} - \overline{uv}$",
        units="meters second ^-2",
    ),
    vv_subgrid_flux=dict(
        long_name=r"yy-component of advected velocity subgrid flux, $\overline{v}^2 - \overline{v^2}$",
        units="meters second ^-2",
    ),
    q_forcing_advection=dict(
        long_name=r"PV subgrid forcing from advection, $\overline{(\mathbf{u} \cdot \nabla)q} - (\overline{\mathbf{u}} \cdot \overline{\nabla})\overline{q}$",
        units="second ^-2",
    ),
    u_forcing_advection=dict(
        long_name=r"x-velocity subgrid forcing from advection, $\overline{(\mathbf{u} \cdot \nabla)u} - (\overline{\mathbf{u}} \cdot \overline{\nabla})\overline{u}$",
        units="second ^-2",
    ),
    v_forcing_advection=dict(
        long_name=r"y-velocity subgrid forcing from advection, $\overline{(\mathbf{u} \cdot \nabla)v} - (\overline{\mathbf{u}} \cdot \overline{\nabla})\overline{v}$",
        units="second ^-2",
    ),
    dqdt_through_lores=dict(
        long_name="PV tendency from passing downscaled high-res initial conditions through low-res simulation",
        units="second ^-2",
    ),
    dqdt_through_hires_downscaled=dict(
        long_name="Downscaled PV tendency from passing high-res initial conditions through high-res simulation",
        units="second ^-2",
    ),
    q_forcing_total=dict(
        long_name="Difference between downscaled high-res tendency and low-res tendency",
        units="second ^-2"
    ),
)

def spatial_var(var, ds):
    return xr.DataArray(var, coords=dict([(d, ds.coords[d]) for d in spatial_dims]), dims=spatial_dims)

def concat_and_convert(datasets, drop_complex=1):
    # Concatenate datasets along the time dimension
    d = xr.concat(datasets, dim='time')
    
    # Diagnostics get dropped by this procedure since they're only present for
    # part of the timeseries; resolve this by saving the most recent
    # diagnostics (they're already time-averaged so this is ok)
    for k,v in datasets[-1].variables.items():
        if k not in d:
            d[k] = v.isel(time=-1)

    # To save on storage, reduce double -> single
    for k,v in d.variables.items():
        if v.dtype == np.float64:
            d[k] = v.astype(np.float32)
        elif v.dtype == np.complex128:
            d[k] = v.astype(np.complex64)

    # Potentially drop complex variables
    if drop_complex:
        complex_vars = [k for k,v in d.variables.items() if np.iscomplexobj(v)]
        d = d.drop(complex_vars)

    return d

def initialize_pyqg_model(**kwargs):
    pyqg_kwargs = dict(DEFAULT_PYQG_PARAMS)
    pyqg_kwargs.update(**kwargs)
    return pyqg.QGModel(**pyqg_kwargs)

def generate_dataset(sampling_freq=1000, sampling_dist='uniform', **kwargs):
    m = initialize_pyqg_model(**kwargs)
    return run_simulation(m, sampling_freq=sampling_freq, sampling_dist=sampling_dist)

def generate_forcing_dataset(hires=256, lores=64, sampling_freq=1000, sampling_dist='uniform', filtr=None, **pyqg_params):
    params1 = dict(DEFAULT_PYQG_PARAMS)
    params1.update(pyqg_params)
    params1['nx'] = hires

    params2 = dict(DEFAULT_PYQG_PARAMS)
    params2.update(pyqg_params)
    params2['nx'] = lores

    m1 = pyqg.QGModel(**params1)
    m2 = pyqg.QGModel(**params2)

    ds = run_forcing_simulations(m1, m2, sampling_freq=sampling_freq, sampling_dist=sampling_dist, filtr=filtr)
    return ds.assign_attrs(pyqg_params=json.dumps(params2))

def run_simulation(m, sampling_freq=1000, sampling_dist='uniform'):
    snapshots = []
    while m.t < m.tmax:
        if m.tc % sampling_freq == 0:
            snapshots.append(m.to_dataset().copy(deep=True))
        m._step_forward()
    return concat_and_convert(snapshots)

def spectral_filter_and_coarsen(hires_var, m1, m2, filtr=None):
    if not isinstance(m1, pyqg.QGModel):
        m1 = pse.Dataset.wrap(m1).m
        m2 = pse.Dataset.wrap(m2).m

    if hires_var.shape == m1.q.shape:
        return m2.ifft(spectral_filter_and_coarsen(m1.fft(hires_var), m1, m2, filtr))
    elif hires_var.shape == m1.qh.shape:
        if filtr is None:
            filtr = np.exp(-m2.wv**2 * (2*m2.dx)**2 / 24)
        elif filtr == 'builtin':
            filtr = m2.filtr
        keep = m2.qh.shape[1]//2
        return np.hstack((
            hires_var[:,:keep,:keep+1],
            hires_var[:,-keep:,:keep+1]
        )) * filtr / (m1.nx // m2.nx)**2
    else:
        raise ValueError

def run_forcing_simulations(m1, m2, sampling_freq=1000, sampling_dist='uniform', filtr=None):
    def downscaled(hires_var):
        return spectral_filter_and_coarsen(hires_var, m1, m2, filtr)

    def advected(var):
        if var.shape == m1.q.shape:
            m = m1
        elif var.shape == m2.q.shape:
            m = m2
        else:
            raise ValueError
        ik = -np.array(m._ik)[np.newaxis,np.newaxis,:]
        il = -np.array(m._il)[np.newaxis,:,np.newaxis]
        return m.ifft(ik * m.fft(m.ufull * var)) + m.ifft(il * m.fft(m.vfull * var))

    # Set the diagnostics of the coarse simulation to be those of the hi-res
    # simulation, but downscaled
    old_inc = m2._increment_diagnostics
    def new_inc():
        original_lores_q = np.array(m2.q)
        m2.set_q1q2(*m2.ifft(downscaled(m1.qh)))
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
            # Update the PV of the low-resolution model to match the downscaled
            # high-resolution PV
            m2.set_q1q2(*m2.ifft(downscaled(m1.qh)))
            m2._invert() # recompute velocities

            # Convert the low-resolution model to an xarray dataset
            ds = m2.to_dataset().copy(deep=True)

            # Compute various versions of the subgrid forcing defined in terms
            # of the advection and downscaling operators
            def save_var(key, val):
                zero = ds.q*0
                if len(val.shape) == 3: val = val[np.newaxis]
                ds[key] = zero + val

            m1._invert()

            save_var('q_forcing_advection', downscaled(advected(m1.q)) - advected(m2.q))
            save_var('u_forcing_advection', downscaled(advected(m1.ufull)) - advected(m2.ufull))
            save_var('v_forcing_advection', downscaled(advected(m1.vfull)) - advected(m2.vfull))

            save_var('uq_subgrid_flux', m2.ufull * m2.q - downscaled(m1.ufull * m1.q))
            save_var('vq_subgrid_flux', m2.vfull * m2.q - downscaled(m1.vfull * m1.q))

            save_var('uu_subgrid_flux', m2.ufull**2 - downscaled(m1.ufull**2))
            save_var('vv_subgrid_flux', m2.vfull**2 - downscaled(m1.vfull**2))
            save_var('uv_subgrid_flux', m2.ufull * m2.vfull - downscaled(m1.ufull * m1.vfull))

            # Now, step both models forward (which recomputes ∂q/∂t)
            m1._step_forward()
            m2._step_forward()

            # Store the resulting values of ∂q/∂t
            save_var('dqdt_through_lores', m2.ifft(m2.dqhdt))
            save_var('dqdt_through_hires_downscaled', m2.ifft(downscaled(m1.dqhdt)))

            # Finally, store the difference between those two quantities (which
            # serves as an alternate measure of subgrid forcing, that takes
            # into account other differences in the simulations beyond just
            # hi-res vs. lo-res advection)
            ds['q_forcing_total'] = ds['dqdt_through_hires_downscaled'] - ds['dqdt_through_lores']

            # Add attributes and units to the xarray dataset
            for key, attrs in FORCING_ATTR_DATABASE.items():
                if key in ds:
                    ds[key] = ds[key].assign_attrs(attrs)

            # Save the datasets
            if 'dqdt' in ds: ds = ds.drop('dqdt')
            snapshots.append(ds)
        else:
            # If we aren't sampling at this index, just step both models
            # forward (with the second model continuing to evolve in lock-step)
            #m2.set_q1q2(*downscaled_hires_q())
            m1._step_forward()
            m2._step_forward()

    # Concatenate the datasets along the time dimension
    return concat_and_convert(snapshots).assign_attrs(hires=m1.nx, lores=m2.nx)

def time_until_uncorrelated(m1, m2, thresh=0.5, perturbation_sd=1e-10, max_timesteps=100000):
    keep = m2.qh.shape[1]//2
    filtr = np.exp(-m2.wv**2 * (2*m2.dx)**2 / 24)
    fac = (m1.nx // m2.nx)**2

    def possibly_downscaled(var):
        if m1.nx == m2.nx:
            return var
        elif var.shape == m1.q.shape:
            return m2.ifft(possibly_downscaled(m1.fft(var)))
        elif var.shape == m1.qh.shape:
            return np.hstack((
                var[:,:keep,:keep+1],
                var[:,-keep:,:keep+1]
            )) * filtr / fac
        else:
            raise ValueError
        
    def correlation(qa, qb):
        return pearsonr(qa.ravel(), qb.ravel())[0]
    
    for _ in range(3):
        m2.set_q1q2(*possibly_downscaled(m1.q))
        m1._step_forward()
        m2._step_forward()
        
    initial_conditions = possibly_downscaled(m1.q)
    perturbed_conditions = initial_conditions + np.random.normal(size=initial_conditions.shape) * perturbation_sd
    
    assert correlation(initial_conditions, perturbed_conditions) > 0.9
    
    m2.set_q1q2(*perturbed_conditions)
    
    steps = 0
    
    while steps < max_timesteps:
        m1._step_forward()
        m2._step_forward() 
        steps += 1
        if correlation(m2.q, possibly_downscaled(m1.q)) <= thresh:
            break
    
    return steps

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_to', type=str)
    parser.add_argument('--control', type=int, default=0)
    parser.add_argument('--zb2020', type=float, default=0)
    parser.add_argument('--ag7531', type=float, default=0)
    parser.add_argument('--cnn_param', type=str, default='')
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

    if args.control:
        ds = generate_dataset(**kwargs)
    elif args.zb2020:
        ds = pse.ZB2020Parameterization(factor_mult=args.zb2020).run_online(**kwargs)
    elif args.ag7531:
        ds = pse.AG7531Parameterization(factor=args.ag7531).run_online(**kwargs)
    elif args.cnn_param:
        ds = pse.CNNParameterization(args.cnn_param).run_online(**kwargs)
    else:
        ds = generate_forcing_dataset(**kwargs)

    ds.to_netcdf(args.save_to)
