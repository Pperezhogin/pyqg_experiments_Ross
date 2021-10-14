import os
import sys
import glob
import pyqg
import pickle
import gcm_filters
import numpy as np
import xarray as xr
import numpy.fft as npfft
from pyqg.xarray_output import spatial_dims
dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dirname)
from symbolic_regression_parameterization import *

def zb2020_uv_parameterization(m, factor_upper=-19723861.3, factor_lower=-32358493.6):
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
    return generate_control_dataset(uv_parameterization=uv_parameterization, **kwargs)

def generate_symbolic_regression_parameterized_dataset(
        m1_path=f"{dirname}/models/upper_layer_symbolic_regression_parameterization.pkl",
        m2_path=f"{dirname}/models/lower_layer_symbolic_regression_parameterization.pkl",
        factor=1.0,
        **kwargs):
    with open(m1_path, 'rb') as f: upper = pickle.load(f)
    with open(m2_path, 'rb') as f: lower = pickle.load(f)

    def q_parameterization(m):
        p1 = upper.predict(m)
        p2 = np.zeros_like(p1) #lower.predict(m)
        return factor * np.array([ p1 - p1.mean(), p2 - p2.mean() ])

    return generate_control_dataset(q_parameterization=q_parameterization, **kwargs)

def generate_physically_parameterized_dataset(factor_upper=-19723861.3, factor_lower=-32358493.6, **kwargs):
    uv_param = lambda m: zb2020_uv_parameterization(m, factor_upper=factor_upper, factor_lower=factor_lower)
    return generate_control_dataset(uv_parameterization=uv_param, **kwargs)

def generate_parameterized_dataset(cnn0, cnn1, inputs, **kwargs):
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn0.to(device)
    cnn1.to(device)

    def get_input(m,inp):
        if 'dqdt' in inp:
            import numpy.fft as npfft
            val = getattr(m,inp.replace('dq','dqh'))
            return npfft.irfftn(val,axes=(-2,-1))
        else:
            return getattr(m,inp)

    def get_inputs(m,z):
        return np.array([[
            get_input(m,inp)[z]
            for inp in inputs.split(",")
        ]]).astype(np.float32)

    def q_parameterization(m):
        dq = np.array([
            cnn0.predict(get_inputs(m,0), device=device)[0,0],
            cnn1.predict(get_inputs(m,1), device=device)[0,0]
        ]).astype(m.q.dtype)
        return dq

    return generate_control_dataset(q_parameterization=q_parameterization, **kwargs)

def spatial_var(var, ds):
    return xr.DataArray(var, coords=dict([(d, ds.coords[d]) for d in spatial_dims]), dims=spatial_dims)

def concat_and_convert(datasets):
    # Convert and save important spectral variables back in real space
    for ds in datasets:
        ds['streamfunction'] = spatial_var(npfft.irfftn(ds.ph, axes=(-2,-1)), ds)

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

def generate_control_dataset(nx=64, dt=3600., sampling_freq=1000, sampling_dist='uniform', after_each=None, **kwargs):
    year = 24*60*60*360.
    pyqg_kwargs = dict(tmax=10*year, dt=dt)
    pyqg_kwargs.update(**kwargs)
    pyqg_kwargs['tavestart'] = pyqg_kwargs['tmax'] * 0.5

    m = pyqg.QGModel(nx=nx, **pyqg_kwargs)
    datasets = []

    while m.t < m.tmax:
        if m.tc % sampling_freq == 0:
            ds = m.to_dataset().copy(deep=True)
            datasets.append(ds)
        m._step_forward()
        if after_each is not None:
            after_each(m)

    return concat_and_convert(datasets)

def generate_forcing_dataset(nx1=256, nx2=64, dt=3600., sampling_freq=1000, sampling_dist='uniform', filter=None, **kwargs):
    scale = nx1//nx2
    
    year = 24*60*60*360.
    pyqg_kwargs = dict(tmax=10*year, dt=dt)
    pyqg_kwargs.update(**kwargs)
    pyqg_kwargs['tavestart'] = pyqg_kwargs['tmax'] * 0.5
    
    if filter is None:
        filter = gcm_filters.Filter(filter_scale=scale, dx_min=1, grid_type=gcm_filters.GridType.REGULAR)

    def downscaled(ds):
        return filter.apply(ds, dims=['y','x']).coarsen(x=scale, y=scale).mean()

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

    m1 = pyqg.QGModel(nx=nx1, **pyqg_kwargs)
    m2 = pyqg.QGModel(nx=nx2, **pyqg_kwargs)

    # Set the diagnostics of the coarse simulation to be those of the hi-res
    # simulation, but downscaled
    old_inc = m2._increment_diagnostics
    def new_inc():
        ds1 = m1.to_dataset().copy(deep=True)
        ds2 = m2.to_dataset().copy(deep=True)
        m2.set_q1q2(*downscaled(ds1).q.isel(time=0).copy().data)
        m2._invert()
        old_inc()
        m2.set_q1q2(*ds2.q.isel(time=0).copy().data)
        m2._invert()
    m2._increment_diagnostics = new_inc
    
    datasets1 = []
    datasets2 = []

    if sampling_dist == 'exponential':
        next_sample = int(np.random.exponential(sampling_freq))

    while m1.t < m1.tmax:
        if sampling_dist == 'exponential':
            should_sample = m1.tc >= next_sample
            if should_sample:
                next_sample = m1.tc + int(np.random.exponential(sampling_freq))
        else:
            should_sample = (m1.tc % sampling_freq == 0)

        if should_sample:
            ds1 = m1.to_dataset().copy(deep=True)

            derivs = ['dqdt', 'dqdt_p', 'dqdt_pp']
            for dq in derivs:
                ds1[dq] = xr.DataArray(
                    npfft.irfftn(getattr(m1, dq.replace('dq', 'dqh')), axes=(-2,-1))[np.newaxis],
                    coords=[ds1.coords[d] for d in spatial_dims]
                )

            ds1_downscaled = downscaled(ds1)
            m2.set_q1q2(*ds1_downscaled.q.isel(time=0).copy().data)
            ds2 = m2.to_dataset().copy(deep=True)

            ds2['q_forcing_advection'] = (
                advected(ds1_downscaled, 'q') -
                downscaled(advected(ds1, 'q'))
            )
            ds2['u_forcing_advection'] = (
                advected(ds1_downscaled, 'ufull') -
                downscaled(advected(ds1, 'ufull'))
            )
            ds2['v_forcing_advection'] = (
                advected(ds1_downscaled, 'vfull') -
                downscaled(advected(ds1, 'vfull'))
            )
            for dq in derivs:
                ds2[dq] = ds1_downscaled[dq]

            m1._step_forward()
            m2._step_forward()

            ds1_post = m1.to_dataset().copy(deep=True)
            ds2_post = m2.to_dataset().copy(deep=True)

            ds1['dqdt_post'] = xr.DataArray(
                npfft.irfftn(m1.dqhdt, axes=(-2,-1))[np.newaxis],
                coords=[ds1.coords[d] for d in spatial_dims]
            )

            ds2['dqdt_post'] = xr.DataArray(
                npfft.irfftn(m2.dqhdt, axes=(-2,-1))[np.newaxis],
                coords=[ds2.coords[d] for d in spatial_dims]
            )

            ds2['dqdt_post_hires_downscaled'] = xr.DataArray(
                downscaled(ds1).dqdt_post.data,
                coords=[ds2.coords[d] for d in spatial_dims]
            )

            ds2['q_forcing_model'] = (
                ds2['dqdt_post_hires_downscaled'] - 
                ds2['dqdt_post']
            )

            datasets1.append(ds1)
            datasets2.append(ds2)
        else:
            m1._step_forward()
            m2._step_forward()
    
    d1 = concat_and_convert(datasets1)
    d2 = concat_and_convert(datasets2)

    return d1, d2

class PYQGSubgridDataset():
    def __init__(self, data_dir='/scratch/zanna/data/pyqg/datasets/train', key='lores', skip=0):
        self.dataset = xr.open_mfdataset(f"{data_dir}/*/{key}.nc", combine="nested", concat_dim="run")
        self.extracted = {}
        
    @property
    def resolution(self):
        r1 = self.dataset.coords['x'].shape[0]
        r2 = self.dataset.coords['y'].shape[0]
        assert r1 == r2
        return r1
    
    @property
    def time(self):
        return self.dataset.time.data.reshape(-1,1).repeat(self.dataset.run.shape[0], axis=1).T.ravel()

    def extract_variable(self, var, z=None):
        key = (var, z)
        if key not in self.extracted:
            ds = self.dataset.isel(lev=z) if z is not None else self.dataset
            self.extracted[key] = np.array(ds[var].data.reshape(-1, self.resolution, self.resolution))
        return self.extracted[key]

    def extract_variables(self, vars, z=None):
        if isinstance(vars, str):
            vars = vars.split(",")
        return np.swapaxes(np.array([
            self.extract_variable(var, z) for var in vars
        ]), 0, 1)

    @property
    def q1(self): return self.extract_variable('q', 0)
    @property
    def q2(self): return self.extract_variable('q', 1)
    @property
    def u1(self): return self.extract_variable('ufull', 0)
    @property
    def u2(self): return self.extract_variable('ufull', 1)
    @property
    def v1(self): return self.extract_variable('vfull', 0)
    @property
    def v2(self): return self.extract_variable('vfull', 1)
    @property
    def Sq1(self): return self.extract_variable('q_forcing_advection', 0)
    @property
    def Sq2(self): return self.extract_variable('q_forcing_advection', 1)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str)
    parser.add_argument('--ag7531', type=int, default=0)
    parser.add_argument('--symbolic', type=int, default=0)
    parser.add_argument('--control', type=int, default=0)
    parser.add_argument('--physical', type=int, default=0)
    parser.add_argument('--sampling_freq', type=int, default=1000)
    parser.add_argument('--sampling_dist', type=str, default='uniform')
    parser.add_argument('--transfer_test', type=int, default=0)
    args, extra = parser.parse_known_args()

    # Setup parameters for dataset generation functions (all extras go into
    # pyqg)
    kwargs = dict(sampling_freq=args.sampling_freq, sampling_dist=args.sampling_dist)
    for param in extra:
        key, val = param.split('=')
        kwargs[key.replace('--', '')] = float(val)
    if args.transfer_test:
        kwargs.update(dict(rek=7.000000e-08, delta=0.1, beta=1.0e-11))

    os.system(f"mkdir -p {args.save_dir}")

    def save(ds, key):
        ds.to_netcdf(os.path.join(args.save_dir, f"{key}.nc"))

    # Run the dataset generation function and save the results to save_dir
    if args.physical:
        save(generate_physically_parameterized_dataset(**kwargs), 'physical')
    elif args.ag7531:
        if 'factor' in kwargs:
            save(generate_ag7531_parameterized_dataset(**kwargs), f"ag7531_{kwargs['factor']}")
        else:
            save(generate_ag7531_parameterized_dataset(**kwargs), 'ag7531')
    elif args.symbolic:
        save(generate_symbolic_regression_parameterized_dataset(**kwargs), 'symbolic')
    elif args.control:
        save(generate_control_dataset(**kwargs), 'control')
    else:
        hires, lores = generate_forcing_dataset(**kwargs)
        save(hires, 'hires')
        save(lores, 'lores')
