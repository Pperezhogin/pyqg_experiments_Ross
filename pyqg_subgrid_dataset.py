import os
import pyqg
import gcm_filters
import numpy as np
import xarray as xr
import pandas as pd
import json
import gc
import pickle

class cachedproperty(object):
  def __init__(self, function):
    self.__doc__ = getattr(function, '__doc__')
    self.function = function

  def __get__(self, instance, klass):
    if instance is None: return self
    value = instance.__dict__[self.function.__name__] = self.function(instance)
    return value

def Struct(**kwargs):
    from collections import namedtuple
    return namedtuple('Struct', ' '.join(kwargs.keys()))(**kwargs)

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

def generate_parameterized_dataset(cnn0, cnn1, inputs, **kwargs):
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn0.to(device)
    cnn1.to(device)

    def get_inputs(m,z):
        return np.array([[
            getattr(m,inp)[z]
            for inp in inputs.split(",")
        ]]).astype(np.float32)

    def q_parameterization(m):
        dq = np.array([
            cnn0.predict(get_inputs(m,0), device=device)[0,0],
            cnn1.predict(get_inputs(m,1), device=device)[0,0]
        ]).astype(m.q.dtype)
        return -dq

    return generate_control_dataset(q_parameterization=q_parameterization, **kwargs)

def generate_parameterized_dataset2(cnn0, cnn1, inputs, sampling_freq=1000, nx=64, dt=3600., **kwargs):
    import torch
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cnn0.to(device)
    cnn1.to(device)

    def get_inputs(m,z):
        return np.array([[
            getattr(m,inp)[z]
            for inp in inputs.split(",")
        ]]).astype(np.float32)
    
    def center(x):
        return x - x.mean()

    def q_parameterization(m):
        dq = np.array([
            center(-cnn0.predict(get_inputs(m,0), device=device)[0,0]),
            center(-cnn1.predict(get_inputs(m,1), device=device)[0,0])
        ]).astype(m.q.dtype)
        return dq

    year = 24*60*60*360.
    pyqg_kwargs = dict(tmax=5*year, tavestart=2.5*year, dt=dt)
    pyqg_kwargs.update(**kwargs)

    m = pyqg.QGModel(nx=nx, **pyqg_kwargs)
    datasets = []

    while m.t < m.tmax:
        dq = q_parameterization(m) * dt
        if m.tc % sampling_freq == 0:
            ds = m.to_dataset().copy(deep=True)
            datasets.append(ds)
        m._step_forward()
        m.set_q1q2(*(m.q + dq))

    d = xr.concat(datasets, dim='time')
    
    for k,v in datasets[-1].variables.items():
        if k not in d:
            d[k] = v.isel(time=-1)

    for k,v in d.variables.items():
        if v.dtype == np.float64:
            d[k] = v.astype(np.float32)

    return d

def generate_control_dataset(nx=64, dt=3600., sampling_freq=1000, sampling_dist='uniform', **kwargs):
    year = 24*60*60*360.
    pyqg_kwargs = dict(tmax=5*year, tavestart=2.5*year, dt=dt)
    pyqg_kwargs.update(**kwargs)

    m = pyqg.QGModel(nx=nx, **pyqg_kwargs)
    datasets = []

    while m.t < m.tmax:
        if m.tc % sampling_freq == 0:
            ds = m.to_dataset().copy(deep=True)
            datasets.append(ds)
        m._step_forward()

    d = xr.concat(datasets, dim='time')
    
    for k,v in datasets[-1].variables.items():
        if k not in d:
            d[k] = v.isel(time=-1)

    for k,v in d.variables.items():
        if v.dtype == np.float64:
            d[k] = v.astype(np.float32)

    return d

def generate_forcing_dataset(nx1=256, nx2=64, dt=3600., sampling_freq=1000, sampling_dist='uniform', filter=None, **kwargs):
    scale = nx1//nx2
    
    year = 24*60*60*360.
    pyqg_kwargs = dict(tmax=5*year, tavestart=2.5*year, dt=dt)
    pyqg_kwargs.update(**kwargs)
    
    if filter is None:
        filter = gcm_filters.Filter(filter_scale=scale, dx_min=1, grid_type=gcm_filters.GridType.REGULAR)

    def downscaled(ds):
        return filter.apply(ds, dims=['y','x']).coarsen(x=scale, y=scale).mean()

    m1 = pyqg.QGModel(nx=nx1, **pyqg_kwargs)
    m2 = pyqg.QGModel(nx=nx2, **pyqg_kwargs)
    #m3 = pyqg.QGModel(nx=nx2, **pyqg_kwargs)
    
    # Ensure dissipation on the lower-res simulation matches that of high-res simulation
    #cphi = 0.65 * np.pi
    #wvx = np.sqrt((m3.k * m1.dx)**2. + (m3.l * m1.dy)**2.)
    #filtr = np.exp(-m3.filterfac*(wvx - cphi))
    #filtr[wvx <= cphi] = 1.
    #m3.filtr = filtr

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
            ds1_downscaled = downscaled(ds1)
            m2.set_q1q2(*ds1_downscaled.q.isel(time=0).copy().data)
            #m3.set_q1q2(*ds1_downscaled.q.isel(time=0).copy().data)
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

            m1._step_forward()
            m2._step_forward()
            #m3._step_forward()

            ds1_downscaled2 = downscaled(m1.to_dataset()).isel(time=0)
            q_post = ds1_downscaled2.q.data
            u_post = ds1_downscaled2.ufull.data
            v_post = ds1_downscaled2.vfull.data

            ds2['q_forcing_model'] = xr.DataArray(
                np.expand_dims(m2.q - q_post, 0),
                coords=[ds2.coords[d] for d in ds2.q.coords]
            )

            ds2['u_forcing_model'] = xr.DataArray(
                np.expand_dims(m2.ufull - u_post, 0),
                coords=[ds2.coords[d] for d in ds2.q.coords]
            )

            ds2['v_forcing_model'] = xr.DataArray(
                np.expand_dims(m2.vfull - v_post, 0),
                coords=[ds2.coords[d] for d in ds2.q.coords]
            )

            datasets1.append(ds1)
            datasets2.append(ds2)
        else:
            m1._step_forward()
            m2._step_forward()
            #m3._step_forward()
    
    d1 = xr.concat(datasets1, dim='time')
    d2 = xr.concat(datasets2, dim='time')
    
    for k,v in datasets1[-1].variables.items():
        if k not in d1:
            d1[k] = v.isel(time=-1)
            
    for k,v in datasets2[-1].variables.items():
        if k not in d2:
            d2[k] = v.isel(time=-1)

    for d in [d1,d2]:
        for k,v in d.variables.items():
            if v.dtype == np.float64:
                d[k] = v.astype(np.float32)
            
    return d1, d2

class PYQGSubgridDataset(object):
    def __init__(self, data_dir='./pyqg_datasets', sampling_freq=1000, sampling_dist='uniform', **pyqg_kwargs):
        self.data_dir = data_dir
        self.config = dict(
            pyqg_kwargs=pyqg_kwargs,
            sampling_freq=sampling_freq,
            sampling_dist=sampling_dist,
        )

    @property
    def name(self):
        return self.data_dir.split('/')[-1]

    def run_with_model(self, net, res):
        def q_parameterization(run):
            print("CALLING MODEL")
            q = run.q.reshape(-1,res*res)
            dq = net.predict(q).reshape(run.q.shape)
            print(q.mean())
            print(dq.mean())
            return dq
        kws = self.config['pyqg_kwargs']
        kws.update(nx=res, q_parameterization=q_parameterization)
        run = pyqg.QGModel(**kws)
        run.run()
        return run

    @property
    def pyqg_run_dir(self):
        return self.data_dir

    def load(self, key, **kw):
        return xr.open_mfdataset(f"{self.pyqg_run_dir}/*/{key}.nc", combine="nested", concat_dim="run", **kw)

    def execute_control_run(self, i):
        config = Struct(**self.config)
        simulation_dir = os.path.join(self.pyqg_run_dir, str(i))
        os.system(f"mkdir -p {simulation_dir}")
        ds = generate_control_dataset(
                sampling_freq=config.sampling_freq,
                sampling_dist=config.sampling_dist,
                **config.pyqg_kwargs)
        complex_vars = [k for k,v in ds.variables.items() if v.dtype == np.complex128]
        ds = ds.drop(complex_vars)
        ds.to_netcdf(os.path.join(simulation_dir, 'control.nc'))

    def execute_forcing_run(self, i):
        config = Struct(**self.config)
        simulation_dir = os.path.join(self.pyqg_run_dir, str(i))
        os.system(f"mkdir -p {simulation_dir}")
        hires, lores = generate_forcing_dataset(
                sampling_freq=config.sampling_freq,
                sampling_dist=config.sampling_dist,
                **config.pyqg_kwargs)
        complex_vars = [k for k,v in hires.variables.items() if v.dtype == np.complex128]
        hires = hires.drop(complex_vars)
        lores = lores.drop(complex_vars)
        hires.to_netcdf(os.path.join(simulation_dir, 'hires.nc'))
        lores.to_netcdf(os.path.join(simulation_dir, 'lores.nc'))

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--run_idx', type=int)
    parser.add_argument('--control', type=int, default=0)
    parser.add_argument('--sampling_freq', type=int, default=1000)
    parser.add_argument('--sampling_dist', type=str, default='uniform')
    args, extra = parser.parse_known_args()

    kwargs = dict(sampling_freq=args.sampling_freq, data_dir=args.data_dir, sampling_dist=args.sampling_dist)
    for param in extra:
        key, val = param.split('=')
        kwargs[key.replace('--', '')] = float(val)
    idx = args.run_idx
    control = args.control
    print(args)
    print(kwargs)
    ds = PYQGSubgridDataset(**kwargs)
    if control:
        ds.execute_control_run(idx)
    else:
        ds.execute_forcing_run(idx)
