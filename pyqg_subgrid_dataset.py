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

def md5_hash(*args):
    import hashlib
    return hashlib.md5(json.dumps(args).encode('utf-8')).hexdigest()

def Struct(**kwargs):
    from collections import namedtuple
    return namedtuple('Struct', ' '.join(kwargs.keys()))(**kwargs)

class UniformSampler(object):
    def __init__(self, freq, delay=0):
        self.freq = freq
        self.delay = delay

    def sample_at(self, t):
        return t >= self.delay and t % self.freq == 0

class ExponentialSampler(object):
    def __init__(self, freq, delay=0):
        self.freq = freq
        self.reset(delay)

    def reset(self, t):
        self.next_time = t + int(np.random.exponential(self.freq))

    def sample_at(self, t):
        if t >= self.next_time:
            self.reset(t)
            return True
        else:
            return False

SAMPLERS = { 'uniform': UniformSampler, 'exponential': ExponentialSampler }

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

def generate_control_dataset(nx=64, dt=3600., sampling_freq=1000, **kwargs):
    year = 24*60*60*360.
    pyqg_kwargs = dict(tmax=10*year, tavestart=5*year, dt=dt)
    pyqg_kwargs.update(**kwargs)

    m = pyqg.QGModel(nx=nx, **pyqg_kwargs)
    datasets = []

    while m.t < m.tmax:
        if m.tc % sampling_freq == 0:
            ds = m.to_dataset().copy(deep=True)
            datasets.append(ds)
        m._step_forward()

    return xr.concat(datasets, dim='time')

def generate_forcing_dataset(nx1=256, nx2=64, dt=3600., sampling_freq=1000, filter=None, **kwargs):
    scale = nx1//nx2
    
    year = 24*60*60*360.
    pyqg_kwargs = dict(tmax=10*year, tavestart=5*year, dt=dt)
    pyqg_kwargs.update(**kwargs)
    
    if filter is None:
        filter = gcm_filters.Filter(filter_scale=scale, dx_min=1, grid_type=gcm_filters.GridType.REGULAR)

    def downscaled(ds):
        return filter.apply(ds, dims=['y','x']).coarsen(x=scale, y=scale).mean()

    m1 = pyqg.QGModel(nx=nx1, **pyqg_kwargs)
    m2 = pyqg.QGModel(nx=nx2, **pyqg_kwargs)
    m3 = pyqg.QGModel(nx=nx2, **pyqg_kwargs)
    
    # Ensure dissipation on the lower-res simulation matches that of high-res simulation
    cphi = 0.65 * np.pi
    wvx = np.sqrt((m3.k * m1.dx)**2. + (m3.l * m1.dy)**2.)
    filtr = np.exp(-m3.filterfac*(wvx - cphi))
    filtr[wvx <= cphi] = 1.
    m3.filtr = filtr
    
    datasets1 = []
    datasets2 = []
    
    while m1.t < m1.tmax:
        if m1.tc % sampling_freq == 0:
            ds1 = m1.to_dataset().copy(deep=True)
            ds1_downscaled = downscaled(ds1)
            m2.set_q1q2(*ds1_downscaled.q.isel(time=0).copy().data)
            m3.set_q1q2(*ds1_downscaled.q.isel(time=0).copy().data)
            ds2 = m2.to_dataset().copy(deep=True)
            ds2['q_forcing_advection'] = (
                downscaled(advected(ds1, 'q')) -
                advected(ds1_downscaled, 'q')
            )
            m1._step_forward()
            m2._step_forward()
            m3._step_forward()
            q_post = downscaled(m1.to_dataset()).q.isel(time=0).data
            ds2['q_forcing_diff_dissipation'] = xr.DataArray(
                np.expand_dims(m2.q - q_post, 0) / dt,
                coords=[ds2.coords[d] for d in ds2.q.coords]
            )
            ds2['q_forcing_same_dissipation'] = xr.DataArray(
                np.expand_dims(m3.q - q_post, 0) / dt,
                coords=[ds2.coords[d] for d in ds2.q.coords]
            )
            datasets1.append(ds1)
            datasets2.append(ds2)
        else:
            m1._step_forward()
            m2._step_forward()
            m3._step_forward()
    
    return (xr.concat(datasets1, dim='time'),
            xr.concat(datasets2, dim='time'))

class PYQGSubgridDataset(object):
    def __init__(self, data_dir='./pyqg_datasets', n_runs=5, sampling_freq=1000, sampling_mode='uniform', sampling_delay=0,
            scale_factors=[2], samples_per_timestep=1,
            **pyqg_kwargs):
        self.data_dir = data_dir
        self.sampler = lambda: SAMPLERS[sampling_mode](sampling_freq, sampling_delay)
        self.config = dict(
            n_runs=n_runs,
            pyqg_kwargs=pyqg_kwargs,
            scale_factors=scale_factors,
            sampling_freq=sampling_freq,
            sampling_mode=sampling_mode,
            sampling_delay=sampling_delay,
            samples_per_timestep=samples_per_timestep
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
        return os.path.join(self.data_dir, 'pyqg_runs')#, md5_hash(self.config['pyqg_kwargs']))

    def load(self, key, **kw):
        return xr.open_mfdataset(f"{self.pyqg_run_dir}/*/{key}.nc", combine="nested", concat_dim="run", **kw)

    def execute_control_run(self, i):
        config = Struct(**self.config)
        simulation_dir = os.path.join(self.pyqg_run_dir, str(i))
        os.system(f"mkdir -p {simulation_dir}")
        ds = generate_control_dataset(sampling_freq=config.sampling_freq, **config.pyqg_kwargs)
        complex_vars = [k for k,v in ds.variables.items() if v.dtype == np.complex128]
        ds = ds.drop(complex_vars)
        ds.to_netcdf(os.path.join(simulation_dir, 'control.nc'))

    def execute_forcing_run(self, i):
        config = Struct(**self.config)
        simulation_dir = os.path.join(self.pyqg_run_dir, str(i))
        os.system(f"mkdir -p {simulation_dir}")
        hires, lores = generate_forcing_dataset(sampling_freq=config.sampling_freq, **config.pyqg_kwargs)
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
    args, extra = parser.parse_known_args()

    year = 24*60*60*360.
    kwargs = dict(tmax=10*year, tavestart=5*year)
    kwargs.update(args.__dict__)
    for param in extra:
        key, val = param.split('=')
        kwargs[key.replace('--', '')] = float(val)
    idx = kwargs.pop('run_idx')
    control = kwargs.pop('control')
    ds = PYQGSubgridDataset(**kwargs)
    if control:
        ds.execute_control_run(idx)
    else:
        ds.execute_forcing_run(idx)
