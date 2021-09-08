import os
import pyqg
import numpy as np
import xarray as xr
import pandas as pd
import subgrid_forcing_tools as sg
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

    def _generate_dataset(self):
        config = Struct(**self.config)
        pyqg_dir = self.pyqg_run_dir

        os.system(f"mkdir -p {pyqg_dir}")

        with open(os.path.join(self.data_dir, 'config.json'), 'w') as f:
            f.write(json.dumps(self.config))

        with open(os.path.join(pyqg_dir, 'params.json'), 'w') as f:
            f.write(json.dumps(config.pyqg_kwargs))

        for run_idx in range(config.n_runs):
            gc.collect()

            simulation_dir = os.path.join(pyqg_dir, str(run_idx))

            print(simulation_dir)

            if os.path.exists(simulation_dir):
                print("loading")
                simulation = xr.open_dataset(os.path.join(simulation_dir, 'simulation.nc'))
            else:
                print("running")
                os.system(f"mkdir -p {simulation_dir}")

                model = pyqg.QGModel(**config.pyqg_kwargs)
                kw = dict(dims=('x','y'), coords={'x': model.x[0], 'y': model.y[:,0]})
                sampler = self.sampler()

                timevals = []
                datasets = []

                zvals = pd.Index(np.array(['U', 'L']), name='z')

                while model.t < model.tmax:
                    if sampler.sample_at(model.tc):
                        for _ in range(config.samples_per_timestep):
                            layers = []
                            for layer in range(len(model.u)):
                                u = xr.DataArray(model.ufull[layer], **kw)
                                v = xr.DataArray(model.vfull[layer], **kw)
                                q = xr.DataArray(model.q[layer], **kw)
                                layers.append(xr.Dataset(data_vars=dict(
                                    x_velocity=u, y_velocity=v, potential_vorticity=q)))
                            datasets.append(xr.concat(layers, zvals))
                            timevals.append(model.t)
                            model._step_forward()
                    else:
                        model._step_forward()

                timevals = pd.Index(np.array(timevals), name='time')
                simulation = xr.concat(datasets, timevals)
                simulation.to_netcdf(os.path.join(simulation_dir, 'simulation.nc'))

                diagnostics_dir = os.path.join(simulation_dir, "diagnostics")
                os.system(f"mkdir -p {diagnostics_dir}")

                for k, v in model.diagnostics.items():
                    np.save(os.path.join(diagnostics_dir, f"{k}.npy"), v['value'])

                model_attrs = {}
                for k, v in model.__dict__.items():
                    if isinstance(v, float) or isinstance(v, int):
                        model_attrs[k] = v

                with open(os.path.join(simulation_dir, "model_attrs.json"), 'w') as f:
                    f.write(json.dumps(model_attrs))


            layers = sg.FluidLayer(simulation, periodic=True)

            for sf in config.scale_factors:
                coarse_data = layers.downscaled(sf).dataset
                coarse_data.to_netcdf(os.path.join(simulation_dir, f"coarse{sf}.nc"))
                del coarse_data

                forcing_data = layers.subgrid_forcings(sf)
                forcing_data.to_netcdf(os.path.join(simulation_dir, f"forcings{sf}.nc"))
                del forcing_data

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--nx', type=int, default=64)
    parser.add_argument('--scale_factors', type=str, default='2,4')
    parser.add_argument('--sampling_freq', type=int, default=1)
    parser.add_argument('--sampling_mode', type=str, default='uniform')
    parser.add_argument('--sampling_delay', type=int, default=0)
    parser.add_argument('--samples_per_timestep', type=int, default=1)
    args, extra = parser.parse_known_args()

    year = 24*60*60*360.
    kwargs = dict(tmax=10*year, twrite=10000, tavestart=5*year)
    kwargs.update(args.__dict__)
    for param in extra:
        key, val = param.split('=')
        kwargs[key.replace('--', '')] = float(val)
    if len(kwargs['scale_factors']):
        kwargs['scale_factors'] = [int(sf) for sf in kwargs['scale_factors'].split(',')]
    else:
        kwargs['scale_factors'] = []


    ds = PYQGSubgridDataset(**kwargs)
    print(f"Generating {ds.data_dir}")
    print(kwargs)
    ds._generate_dataset()
    print("Done!")
