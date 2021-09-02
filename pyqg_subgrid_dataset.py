import os
import pyqg
import numpy as np
import xarray as xr
import subgrid_forcing_tools as sg
import json

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
    def __init__(self, data_dir='./pyqg_datasets', n_runs=5, sampling_freq=100, sampling_mode='exponential', sampling_delay=0,
            scale_factor=2,
            **pyqg_kwargs):
        self.data_dir = data_dir
        self.sampler = lambda: SAMPLERS[sampling_mode](sampling_freq, sampling_delay)
        self.config = dict(
            n_runs=n_runs,
            scale_factor=scale_factor,
            pyqg_kwargs=pyqg_kwargs,
            sampling_freq=sampling_freq,
            sampling_mode=sampling_mode,
            sampling_delay=sampling_delay
        )

    def path(self, f):
        return os.path.join(self.data_dir, f)

    @property
    def name(self):
        return self.data_dir.split('/')[-1]

    @cachedproperty
    def random_state(self):
        return np.random.RandomState(seed=0)

    @property
    def resolution(self):
        return self.dataset.coarse_data.potential_vorticity.shape[-1]

    def to_np(self, d):
        return d.data.astype(np.float32).reshape(-1, self.resolution**2)

    def train_test_split(self, inputs='potential_vorticity',
            targets='potential_vorticity', test_frac=0.25):
        x = self.to_np(self.dataset.coarse_data[inputs])
        y = self.to_np(self.dataset.forcing_data[targets])
        order = np.arange(len(x))
        self.random_state.shuffle(order)
        split_at = int(len(x)*test_frac)
        train = order[split_at:]
        test = order[:split_at]
        return x[train], x[test], y[train], y[test]

    @cachedproperty
    def dataset(self):
        if not os.path.exists(self.data_dir):
            self._generate_dataset()
        data = {}
        for f in os.listdir(self.data_dir):
            parts = f.split('.')
            if len(parts) == 2:
                name, ext = parts
                if ext == 'npy':
                    data[name] = np.load(self.path(f))
                elif ext == 'nc':
                    data[name] = xr.open_dataset(self.path(f))
                elif ext == 'json':
                    with open(self.path(f), 'r') as ff:
                        data[name] = json.load(ff)
        return Struct(**data)

    def _generate_dataset(self):
        datasets = []
        metadata = Struct(run_idxs=[], time_idxs=[], time_vals=[], layer_idxs=[])
        config = Struct(**self.config)

        for run_idx in range(config.n_runs):
            model = pyqg.QGModel(**config.pyqg_kwargs)
            kw = dict(dims=('x','y'), coords={'x': model.x[0], 'y': model.y[:,0]})
            t = 0
            sampler = self.sampler()

            while model.t < model.tmax:
                model._step_forward()

                if sampler.sample_at(t):
                    for layer in range(len(model.u)):
                        u = xr.DataArray(model.ufull[layer], **kw)
                        v = xr.DataArray(model.vfull[layer], **kw)
                        q = xr.DataArray(model.q[layer], **kw)
                        datasets.append(xr.Dataset(data_vars=dict(
                            x_velocity=u, y_velocity=v, potential_vorticity=q)))
                        metadata.run_idxs.append(run_idx)
                        metadata.time_idxs.append(t)
                        metadata.time_vals.append(model.t)
                        metadata.layer_idxs.append(layer)

                t += 1

        hi_res_data = xr.concat(datasets, 'batch')
        layers = sg.FluidLayer(hi_res_data)

        coarse_data = layers.downscaled(config.scale_factor).dataset
        forcing_data = layers.subgrid_forcings(config.scale_factor)

        os.system(f"mkdir -p {self.data_dir}")
        coarse_data.to_netcdf(self.path('coarse_data.nc'))
        forcing_data.to_netcdf(self.path('forcing_data.nc'))
        for key, val in metadata._asdict().items():
            np.save(self.path(key+'.npy'), val)
        with open(self.path('config.json'), 'w') as f:
            f.write(json.dumps(self.config))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--sampling_freq', type=int, default=1)
    parser.add_argument('--sampling_mode', type=str, default='uniform')
    parser.add_argument('--sampling_delay', type=int, default=0)
    args, extra = parser.parse_known_args()

    year = 24*60*60*360.
    kwargs = dict(tmax=10*year, twrite=10000, tavestart=5*year)
    kwargs.update(args.__dict__)
    for param in extra:
        key, val = param.split('=')
        kwargs[key.replace('--', '')] = float(val)

    ds = PYQGSubgridDataset(**kwargs)
    print(f"Generating {ds.data_dir}")
    print(kwargs)
    ds._generate_dataset()
    print("Done!")
