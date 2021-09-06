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
            scale_factors=[2],
            **pyqg_kwargs):
        self.data_dir = data_dir
        self.sampler = lambda: SAMPLERS[sampling_mode](sampling_freq, sampling_delay)
        self.config = dict(
            n_runs=n_runs,
            pyqg_kwargs=pyqg_kwargs,
            scale_factors=scale_factors,
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

    @cachedproperty
    def persistent_order(self):
        path = os.path.join(self.data_dir, 'order.npy')
        if not os.path.exists(path):
            N = len(self.dataset.time_idxs)
            order = np.arange(N)
            self.random_state.shuffle(order)
            np.save(path, order)
        return np.load(path)

    def train_test_split(self, inputs='potential_vorticity',
            targets='potential_vorticity', test_frac=0.25):
        x = self.to_np(self.dataset.coarse_data[inputs])
        y = self.to_np(self.dataset.forcing_data[targets])
        order = self.persistent_order
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

    def run_with_model(self, net):
        def q_parameterization(run):
            print("CALLING MODEL")
            q = run.q.reshape(-1,res*res)
            dq = net.predict(q).reshape(run.q.shape)
            print(q.mean())
            print(dq.mean())
            return dq
        res = self.resolution
        kws = self.config['pyqg_kwargs']
        kws.update(nx=res, q_parameterization=q_parameterization)
        run = pyqg.QGModel(**kws)
        run.run()
        return run

    def _generate_dataset(self):
        os.system(f"mkdir -p {self.data_dir}")

        with open(self.path('config.json'), 'w') as f:
            f.write(json.dumps(self.config))

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

        for key, val in metadata._asdict().items():
            np.save(self.path(key+'.npy'), val)

        hi_res_data = xr.concat(datasets, 'batch')
        layers = sg.FluidLayer(hi_res_data, periodic=True)

        hires_data = layers.dataset
        hires_data.to_netcdf(self.path('hires_data.nc'))

        for sf in config.scale_factors:
            coarse_data = layers.downscaled(sf).dataset
            forcing_data = layers.subgrid_forcings(sf)
            coarse_data.to_netcdf(self.path(f"coarse_data{sf}.nc"))
            forcing_data.to_netcdf(self.path("forcing_data{sf}.nc"))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--n_runs', type=int, default=1)
    parser.add_argument('--scale_factors', type=str, default='2,4')
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
    kwargs['scale_factors'] = [int(sf) for sf in kwargs['scale_factors'].split(',')]

    ds = PYQGSubgridDataset(**kwargs)
    print(f"Generating {ds.data_dir}")
    print(kwargs)
    ds._generate_dataset()
    print("Done!")
