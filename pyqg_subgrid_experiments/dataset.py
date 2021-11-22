import pyqg
import numpy as np
import numpy.fft as npfft
import xarray as xr
import inspect
import re
from pyqg.diagnostic_tools import calc_ispec
from collections import OrderedDict, defaultdict
from scipy.stats import pearsonr, linregress, wasserstein_distance

class cachedproperty(object):
  def __init__(self, function):
    self.__doc__ = getattr(function, '__doc__')
    self.function = function

  def __get__(self, instance, klass):
    if instance is None: return self
    value = instance.__dict__[self.function.__name__] = self.function(instance)
    return value

class Dataset(object):
    @classmethod
    def wrap(kls, ds):
        if isinstance(ds, kls):
            return ds
        else:
            return kls(ds)

    def __init__(self, ds):
        if isinstance(ds, xr.Dataset):
            # Wrap an xarray dataset
            self.ds = ds
            self.m = pyqg_model_for(ds)
        elif isinstance(ds, pyqg.QGModel):
            # Wrap a pyqg model
            self.ds = ds.to_dataset().isel(time=-1)
            self.m = ds
        elif isinstance(ds, str):
            # Load an xarray dataset from a glob path
            self.ds = xr.open_mfdataset(ds, combine="nested", concat_dim="run")
            self.m = pyqg_model_for(self.ds)
        else:
            raise ValueError("must pass xr.Dataset, pyqg.QGModel, or glob path")

    @property
    def pyqg_params(self):
        return pyqg_kwargs_for(self.ds)

    ###########################################
    #
    # Helpers for delegating methods to xarray
    #
    ###########################################
        
    def __getattr__(self, name):
        return getattr(self.ds, name)

    def __getitem__(self, q):
        if isinstance(q, str):
            if q in self.__dict__:
                return self.__dict__[q]
            else:
                try:
                    return object.__getattribute__(self, q)
                except AttributeError:
                    return self.ds[q]
        elif isinstance(q, list) and isinstance(q[0], str):
            return self.ds[q]
        elif isinstance(q, xr.DataArray) or isinstance(q, np.ndarray):
            # slight hack to enable e.g. `advected('q')` and
            # `advected(q_array)` to both work
            return q
        else:
            raise KeyError(q)
    
    def __setitem__(self, key, val):
        self.ds[key] = val
    
    def __repr__(self):
        return f"wrapper around\n{self.ds}"

    def isel(self, *args, **kwargs):
        return self.__class__(self.ds.isel(*args, **kwargs))

    def assign_attrs(self, **kw):
        self.ds = self.ds.assign_attrs(**kw)
        return self

    def train_test_split(self, test_frac=0.25):
        assert 'run' in self.ds.dims
        N = len(self.ds.run)
        assert N > 1
        cutoff = int(N*test_frac)
        order = np.arange(N)
        np.random.shuffle(order)
        return self.isel(run=order[:cutoff]), self.isel(run=order[cutoff:])

    def extract_feature(self, feature):
        def split_by(s): 
            parts = feature.split(s)
            part1 = parts[0]
            part2 = s.join(parts[1:])
            return self.extract_feature(part1), self.extract_feature(part2)

        if feature not in self.ds:
            if '_times_' in feature:
                part1, part2 = split_by('_times_')
                self.ds[feature] = part1 * part2
            elif '_plus_' in feature:
                part1, part2 = split_by('_plus_')
                self.ds[feature] = part1 + part2
            elif '_minus_' in feature:
                part1, part2 = split_by('_minus_')
                self.ds[feature] = part1 - part2
            elif feature.startswith('ddx_'):
                self.ds[feature] = self.ddx(self.extract_feature(feature[4:]))
            elif feature.startswith('ddy_'):
                self.ds[feature] = self.ddy(self.extract_feature(feature[4:]))
            elif feature == 'dqdt_through_lores':
                if 'dqdt' in self.ds:
                    self.ds[feature] = self.ds['dqdt']
                else:
                    self.ds[feature] = self.real_var(self.m.ifft(self.m.dqhdt))
            else:
                raise ValueError(f"could not interpret {feature}")

        return self.ds[feature]

    ###########################################
    #
    # Helpers for spectral calculations
    #
    ###########################################
    
    @cachedproperty
    def spec_0(self):
        """Return a zeroed-out xarray.DataArray with spectral dimensions"""
        if 'qh' in self.ds:
            return self['qh'] * 0
        else:
            q = self['q']
            qh = npfft.rfftn(q, axes=(-2,-1))
            dims = [d.replace('y','l').replace('x','k') for d in q.dims]
            coords = {}
            for d in dims: coords[d] = self[d]
            return xr.DataArray(qh * 0, coords=coords, dims=dims)

    @cachedproperty
    def real_0(self):
        """Return a zeroed-out xarray.DataArray with real dimensions"""
        return self['q'] * 0

    def spec_var(self, arr):
        """Convert a spectral array to an xarray.DataArray"""
        return self.spec_0 + arr

    def real_var(self, arr):
        """Convert a real array to an xarray.DataArray"""
        return self.real_0 + arr

    def fft(self, x):
        """Convert real -> spectral"""
        return self.spec_var(npfft.rfftn(self[x], axes=(-2,-1)))
    
    def ifft(self, xh):
        """Convert spectral -> real"""
        return self.real_var(npfft.irfftn(self[xh], axes=(-2,-1)))
    
    def ddx(self, q):
        """Take x derivative in spectral space"""
        return self.ifft(self.fft(self[q]) * 1j * self.k)
    
    def ddy(self, q):
        """Take y derivative in spectral space"""
        return self.ifft(self.fft(self[q]) * 1j * self.l)

    ###########################################
    #
    # Helpers for computing physical quantities
    #
    ###########################################

    @property
    def nx(self):
        return self.m.nx
    
    def advected(self, q):
        return self.ddx(self[q] * self.ufull) + self.ddy(self[q] * self.vfull)

    @cachedproperty
    def ke(self):
        return 0.5 * (self.ufull**2 + self.vfull**2)

    @cachedproperty
    def vorticity(self):
        return self.ddx('vfull') - self.ddx('ufull')

    @cachedproperty
    def enstrophy(self):
        return self.vorticity**2

    @cachedproperty
    def relative_vorticity(self):
        return self.ddx('v') - self.ddy('u')

    @cachedproperty
    def shearing_deformation(self):
        return self.ddx('v') + self.ddy('u')

    @cachedproperty
    def stretching_deformation(self):
        return self.ddx('u') - self.ddy('v')

    @cachedproperty
    def zb2020_parameterization(self):
        # Eq. 6 of https://laurezanna.github.io/files/Zanna-Bolton-2020.pdf
        vort_shear = self.relative_vorticity * self.shearing_deformation
        vort_stretch = self.relative_vorticity * self.stretching_deformation
        sum_sq = (self.relative_vorticity**2
                + self.shearing_deformation**2
                + self.stretching_deformation**2) / 2.0

        du = self.ddx(sum_sq-vort_shear) + self.ddy(vort_stretch)
        dv = self.ddy(sum_sq+vort_shear) + self.ddx(vort_stretch)

        return du, dv

    ###########################################
    #
    # Helpers for extracting diagnostics
    #
    ###########################################

    def isotropic_spectrum(self, key, z=None, agg=np.mean, time_avg=True):
        if key == 'Dissipation':
            assert z is None
            m = self.m
            k, lower_ke = self.isotropic_spectrum('KEspec', z=-1, agg=agg, time_avg=time_avg)
            return k, -m.rek * m.del2 * m.M**2 * lower_ke
        elif key in self.ds and time_avg:
            spec = self[key]
        elif key in self.m.diagnostics:
            assert 'run' not in self.dims
            assert np.allclose(self.m.q, self.ds.isel(time=-1).q.data)
            diag = self.m.diagnostics[key]
            dims = diag['dims']
            spec = diag['function'](self.m)
            coords = {}
            for d in dims: coords[d] = self.ds.coords[d]
            spec = xr.DataArray(spec, dims=dims, coords=coords)
        else:
            k, dummy = self.isotropic_spectrum('KEflux')
            return k, np.zeros_like(dummy)

        if z == 'sum':
            spec = self[key].sum(dim='lev')
        elif z is not None:
            spec = self[key].isel(lev=z)

        if 'run' in self.dims:
            spectra = []
            for i in range(len(self.run)):
                k, spectrum = calc_ispec(self.m, spec.isel(run=i).data)
                spectra.append(spectrum)
            spectrum = np.array(spectra)
            if agg is not None:
                spectrum = agg(spectrum, axis=0)
        else:
            k, spectrum = calc_ispec(self.m, spec.data)

        return k, spectrum

    def energy_budget(self, **kw):
        budget = OrderedDict()
        k, apegen = self.isotropic_spectrum('APEgenspec', **kw)
        budget['APE generation'] = apegen
        budget['APE flux'] = self.isotropic_spectrum('APEflux', **kw)[1]
        budget['KE flux'] = self.isotropic_spectrum('KEflux', **kw)[1]
        budget['Dissipation'] = self.isotropic_spectrum('Dissipation', **kw)[1]
        budget['Parameterization'] = self.isotropic_spectrum('paramspec', **kw)[1]
        budget['Residual'] = np.array(list(budget.values())).sum(axis=0)
        return k, budget

    def spectral_linregress(self, key, kmin=5e-5, kmax=1.5e-4, **kw):
        k, spectrum = self.isotropic_spectrum(key, **kw)
        i = np.argmin(np.abs(np.log(k) - np.log(kmin)))
        j = np.argmin(np.abs(np.log(k) - np.log(kmax)))
        lr = linregress(np.log(k[i:j]), np.log(spectrum[i:j]))
        return lr

    @property
    def spectral_diagnostics(self):
        res = []
        for varname, var in self.variables.items():
            if 'l' in var.dims and 'k' in var.dims:
                if not np.iscomplexobj(var):
                    res.append(varname)
        return res

    @property
    def layers(self):
        return range(len(self.lev))

    def distributional_distances(self, other):
        other = self.__class__.wrap(other)
        distances = defaultdict(dict)
        datasets = [self, other]

        for z in self.layers:
            for quantity in ['q','u','v','ke','enstrophy','vorticity']:
                distributions = [ds[quantity].isel(lev=z, time=-1).data.ravel() for ds in datasets]
                distance = wasserstein_distance(*distributions)
                distances[f"{quantity}{z+1}_wasserstein_distance"] = distance

        def spectral_difference(k, spec1, spec2, kmax=1.5e-4):
            j = np.argmin(np.abs(np.log(k) - np.log(kmax)))
            return np.mean((spec1[:j] - spec2[:j])**2)

        def spectral_linregress(k, spectrum, kmin=5e-5, kmax=1.5e-4):
            i = np.argmin(np.abs(np.log(k) - np.log(kmin)))
            j = np.argmin(np.abs(np.log(k) - np.log(kmax)))
            lr = linregress(np.log(k[i:j]), np.log(spectrum[i:j]))
            return lr

        for varname in self.spectral_diagnostics:
            var = self[varname]
            if 'lev' in var.dims:
                for z in self.layers:
                    k1, spec1 = self.isotropic_spectrum(varname, z=z) 
                    k2, spec2 = other.isotropic_spectrum(varname, z=z)
                    distances[f"{varname}{z+1}_mean_difference"] = spectral_difference(k1, spec1, spec2)

                    if self[varname].min() >= 0 and other[varname].min() >= 0:
                        lr1 = spectral_linregress(k1, spec1)
                        lr2 = spectral_linregress(k2, spec2)
                        distances[f"{varname}{z+1}_loglog_slope_difference"] = lr1.slope - lr2.slope
                        distances[f"{varname}{z+1}_loglog_inter_difference"] = lr1.intercept - lr2.intercept
            else:
                k1, spec1 = self.isotropic_spectrum(varname) 
                k2, spec2 = other.isotropic_spectrum(varname)
                distances[f"{varname}_mean_difference"] = spectral_difference(k1, spec1, spec2)

                if self[varname].min() >= 0 and other[varname].min() >= 0:
                    lr1 = spectral_linregress(k1, spec1)
                    lr2 = spectral_linregress(k2, spec2)
                    distances[f"{varname}_loglog_slope_difference"] = lr1.slope - lr2.slope
                    distances[f"{varname}_loglog_inter_difference"] = lr1.intercept - lr2.intercept

        return distances

def pyqg_kwargs_for(run):
    sig1 = inspect.signature(pyqg.Model.__init__)
    sig2 = inspect.signature(pyqg.QGModel.__init__)
    return dict([(k.split(':')[1], v) for k,v in run.attrs.items() 
                    if 'pyqg:' in k and 'nz' not in k and
                         (k.split(':')[1] in sig1.parameters or k.split(':')[1] in sig2.parameters)])

def pyqg_model_for(run):
    return pyqg.QGModel(log_level=0, **pyqg_kwargs_for(run))
