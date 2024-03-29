import pyqg
import numpy as np
import numpy.fft as npfft
import xarray as xr
import inspect
import operator
import re
from pyqg.diagnostic_tools import calc_ispec, calc_ispec_perezhogin
from collections import OrderedDict, defaultdict
from scipy.stats import linregress, wasserstein_distance
from sklearn.decomposition import PCA

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
    
    def expand_dims(self, *args, **kwargs):
        return self.__class__(self.ds.expand_dims(*args, **kwargs))

    def assign_attrs(self, **kw):
        self.ds = self.ds.assign_attrs(**kw)
        return self
    
    def load(self):
        self.ds = self.ds.load()
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
        """Extract a possibly-derived feature from the dataset using a string
        DSL to handle arbitrary combinations of features using addition,
        multiplication, and differentiation, with intermediate caching.
        
        For example,
            
            ds.extract_feature('mul(q, add(ddx(u), ddy(v)))')
            
        would be equivalent to
        
            q * (ds.ddx('u') + ds.ddy('v'))
            
        """
        def extract_pair(s):
            depth = 0
            for i, char in enumerate(s):
                if char == "(":
                    depth += 1
                elif char == ")":
                    depth -= 1
                elif char == "," and depth == 0:
                    return self.extract_feature(s[:i].strip()), self.extract_feature(s[i+1:].strip())
            raise ValueError(f"string {s} is not a comma-separated pair")
        
        if feature not in self.ds:
            match = re.search(f"^([a-z]+)\((.*)\)$", feature)
            if match:
                op, inner = match.group(1), match.group(2)
                if op in ['mul', 'add', 'sub']:
                    self.ds[feature] = getattr(operator, op)(*extract_pair(inner))
                elif op in ['neg', 'abs']:
                    self.ds[feature] = getattr(operator, op)(self.extract_feature(inner))
                elif op in ['div', 'curl']:
                    self.ds[feature] = getattr(self, op)(*extract_pair(inner))
                elif op in ['ddx', 'ddy']:
                    self.ds[feature] = getattr(self, op)(self.extract_feature(inner))
                else:
                    raise ValueError(f"could not interpret {feature}")
            elif feature == 'dqdt_through_lores':
                if 'dqdt' in self.ds:
                    self.ds[feature] = self.ds['dqdt']
                else:
                    self.ds[feature] = self.real_var(self.m.ifft(self.m.dqhdt))
            elif feature.endswith('1'):
                self.ds[feature] = self[feature[:-1]].isel(lev=0)
            elif feature.endswith('2'):
                self.ds[feature] = self[feature[:-1]].isel(lev=1)
            elif feature.startswith('ddx_'):
                self.ds[feature] = self.ddx(self.extract_feature(feature[4:]))
            elif feature.startswith('ddy_'):
                self.ds[feature] = self.ddy(self.extract_feature(feature[4:]))
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
        if len(arr.shape) == len(self.spec_0.shape):
            return self.spec_0 + arr
        elif len(arr.shape) == len(self.spec_0.shape)-1:
            return self.spec_0.isel(lev=0) + arr

    def real_var(self, arr):
        """Convert a real array to an xarray.DataArray"""
        if len(arr.shape) == len(self.real_0.shape):
            return self.real_0 + arr
        elif len(arr.shape) == len(self.real_0.shape)-1:
            return self.real_0.isel(lev=0) + arr

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

    def div(self, *xy):
        x, y = xy; return self.ddx(x) + self.ddy(y)

    def curl(self, *xy):
        x, y = xy; return self.ddx(y) - self.ddy(x)
    
    ###########################################
    #
    # Helpers for computing physical quantities
    #
    ###########################################
    
    @property
    def final_q(self):
        """Return the final value of the potential vorticity (for the final
        run, if there are multiple runs)"""
        if 'run' in self.dims:
            q = self.q.isel(time=-1, run=-1).data.astype(self.m.q.dtype)
        elif 'time' in self.dims:
            q = self.q.isel(time=-1).data.astype(self.m.q.dtype)
        else:
            q = self.q.data.astype(self.m.q.dtype)
        assert q.shape == self.m.q.shape
        return q    

    @property
    def nx(self):
        """Number of grid points in the x/y directions"""
        return self.m.nx
    
    def advected(self, q):
        """Apply the advection operator to the quantity `q` in spectral space
        (and inverting back), assuming incompressibility"""
        return self.ddx(self[q] * self.ufull) + self.ddy(self[q] * self.vfull)

    @cachedproperty
    def ke(self):
        return 0.5 * (self.ufull**2 + self.vfull**2)

    @cachedproperty
    def vorticity(self):
        return self.ddx('vfull') - self.ddy('ufull')

    @cachedproperty
    def enstrophy(self):
        return self.vorticity**2
    
    @property
    def en(self):
        return self.enstrophy

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

    def pca(self, x_array, z=0, nsamples = 10, ncomponents = 1, svd_solver='auto'):
        """
        x_array - any xarray with dimensions run x time x lev x Y x X
        z - specify level
        Method:
            1) Form array of nsamples x nfeatures, 
            where nsamples is formed from run*time dimensions,
            and nfeatures from Y*X dimensions
            2) apply pca
            3) Back transformation to nsamples x Y x X
        """
        Nruns, Ntimes, Nlev, Ny, Nx = x_array.shape

        if nsamples > Nruns * Ntimes:
            nsamples = Nruns * Ntimes
            print('nsamples is reduced to', nsamples)

        x = np.reshape(x_array.data[:,:,z,:,:], [Nruns*Ntimes, Ny*Nx]);
        idx = np.random.permutation(Nruns*Ntimes)[:nsamples]
        x = x[idx,:].astype('float64')

        print('Matrix for PCA: ', x.shape)

        if svd_solver == 'my':
            u, s, vh = np.linalg.svd(x, full_matrices = True)
            var_ratio = s / s.sum()
            nvars = var_ratio.size
            components = vh[0:nvars,:].reshape([nvars, Ny, Nx])
        else:        
            pca = PCA(n_components = ncomponents, svd_solver=svd_solver)
            pca.fit(x)

            var_ratio = pca.explained_variance_ratio_
            components = pca.components_.reshape([ncomponents, Ny, Nx])

        print('Var ratio = ', var_ratio[0:10])
        print('Sum var ratio = ', var_ratio.sum())
        print('Components shape = ', components.shape)

        return var_ratio, components

    def isotropic_spectrum_2darray(self, py_2darray):
        x = np.zeros((self.m.nz,self.m.ny,self.m.nx))
        x[0,:,:] = py_2darray.astype('float64')
        
        af  = self.m.fft(x)
                
        # lev x Y x X array
        af2 = (np.abs(af) ** 2 / self.m.M**2).astype('float32')
        af2 = af2[0,:,:]

        k, spectrum = calc_ispec_perezhogin(self.m, af2)

        return k, spectrum

    def isotropic_spectrum_compute(self, x_array, z=0):
        """
        x_array - any xarray with dimensions run x time x lev x Y x X
        z - specify level, or pass 'sum'
        Method:
            1) For each run and time compute fft, square coefficients and
            normalize it like in pyqg

            2) compute isotropic spectrum for each snapshot

            3) average

        """
        af2 = np.zeros_like(self.m.qh, dtype='float32')

        Nruns, Ntimes = x_array.shape[0:2]
        for r in range(Nruns):
            for t in range(Ntimes):
                a   = np.array(x_array.isel(run = r, time = t)).astype('float64')
                af  = self.m.fft(a)
                
                # lev x Y x X array
                af2 += (np.abs(af) ** 2 / self.m.M**2).astype('float32')

        af2 = af2 / (Nruns * Ntimes)

        if isinstance(z,int):
            af2 = af2[z,:,:]
        if z == 'sum':
            af2 = af2.sum(axis=0)
        
        k, spectrum = calc_ispec(self.m, af2)

        return k, spectrum

    def isotropic_spectrum(self, key, z=None, agg=np.mean, time_avg=True):
        """Compute the spectrum of a given diagnostic quantity in terms of the
        radial wavenumber (possibly averaged across snapshots at different
        simulation timesteps if `time_avg`, possibly at a given depth `z` (or
        summed if `z='sum'`) and aggregated across different simulation runs).
        """
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

    def calc_energy_budget(self, **kw):
        budget = OrderedDict()
        k, apegen = self.isotropic_spectrum('APEgenspec', **kw)
        budget['APE generation'] = apegen
        budget['APE flux'] = self.isotropic_spectrum('APEflux', **kw)[1]
        budget['KE flux'] = self.isotropic_spectrum('KEflux', **kw)[1]
        budget['Dissipation'] = self.isotropic_spectrum('Dissipation', **kw)[1]
        budget['Parameterization'] = self.isotropic_spectrum('paramspec', **kw)[1]
        budget['Residual'] = np.array(list(budget.values())).sum(axis=0)
        return k, budget
    
    @cachedproperty
    def energy_budget(self):
        return self.calc_energy_budget()
    
    @property
    def normalized_energy_budget(self):
        """Return elements of the energy budget normalized to account for the
        discrete Fourier transform"""
        k, budget = self.energy_budget
        normalized = {}
        for key, val in budget.items():
            normalized[key] = val / self.m.M**2
        return k, normalized
    
    @property
    def twothirds_nyquist_frequency_index(self):
        return np.argwhere(np.array(self.m.filtr)[0]<1)[0,0]
    
    @property
    def twothirds_nyquist_frequency(self):
        return self.k.data[self.twothirds_nyquist_frequency_index]

    def spectral_linregress(self, key, kmin=5e-5, kmax=None, **kw):
        """Run linear regression in log-log space for the spectrum given by
        `key` over a frequency range of `kmin` to `kmax`"""
        if kmax is None:
            kmax = self.twothirds_nyquist_frequency
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
    
    def decorrelation_timescale(self, other, **kw):
        params1 = dict(self.pyqg_params)
        params2 = dict(pse.Dataset.wrap(other).pyqg_params)
        
        m1 = pyqg.QGModel(**params1)
        m1.set_q1q2(*self.final_q)
        m1._invert()
        
        m2 = pyqg.QGModel(**params2)
        
        return time_until_uncorrelated(m1, m2, **kw)

    def distributional_distances(self, other):
        other = self.__class__.wrap(other)
        distances = defaultdict(dict)
        datasets = [self, other]

        for z in self.layers:
            for quantity in ['q','u','v','ke','en']:
                distributions = [ds[quantity].isel(lev=z, time=-1).data.ravel() for ds in datasets]
                distance = wasserstein_distance(*distributions)
                distances[f"{quantity}{z+1}_wasserstein_distance"] = distance
                
        kmax = min(
            self.twothirds_nyquist_frequency,
            other.twothirds_nyquist_frequency
        )

        def spectral_difference(k, spec1, spec2):
            j = np.argmin(np.abs(np.log(k) - np.log(kmax)))
            return np.sqrt(np.mean((spec1[:j] - spec2[:j])**2))

        def spectral_linregress(k, spectrum, kmin=5e-5):
            i = np.argmin(np.abs(np.log(k) - np.log(kmin)))
            j = np.argmin(np.abs(np.log(k) - np.log(kmax)))
            lr = linregress(np.log(k[i:j]), np.log(spectrum[i:j]))
            return lr
        
        for varname in ['KEspec', 'Ensspec']:
            for z in self.layers:
                k1, spec1 = self.isotropic_spectrum(varname, z=z) 
                k2, spec2 = other.isotropic_spectrum(varname, z=z)
                distances[f"{varname}{z+1}_curve_rmse"] = spectral_difference(k1, spec1, spec2)
                lr1 = spectral_linregress(k1, spec1)
                lr2 = spectral_linregress(k2, spec2)
                distances[f"{varname}{z+1}_slope_diff"] = lr1.slope - lr2.slope

        for varname in ['APEgenspec', 'APEflux', 'KEflux', 'Dissipation']:
            k1, spec1 = self.isotropic_spectrum(varname) 
            k2, spec2 = other.isotropic_spectrum(varname)
            distances[f"{varname}_curve_rmse"] = spectral_difference(k1, spec1/self.m.M**2, spec2/other.m.M**2)

        return distances
    
    @property
    def label(self):
        if 'label' in self.attrs:
            return self.attrs['label']
        elif 'plot_kwargs' in self.attrs and 'label' in self.attrs['plot_kwargs']:
            return self.attrs['plot_kwargs']['label']

def pyqg_kwargs_for(run):
    sig1 = inspect.signature(pyqg.Model.__init__)
    sig2 = inspect.signature(pyqg.QGModel.__init__)
    return dict([(k.split(':')[1], v) for k,v in run.attrs.items() 
                    if 'pyqg:' in k and 'nz' not in k and
                         (k.split(':')[1] in sig1.parameters or k.split(':')[1] in sig2.parameters)])

def pyqg_model_for(run):
    return pyqg.QGModel(log_level=0, **pyqg_kwargs_for(run))
