import pyqg
import numpy as np
import xarray as xr
from scipy.stats import pearsonr
from scipy.stats import linregress
from pyqg.diagnostic_tools import calc_ispec
import numpy.fft as npfft
import os

class Dataset(object):
    def __init__(self, ds):
        if isinstance(ds, xr.Dataset):
            self.ds = ds
            self.m = pyqg_model_for(ds)
        elif isinstance(ds, pyqg.QGModel):
            self.ds = ds.to_dataset().isel(time=-1)
            self.m = ds
        elif isinstance(ds, str):
            # TODO: support remote paths
            self.ds = xr.load_mfdataset(ds, combine="nested", concat_dim="run")
            self.m = pyqg_model_for(self.ds)
        else:
            raise ValueError("must pass xr.Dataset, pyqg.QGModel, or glob path")
    
    def __repr__(self):
        return f"wrapper around\n{self.ds}"
        
    def __getattr__(self, q):
        if q in ['m','ds']:
            return super().__getattr__(q)
        return self.ds[q]

    def __getitem__(self, q):
        if isinstance(q, str):
            return self.ds[q]
        elif isinstance(q, xr.DataArray) or isinstance(q, np.ndarray):
            # slight hack to enable e.g. `advected('q')` and
            # `advected(q_array)` to both work
            return q
        else:
            raise KeyError(q)
    
    def __setitem__(self, key, val):
        self.ds[key] = val

    def isel(self, *args, **kwargs):
        return self.__class__(self.ds.isel(*args, **kwargs))
    
    @property
    def spec_0(self):
        return self['qh'] * 0

    @property
    def real_0(self):
        return self['q'] * 0

    def spec_var(self, arr):
        return self.spec_0 + arr

    def real_var(self, arr):
        return self.real_0 + arr

    def fft(self, x):
        return self.spec_var(npfft.rfftn(self[x], axes=(-2,-1)))
    
    def ifft(self, xh):
        return self.real_var(npfft.irfftn(self[xh], axes=(-2,-1)))
    
    def ddx(self, q):
        return self.ifft(self.fft(self[q]) * 1j * self.k)
    
    def ddy(self, q):
        return self.ifft(self.fft(self[q]) * 1j * self.l)
    
    def advected(self, q):
        return self.ddx(self[q] * self.ufull) + self.ddy(self[q] * self.vfull)

    def train_test_split(self, test_frac=0.25):
        raise NotImplementedError

    @property
    def ke(self):
        return 0.5 * (self.ufull**2 + self.vfull**2)

    @property
    def vorticity(self):
        return self.ddx('vfull') - self.ddx('ufull')

    @property
    def enstrophy(self):
        return self.vorticity**2

    @property
    def relative_vorticity(self):
        return self.ddx('v') - self.ddx('u')

    @property
    def shearing_deformation(self):
        return self.ddx('u') - self.ddy('v')

    @property
    def stretching_deformation(self):
        return self.ddx('v') + self.ddy('u')

    @property
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

def pyqg_kwargs_for(run):
    import inspect
    sig1 = inspect.signature(pyqg.Model.__init__)
    sig2 = inspect.signature(pyqg.QGModel.__init__)
    return dict([(k.split(':')[1], v) for k,v in run.attrs.items() 
                    if 'pyqg' in k and 'nz' not in k and
                         (k.split(':')[1] in sig1.parameters or k.split(':')[1] in sig2.parameters)])

def pyqg_model_for(run):
    return pyqg.QGModel(log_level=0, **pyqg_kwargs_for(run))

def energy_budget(run):
    m = pyqg_model_for(run)
    
    def get_diagnostic(s):
        if s not in run and 'KEflux' in run:
            return np.zeros_like(run['KEflux'].data)
        else:
            return run[s].data
    
    KE1spec, KE2spec = get_diagnostic('KEspec')
    k, KE1spec = calc_ispec(m, KE1spec)
    k, KE2spec = calc_ispec(m, KE2spec)
    
    ebud = [get_diagnostic('APEgenspec') / m.M**2,
            get_diagnostic('APEflux')/ m.M**2,
            get_diagnostic('KEflux')/ m.M**2,
            -m.rek * m.del2 * get_diagnostic('KEspec')[1],
            get_diagnostic('paramspec') / m.M**2,
           ]
    ebud.append(-np.stack(ebud).sum(axis=0))
    ebud = [calc_ispec(m, term)[1] for term in ebud]
    ebud = np.stack(ebud)
        
    budget = OrderedDict()
    budget['APEgenspec'] = ebud[0]
    budget['APEflux'] = ebud[1]
    budget['KEflux'] = ebud[2]
    budget['KEspec'] = ebud[3]
    budget['Parameterization'] = ebud[4]
    budget['Residual'] = ebud[5]
    return k, budget

def power_spectrum(key, run, z=None):
    model = pyqg_model_for(run)
    data = run[key].data
    if z == 'sum': data = data.sum(axis=0)
    elif z is not None: data = data[z]
    return calc_ispec(model, data)

def median_spectra(quantity, ds, z=None):
    specs = []
    for i in range(len(ds.run)):
        k, spec = power_spectrum(quantity, ds.isel(run=i), z=z)
        specs.append(spec)                
    return k, np.percentile(specs, 50, axis=0)

def spectral_loglog_linregress(k, spectrum, kmin=5e-5, kmax=1.5e-4):
    i = np.argmin(np.abs(np.log(k) - np.log(kmin)))
    j = np.argmin(np.abs(np.log(k) - np.log(kmax)))
    lr = linregress(np.log(k[i:j]), np.log(spectrum[i:j]))
    return lr

def spectral_difference(k, spec1, spec2, kmax=1.5e-4):
    j = np.argmin(np.abs(np.log(k) - np.log(kmax)))
    return np.mean((spec1[:j] - spec2[:j])**2)

def compare_simulation_datasets(ds1, ds2):
    datasets = [ds1, ds2]

    comparisons = defaultdict(dict)

    layers = range(len(ds1.lev))

    for z in layers:
        for quantity in ['q','u','v','ke','enstrophy','vorticity']:
            distributions = [ds[quantity].isel(lev=z, time=-1).data.ravel() for ds in datasets]
            distance = wasserstein_distance(*distributions)
            comparisons[f"{quantity}{z+1}_wasserstein_distance"] = distance

    for varname, var in ds1.variables.items():
        if 'l' in var.dims and 'k' in var.dims:
            if 'lev' in var.dims:
                for z in layers:
                    k1, spec1 = median_spectra(varname, ds1, z=z) 
                    k2, spec2 = median_spectra(varname, ds2, z=z) 
                    comparisons[f"{varname}{z+1}_mean_difference"] = spectral_difference(k1, spec1, spec)

                    if ds1[varname].min() >= 0 and ds2[varname].min() >= 0:
                        lr1 = spectral_loglog_linregress(k1, spec1)
                        lr2 = spectral_loglog_linregress(k2, spec2)
                        comparisons[f"{varname}{z+1}_loglog_slope_difference"] = lr1.slope - lr2.slope
                        comparisons[f"{varname}{z+1}_loglog_inter_difference"] = lr1.intercept - lr2.intercept
            else:
                k1, spec1 = median_spectra(varname, ds1) 
                k2, spec2 = median_spectra(varname, ds2) 
                comparisons[f"{varname}_mean_difference"] = spectral_difference(k1, spec1, spec)

                if ds1[varname].min() >= 0 and ds2[varname].min() >= 0:
                    lr1 = spectral_loglog_linregress(k1, spec1)
                    lr2 = spectral_loglog_linregress(k2, spec2)
                    comparisons[f"{varname}_loglog_slope_difference"] = lr1.slope - lr2.slope
                    comparisons[f"{varname}_loglog_inter_difference"] = lr1.intercept - lr2.intercept

    return comparisons
