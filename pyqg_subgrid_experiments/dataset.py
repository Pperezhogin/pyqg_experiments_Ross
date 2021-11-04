import pyqg
import numpy as np
import xarray as xr
from scipy.stats import pearsonr
from scipy.stats import linregress
from pyqg.diagnostic_tools import calc_ispec

def time_until_uncorrelated(m1, m2, filter=None, thresh=0.5, perturbation_sd=1e-10, max_timesteps=100000):
    scale = m1.nx // m2.nx

    if scale > 1 and filter is None:
        import gcm_filters
        filter = gcm_filters.Filter(filter_scale=scale, dx_min=1, grid_type=gcm_filters.GridType.REGULAR)

    def possibly_downscaled_q(m):
        if scale == 1:
            return m1.q
        else:
            ds = m.to_dataset().isel(time=-1)
            downscaled = filter.apply(ds, dims=['y','x']).coarsen(x=scale, y=scale).mean()
            return downscaled.q.data
        
    def correlation(qa, qb):
        return pearsonr(qa.ravel(), qb.ravel())[0]
    
    for _ in range(3):
        m2.set_q1q2(*possibly_downscaled_q(m1))
        m1._step_forward()
        m2._step_forward()
        
    initial_conditions = possibly_downscaled_q(m1)
    perturbed_conditions = initial_conditions + np.random.normal(size=initial_conditions.shape) * perturbation_sd
    
    assert correlation(initial_conditions, perturbed_conditions) > 0.9
    
    m2.set_q1q2(*perturbed_conditions)
    
    steps = 0
    
    while steps < max_timesteps:
        m1._step_forward()
        m2._step_forward() 
        steps += 1
        corr = correlation(m2.q, possibly_downscaled_q(m1))
        if corr <= thresh:
            break
    
    return steps

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

    for i, ds in enumerate(datasets):
        ds['ke'] = ds.ufull**2 + ds.vfull**2
        ds['vorticity'] = (-ds.ufull.differentiate(coord='y') + ds.vfull.differentiate(coord='x'))
        ds['enstrophy'] = ds['vorticity']**2

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
