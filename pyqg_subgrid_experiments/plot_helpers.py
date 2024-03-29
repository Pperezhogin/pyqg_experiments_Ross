import os
import matplotlib
if 'USE_AGG' in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import xarray as xr
import numpy as np
from scipy.stats import linregress
import pyqg
from pyqg.errors import DiagnosticNotFilledError
from pyqg.particles import GriddedLagrangianParticleArray2D
from numpy import pi
from scipy.stats import gaussian_kde, wasserstein_distance
from collections import defaultdict, OrderedDict
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

import pyqg_subgrid_experiments as pse

def PDF_histogram(x, xmin = None, xmax = None, Nbins = None, bandwidth = None):
    """
    x is 1D numpy array with data

    How to use:
        first apply without arguments
        Then adjust xmin, xmax, Nbins or bandwidth
    """    
    N = x.shape[0]

    if xmin is None:
        xmin = x.min()
    if xmax is None:
        xmax = x.max()

    if Nbins is None:
        Nbins = 20

    if bandwidth is not None:
        Nbins = int(np.floor(xmax - xmin) / bandwidth)

    bandwidth = (xmax - xmin) / Nbins
    
    hist, bin_edges = np.histogram(x, range=(xmin,xmax), bins = Nbins)

    # hist / N is probability to go into bin
    # probability / bandwidth = probability density
    density = hist / N / bandwidth

    # we assign one values to each bin
    points = (bin_edges[0:-1] + bin_edges[1:]) * 0.5

    print(f"Number of bins = {Nbins}, over the interval ({xmin},{xmax}), with bandwidth = {bandwidth}")
    print(f"This interval covers {sum(hist)/N} of total probability")
    
    return density, points

def calc_ispec(model, ph, lo_mult=1.5):
    """Compute isotropic spectrum `phr` of `ph` from 2D spectrum.
    Parameters
    ----------
    model : pyqg.Model instance
        The model object from which `ph` originates
    ph : complex array
        The field on which to compute the variance
    Returns
    -------
    kr : array
        isotropic wavenumber
    phr : array
        isotropic spectrum
    """

    if model.kk.max()>model.ll.max():
        kmax = model.ll.max()
    else:
        kmax = model.kk.max()

    # create radial wavenumber
    dkr = np.sqrt(model.dk**2 + model.dl**2)
    
    kr =  np.arange(dkr*lo_mult,kmax+dkr,dkr)
    phr = np.zeros(kr.size)

    for i in range(kr.size):
        fkr =  (model.wv>=kr[i]-dkr/2) & (model.wv<=kr[i]+dkr/2)
        dth = pi / (fkr.sum()-1)
        phr[i] = ph[fkr].sum() * kr[i] * dth

    return kr, phr

class AnimatedPlot():
    def __init__(self, ax, func):
        self.ax = ax
        self.func = func

    @property
    def x(self):
        x = self.func()
        if isinstance(x, xr.DataArray): x = x.data
        return x

class AnimatedSpectrum(AnimatedPlot):
    @property
    def started_averaging(self):
        return self.m.diagnostics['KEspec']['count'] > 0

    def __init__(self, ax, m, spec, logy=True, fit_opts={}, label=None):
        def maybe_sum(arr):
            if len(arr.shape) == 3:
                return arr.sum(axis=0)
            else:
                return arr
                
        def get_curr():
            if spec == 'Diss.':
                k, s = calc_ispec(m, m.diagnostics['KEspec']['function'](m)[1])
                return k, -m.rek * m.del2 * s * m.M**2
            else:
                return calc_ispec(m, maybe_sum(m.diagnostics[spec]['function'](m)))

        def get_mean():
            try:
                if spec == 'Diss.':
                    k, s = calc_ispec(m, m.get_diagnostic('KEspec')[1])
                    return k, -m.rek * m.del2 * s * m.M**2
                else:
                    return calc_ispec(m, maybe_sum(m.get_diagnostic(spec)))
            except DiagnosticNotFilledError:
                return get_curr()

        self.m = m
        self.ax = ax
        self.spec = spec
        self.curr_line = AnimatedLine(ax, get_curr, ls='--', logy=logy)
        if label is None:
            label = spec
        self.mean_line = AnimatedLine(ax, get_mean, lw=3, logy=logy, show_best_fit=logy, fit_opts=fit_opts, color=self.curr_line.color, label=label)
        self.ax.set_xlabel("Wavenumber $k$", fontsize=14)
        self.logy = logy
        #self.ax.set_ylabel(f"{spec}", fontsize=14)

    def animate(self):
        res = []
        avg = self.started_averaging
        self.mean_line.set_alpha(int(avg))
        res += self.curr_line.animate()
        res += self.mean_line.animate()
        if not self.logy:
            if self.started_averaging:
                self.ylim = self.mean_line.ylim
            else:
                self.ylim = self.curr_line.ylim
        return res
        
class AnimatedLine(AnimatedPlot):
    def __init__(self, ax, func, logx=True, logy=True, show_best_fit=False, fit_opts={}, **kw):
        super().__init__(ax, func)
        x, y = self.x
        self.line = ax.plot(x, y, **kw)[0]
        self.best_fit = None
        self.fit_text = None
        self.show_best_fit = show_best_fit
        self.logx = logx
        self.logy = logy
        self.fit_opts = fit_opts

        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')

        if self.show_best_fit:
            self.loglog_fit(x, y, **fit_opts)

    def set_alpha(self, alpha):
        self.line.set_alpha(alpha)
        if self.best_fit is not None:
            self.best_fit.set_alpha(alpha)
            self.fit_text.set_alpha(alpha)

    @property
    def color(self):
        return self.line._color

    def loglog_fit(self, x, y, fudge=1.5, offset=4, mult=5):
        i = np.argmax(y) + offset
        j = np.argmin(np.abs(np.log(x)-np.log(x[i]*mult)))
        line_x = x[i:]
        if y.min() <= 0:
            line_y = y
            text_t = ""
        else:
            lr = linregress(np.log(x[i:j]), np.log(y[i:j]))
            self.ax.set_xlim(x[i]/10,x[j]*10)
            self.ax.set_ylim(y[j]/10,y[i]*10)
            line_y = np.exp(np.log(x[i:]) * lr.slope + lr.intercept)*fudge
            text_t = r"$\propto k^{"+f"{lr.slope:.2f}"+"}$"
        text_x = x[i]*fudge
        text_y = y[i]
        if self.best_fit is None:
            self.best_fit = self.ax.plot(line_x, line_y, ls='--', color='gray', lw=3, alpha=0.5)[0]
            self.fit_text = self.ax.text(text_x, text_y, text_t, color='gray', fontsize=14)
        else:
            self.fit_text.set_x(text_x)
            self.fit_text.set_y(text_y)
            self.fit_text.set_text(text_t)
            self.best_fit.set_data(line_x, line_y)

    def animate(self):
        x, y = self.x
        self.line.set_data(x,y)
        res = [self.line]

        self.ymin = y.min()
        self.ymax = y.max()
        self.xmin = x.min()
        self.xmax = x.max()
        if not self.logx:
            self.xlim = max(self.xmax, -self.xmin)
        if not self.logy:
            self.ylim = max(self.ymax, -self.ymin)

        if self.show_best_fit:
            self.loglog_fit(x, y, **self.fit_opts)
            res = res + [self.best_fit, self.fit_text]
        return res

class AnimatedScatterplot(AnimatedPlot):
    def __init__(self, ax, func, **kw):
        super().__init__(ax, func)
        x, y = self.func()
        self.scatter = ax.scatter(x, y, **kw)

    def animate(self):
        self.scatter.set_offsets(np.array(self.func()).T)
        return [self.scatter]

class AnimatedImage(AnimatedPlot):
    def __init__(self, ax, func, min_vmin=-float('inf'), max_vmax=float('inf'), cbar=True):
        super().__init__(ax, func)
        x = self.x
        ax.set_xticks([])
        ax.set_yticks([])
        self.max_vmax = max_vmax
        self.vmax = min(np.percentile(np.abs(x).ravel(), 99)*1.01, self.max_vmax)
        if min_vmin == 0:
            self.vmin = 0
            self.cmap = 'inferno'
        else:
            self.vmin = -self.vmax
            self.cmap = 'bwr'
        self.im = ax.imshow(x, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
        if cbar:
            self.cb = ax.figure.colorbar(self.im, ax=ax)

    def animate(self):
        x = self.x
        xmax = np.percentile(np.abs(x)*1.01, 99)
        if xmax > self.vmax:
            self.vmax = min(xmax, self.max_vmax)
        if self.vmin < 0:
            self.vmin = -self.vmax
        self.im.set_data(x)
        self.im.set_clim(self.vmin, self.vmax)
        return [self.im]

class AnimatedSimulationGroup(object):
    def ds(self, i):
        return self.datasets[i]

    def q(self, i, z):
        return self.ds(i).isel(lev=z).q.data

    def ke(self, i):
        return self.ds(i).ke.sum(dim='lev').data

    def en(self, i):
        return self.ds(i).enstrophy.sum(dim='lev').data

    def kespec(self, i):
        try:
            return self.ds(i).isotropic_spectrum('KEspec', z='sum')
        except:
            m = self.models[i]
            return calc_ispec(m, m.diagnostics['KEspec']['function'](m).sum(axis=0))

    def __init__(self, models, labels=None, steps_per_frame=100, title=None):
        fig = plt.figure(figsize=(16.75,3.25*len(models)))

        if title is not None:
            fig.suptitle(title, fontsize=20)

        gs = fig.add_gridspec(len(models)*2, 2+2+2+3)

        self.fig = fig
        self.gs = gs
        self.steps_per_frame = steps_per_frame

        def titleize(sp, title):
            ax = fig.add_subplot(sp)
            ax.set_title(title, fontsize=16)
            return ax

        axes = dict(spect=titleize(gs[:,6:], "Kinetic energy spectra"))
        
        self.datasets = [pse.Dataset.wrap(m) for m in models]
        self.models = [ds.m for ds in self.datasets]
        if labels is None:
            self.labels = [ds.label for ds in self.datasets]
        else:
            self.labels = labels
        
        anims = []

        for i in range(len(models)):
            axes[f"pv{i}"] = titleize(gs[i*2:(i+1)*2,0:2], ("" if i else "Upper PV"))
            axes[f"ke{i}"] = titleize(gs[i*2:(i+1)*2,2:4], ("" if i else "Kinetic energy"))
            axes[f"en{i}"] = titleize(gs[i*2:(i+1)*2,4:6], ("" if i else "Enstrophy"))

            anims += [
                AnimatedImage(axes[f"pv{i}"], lambda j=i: self.q(j, 0), cbar=False),
                AnimatedImage(axes[f"ke{i}"], lambda j=i: self.ke(j), cbar=False, min_vmin=0),
                AnimatedImage(axes[f"en{i}"], lambda j=i: self.en(j), cbar=False, min_vmin=0),
                AnimatedLine(axes["spect"], lambda j=i: self.kespec(j), lw=3, label=labels[i])
            ]

            axes[f"pv{i}"].set_ylabel(labels[i], rotation=0, va='center', ha='right', fontsize=16)
                                         
        axes['spect'].set_ylim(1e-11, 3e-9)
        axes['spect'].set_xlim(1e-5,  2e-4)
        axes['spect'].legend(loc='best', fontsize=16)
        axes['spect'].yaxis.tick_right()
        axes['spect'].yaxis.set_label_position("right")
        axes['spect'].set_ylabel(r"KE spectrum $\kappa^2 |\hat{\psi}|^2 (L / \Delta x)^2$ [$m^2 s^-2$]", fontsize=14)
        axes['spect'].set_xlabel(r"Radial wavenumber $\kappa$ [$m^-1$]", fontsize=14)

        self.anims = anims
        self.axes = axes

    def animate(self):
        for _ in range(self.steps_per_frame):
            for m in self.models:
                m._step_forward()
        self.datasets = [pse.Dataset(m) for m in self.models]
        res = []
        for anim in self.anims:
            res += anim.animate()
        return res

class ModelWithParticles():
    def __init__(self, m, grid_side=8):
        self.m = m
        self.grid_side = grid_side
        self.parts1 = self.init_particles()
        self.parts2 = self.init_particles()
        
        self.U1_old, self.U2_old = self.m.ufull
        self.V1_old, self.V2_old = self.m.vfull
        
    def init_particles(self):
        n = self.m.nx // self.grid_side
        return GriddedLagrangianParticleArray2D(
            self.m.x[:,::n][::n],
            self.m.y[::n,:][:,::n],
            self.m.nx,
            self.m.nx,
            periodic_in_x=True,
            periodic_in_y=True,
            xmin=0,
            xmax=self.m.L,
            ymin=0,
            ymax=self.m.L
        )
        
    def normalize_particle_coords(self, x, y):
        m = self.m
        xi = ((x % m.L) / m.L) * m.nx
        yi = ((y % m.L) / m.L) * m.nx
        return xi, yi
    
    @property
    def normalized_particle_coords(self):
        return [
            self.normalize_particle_coords(self.parts1.x, self.parts1.y),
            self.normalize_particle_coords(self.parts2.x, self.parts2.y),
        ]
       
    def _step_forward(self):
        self.m._step_forward()
        U1_new, U2_new = self.m.ufull
        V1_new, V2_new = self.m.vfull
        self.parts1.step_forward_with_gridded_uv(self.U1_old, self.V1_old, U1_new, V1_new, self.m.dt)
        self.parts2.step_forward_with_gridded_uv(self.U2_old, self.V2_old, U2_new, V2_new, self.m.dt)
        self.U1_old, self.U2_old = self.m.ufull
        self.V1_old, self.V2_old = self.m.vfull

def animate_simulation(m, n_frames=100, steps_per_frame=100, label=None, suptitle_y=0.95, fs=16, t0=None, fit_opts={}):
    #mp = ModelWithParticles(m)
    b = 4
    year = 24*60*60*360.
    fig = plt.figure(figsize=(4.5*b, 2*b))
    gs = fig.add_gridspec(2, 4)
    if t0 is None:
        t0 = m.t

    def titleize(sp, title):
        ax = fig.add_subplot(sp)
        ax.set_title(title, fontsize=fs)
        return ax
    
    axes = {
        'q1': titleize(gs[0,0], "Upper PV $q_1$"),
        'q2': titleize(gs[0,1], "Lower PV $q_2$"),
        'ke': titleize(gs[0,2], "Kinetic Energy $\\int u^2+v^2$"),
        'en': titleize(gs[0,3], "Enstrophy $\\int \\omega^2$"),
        'kespec': titleize(gs[1,2], "Kinetic Energy Spectrum"),
        'enspec': titleize(gs[1,3], "Enstrophy Spectrum"),
        'xfspec':  titleize(gs[1,:2], "Spectral Energy Transfers"),
    }

    for ax in ['q1','q2','ke','en']:
        axes[ax].set_xlabel("Longitude $x$", fontsize=14)
        axes[ax].set_ylabel("Latitude $y$", fontsize=14)
    
    def model_dataset():
        ds = m.to_dataset().isel(time=-1)
        ds['ke'] = ds.ufull**2 + ds.vfull**2
        ds['vorticity'] = -ds.ufull.differentiate(coord='y') + ds.vfull.differentiate(coord='x')
        ds['enstrophy'] = ds['vorticity']**2
        return ds
    
    m._step_forward()
    m._calc_derived_fields()
    
    anims = [
        AnimatedImage(axes['q1'], lambda: m.q[0]),
        AnimatedImage(axes['q2'], lambda: m.q[1]),
        AnimatedImage(axes['ke'], lambda: model_dataset().ke.sum(dim='lev'), min_vmin=0),
        AnimatedImage(axes['en'], lambda: model_dataset().enstrophy.sum(dim='lev'), min_vmin=0),
        AnimatedSpectrum(axes['kespec'], m, 'KEspec', fit_opts=fit_opts),
        AnimatedSpectrum(axes['enspec'], m, 'entspec', fit_opts=fit_opts),
        AnimatedSpectrum(axes['xfspec'], m, 'APEgenspec', logy=False),
        AnimatedSpectrum(axes['xfspec'], m, 'APEflux', logy=False),
        AnimatedSpectrum(axes['xfspec'], m, 'KEflux', logy=False),
        AnimatedSpectrum(axes['xfspec'], m, 'Diss.', logy=False),
    ]
    
    axes['xfspec'].legend(loc='upper left')
    
    def animate(i):
        for _ in range(steps_per_frame):
            m._step_forward()
        m._calc_derived_fields()
        res = []
        for anim in anims:
            res += anim.animate()
        ylim = max(anim.ylim for anim in anims if anim.ax == axes['xfspec'])
        axes['xfspec'].set_ylim(-ylim, ylim)
        if label is not None:
            fig.suptitle(f"{label}, t plus {(m.t-t0)/year:.2f} years", fontsize=20, y=suptitle_y, va='bottom')
        else:
            fig.suptitle(f"t plus {(m.t-t0)/year:.2f} years", fontsize=20, y=suptitle_y, va='bottom')
        return res
    
    gs.tight_layout(fig, rect=[0, 0.03, 1, 0.95])

    anim = animation.FuncAnimation(fig, animate, frames=n_frames, interval=50, blit=True)
    return anim

def imshow(arr, vmin=None, vmax=None, **kw):
    if isinstance(arr, xr.DataArray):
        arr = arr.data
    if vmin is None or vmax is None:
        if arr.min() >= 0:
            vmin = 0; vmax = np.percentile(arr.ravel(), 99)
        else:
            vmax = np.percentile(np.abs(arr).ravel(), 99); vmin = -vmax
    if vmin == 0:
        cmap = 'inferno'
    else:
        cmap = 'bwr'
    plt.imshow(arr, cmap=cmap, interpolation='none', vmin=vmin, vmax=vmax)
    for ticks in [plt.xticks, plt.yticks]:
        ticks([0,len(arr)//2, len(arr)], ['0', 'L/2', 'L'])
    plt.xlabel("Longitude $x$"); plt.ylabel("Latitude $y$")
    if 'title' in kw:
        plt.title(kw.pop('title'))
    cb = plt.colorbar(**kw)
    cb.ax.yaxis.set_offset_position('left')

class figure_grid():
    def next_subplot(self, title=None, **kwargs):
        if self.next_title is not None:
          plt.title(self.next_title)
        self.subplots += 1
        self.next_title = title
        return self.fig.add_subplot(self.rows, self.cols, self.subplots, **kwargs)

    def each_subplot(self):
        for _ in range(self.rows * self.cols):
            yield self.next_subplot()

    def title(self, title, **kwargs):
        self.fig.suptitle(title, **kwargs)

    @property
    def row(self):
        return (self.subplots-1) // self.cols

    @property
    def col(self):
        return (self.subplots-1) % self.cols

    def __init__(self, rows=None, cols=4, total=None, rowheight=3, rowwidth=None, after_each=lambda: None, filename=None):
        if rows is not None and cols is not None:
            self.rows = rows
            self.cols = cols
        elif rows is None and total is not None and cols is not None:
            self.cols = cols
            self.rows = int(np.ceil(total/cols))
        elif cols is None and total is not None and rows is not None:
            self.rows = rows
            self.cols = int(np.ceil(total/rows))

        if rowwidth is None:
            rowwidth = self.cols * 4

        self.fig = plt.figure(figsize=(rowwidth, rowheight*self.rows))
        self.subplots = 0
        self.next_title = None
        self.filename = filename

    def __enter__(self):
        return self

    def __exit__(self, _type, _value, _traceback):
        if self.next_title is not None:
          plt.title(self.next_title)
        if self.filename:
            try:
                plt.tight_layout()
                plt.savefig(self.filename, bbox_inches='tight')
            except:
                print("ERROR SAVING FIGURE")
            plt.close(self.fig)
        else:
            plt.tight_layout()
            plt.show()

    next = next_subplot

def kdeplot(data_, ax=None, lim=7.5, **kw):
    if ax is None: ax = plt.gca()
    data = np.array(data_).ravel()
    kde = gaussian_kde(data)
    lo, hi = np.percentile(data, [lim, 100-lim])
    diff = (hi-lo)
    lims = np.linspace(lo - diff*0.1, hi + diff*0.1, 200)
    ax.plot(lims, kde(lims), **kw)
    ax.set_yticks([])
    ax.set_ylabel('Density')

class PlotOptions():
    def __init__(self, datasets, markers=True):
        if markers:
            markers = list(Line2D.filled_markers)
        else:
            markers = [None]
        styles = ['-','--','-.',':']
        kws = [dict(ls=styles[i%len(styles)],
                    marker=markers[i%len(markers)],
                    markersize=12, markeredgecolor='white')
              for i in range(len(datasets))]
        self.kws = kws
        self.labels = [ds.label for ds in datasets]
        
    def __getitem__(self, k):
        if k in self.labels:
            k = self.labels.index(k)
        return self.kws[k]

def quantity_label(q):
    if q == 'q':
        return "Potential vorticity [$s^{-1}$]"
    elif q == 'u':
        return "x-velocity [$ms^{-1}$]"
    elif q == 'v':
        return "y-velocity [$ms^{-1}$]"
    elif q == 'ke':
        return 'KE density [$m^2s^{-2}$]'
    elif q == 'enstrophy':
        return 'Enstrophy [$m^2s^{-2}$]'
    else:
        return q

def quantity_title(q):
    return quantity_label(q).split(' [')[0]

def quantity_units(q):
    return '['+quantity_label(q).split(' [')[1]

def compare_simulations(*datasets, directory=None, new_fontsize=16, title_suffix='', show_quantities=True, show_timeseries=True, show_distributions=True, show_spectra=True, show_budgets=True):
    if directory is not None:
        os.system(f"mkdir -p {directory}")

    orig_fontsize = plt.rcParams['font.size']
    plt.rcParams.update({ 'font.size': new_fontsize })

    datasets = [pse.Dataset.wrap(d) for d in datasets]
    ds1 = datasets[0]

    for i, ds in enumerate(datasets):
        plot_kwargs = ds.attrs.get('plot_kwargs', {})
        if 'label' not in plot_kwargs:
            plot_kwargs['label'] = ds.attrs.get('label', f"Simulation {i+1}")
            ds = ds.assign_attrs(plot_kwargs=plot_kwargs)

    def filename_for(plot):
        if directory is None:
            return None
        else:
            return os.path.join(directory, f"{plot}.png")

    quantities = ['q','u','ke','enstrophy']

    layers = range(len(ds1.lev))
    
    if show_quantities:
        compare_quantities(datasets, filename=filename_for("quantity_final_values"), title_suffix=title_suffix)
        compare_quantities(datasets, filename=filename_for("quantity_time_averages"), title_suffix=title_suffix, time_average=True)
                        
    if show_timeseries:
        kws = PlotOptions(datasets, markers=False)
        with figure_grid(rows=1, cols=len(quantities), filename=filename_for("quantities_over_time"), rowwidth=24, rowheight=6) as g:
            g.title(f"Temporal evolution of quantities, averaged over space/run{title_suffix}")
            for quantity in quantities:
                g.next(title=quantity_title(quantity))
                for i,ds in enumerate(datasets):
                    time = ds.coords['time']
                    if time.dtype == np.dtype('<m8[ns]'):
                        time = (time.data / np.timedelta64(1, 'D')).astype(int)
                    y = ds[quantity].sum(dim='lev').mean(dim=['y','x','run']).data
                    yerr = ds[quantity].sum(dim='lev').std(dim=['y','x','run']).data
                    plt.plot(time, y, label=ds.label, lw=2, **kws[i])
                    plt.fill_between(time, y-yerr, y+yerr, alpha=0.1)
                    plt.xlabel("Model time [days]")
                    plt.ylabel(quantity_label(quantity))
                if g.col==0:
                    plt.legend(fontsize=14)

    if show_distributions:
        compare_distributions(datasets, filename=filename_for("quantity_distributions"), z='sum', title_suffix=title_suffix)

    if show_spectra:
        with figure_grid(rows=1, cols=2, rowwidth=24, rowheight=8, filename=filename_for("spectra")) as g:
            g.title(f"Spectral comparisons{title_suffix}")
            for key in ['KEspec', 'Ensspec']:
                g.next(title=ds1[key].attrs['long_name'])
                compare_spectra(datasets, key, z='sum')

    if show_budgets:
        compare_budgets(datasets, filename=filename_for("energy_budgets"), title_suffix=title_suffix)

    plt.rcParams.update({ 'font.size': orig_fontsize })
    
def compare_quantities(datasets, quantities=['q','u','ke','enstrophy'], time_average=False, filename=None, title_suffix=''):
    def imshow(x, vmin=None, vmax=None, cb=True, cb_label=''):
        plt.imshow(x, cmap=('inferno' if vmin == 0 else 'bwr'), vmin=vmin, vmax=vmax)
        plt.xticks([])
        plt.yticks([])
        if cb:
            cb = plt.colorbar(label=cb_label)
            cb.ax.yaxis.set_offset_position('left')
            
    layers = range(len(datasets[0].lev))
    
    if time_average:
        extract = lambda ds, q, z: ds[q].isel(lev=z, run=-1, time=slice(-len(ds.time)//2, None)).mean(dim='time').data
    else:
        extract = lambda ds, q, z: ds[q].isel(lev=z, run=-1, time=-1).data
    cols = len(datasets)*len(layers)
    with figure_grid(rows=len(quantities), cols=cols, filename=filename, rowwidth=cols*4.5) as g:
        if time_average:
            g.title(f"Time-averaged quantities{title_suffix}")
        else:
            g.title(f"Final values of quantities{title_suffix}")
        for quantity in quantities:
            for z in layers:
                vmax = 0
                vmin = 0
                for ds in datasets:
                    x = extract(ds, quantity, z).ravel()
                    vmax = max(vmax, np.percentile(x, 99))
                    vmin = min(vmin, np.percentile(x, 1))
                if vmin < 0:
                    vmax = max(vmax, -vmin) * 1.01
                    vmin = -vmax

                for ds in datasets:
                    g.next()
                    if g.row == 0: plt.title(ds.label)
                    if g.col == 0: plt.ylabel(f"{quantity_title(quantity)}, z={z}", rotation=0, ha='right', va='center', fontweight='bold')
                    if g.col == len(datasets): plt.ylabel(f"z={z}", rotation=0, ha='right', va='center', fontweight='bold')
                    imshow(extract(ds, quantity, z), vmin=vmin, vmax=vmax, cb=((g.col+1) % len(datasets) == 0), cb_label=quantity_units(quantity))
                
def compare_distributions(datasets, quantities=['q','u','ke','enstrophy'], z='sum', filename=None, title_suffix=''):
    kws = PlotOptions(datasets,markers=False)
    #layers = range(len(datasets[0].lev))
    with figure_grid(rows=1, cols=len(quantities), filename=filename, rowwidth=24, rowheight=6) as g:
        g.title(f"Final distributions of quantities{title_suffix}")
        for quantity in quantities:
            g.next(title=quantity_title(quantity))
            if z == 'sum':
                distributions = [ds[quantity].isel(time=-1).sum(dim='lev').data.ravel() for ds in datasets]
            else:
                distributions = [ds[quantity].isel(time=-1).isel(lev=z).data.ravel() for ds in datasets]
                
            for i, (ds, dist) in enumerate(zip(datasets, distributions)):
                kw = dict(kws[i])
                kw.update(ds.attrs.get('plot_kwargs', {}))
                kdeplot(dist, lw=2, **kw)
            if g.col==0:
                plt.legend(loc='best', fontsize=14)
                plt.ylabel("Probability Density")
            else:
                plt.ylabel("")
            plt.xlabel(quantity_label(quantity))
            plt.yscale('log')
    
def compare_budgets(datasets, filename=None, title_suffix=''):
    kws = PlotOptions(datasets)
    
    with figure_grid(1, cols=1, rowwidth=16, rowheight=8, filename=filename) as g:
        g.next()
        plt.title(f"Spectral energy budgets{title_suffix}", fontsize=18)
        plt.grid(alpha=0.25)

        colors = OrderedDict()

        for i, ds in enumerate(datasets):
            k, budget = ds.normalized_energy_budget
            plt.axvline(ds.twothirds_nyquist_frequency, color='gray', alpha=0.5, **kws[i])
            plt.axvline(1/ds.m.rd, color='pink', alpha=0.5, **kws[i])
            for key, val in budget.items():
                if key in colors:
                    plt.semilogx(k, val, color=colors[key], **kws[i])
                else:
                    colors[key] = plt.semilogx(k, val, **kws[i])[0]._color
        
        plt.legend(handles=[
            Patch(facecolor=color, label=key) for key, color in colors.items()
        ] + [
            Patch(facecolor='pink', label='Deformation frequency', alpha=0.5),
            Patch(facecolor='gray', label='65% Nyquist frequency', alpha=0.5),
        ] + [
            Line2D([0], [0], color='black', label=ds.label, **kw) for kw, ds in zip(kws, datasets)
        ], fontsize=18, loc='upper right')
        
        plt.xlabel("Radial wavenumber [$m^{-1}$]")
        plt.ylabel("Energy density tendency [$m^2 s^{-3}$]")

def compare_spectra(datasets, key='KEspec', z='sum', ax=None, loglog=True, leg=True, xlim=None, kmin=5e-5, kmax=1.5e-4, fontsize=16, legend_fontsize=16, plot_fits=True, **kw):
    if ax is None: ax = plt.gca()
        
    maxes = []

    for ds in datasets:
        k, q = ds.isotropic_spectrum(key, z=z)
        if q.min() < 0:
            loglog = False
            
    kws = PlotOptions(datasets)
    
    legend_handles = []

    for l, ds in enumerate(datasets):
        k, q = ds.isotropic_spectrum(key, z=z)
        
        if key in ['APEflux','KEflux','APEgenspec','Dissipation']:
            q = q / ds.m.M**2

        if loglog:
            plot_fn = ax.loglog
        else:
            plot_fn = ax.semilogx
            

        kwargs = dict(ds.attrs.get('plot_kwargs', {}))
        kwargs = dict(kwargs)
        if 'label' not in kwargs:
            kwargs['label'] = ds.label
        kwargs.update(kws[l])

        i = np.argmin(np.abs(np.log(k) - np.log(kmin)))
        j = np.argmin(np.abs(np.log(k) - np.log(kmax)))
        lr = linregress(np.log(k[i:j]), np.log(q[i:j]))

        if loglog:
            kwargs['label'] = kwargs.get('label', '') + " (${\propto}k^{"+f"{lr.slope:.2f}"+"}$)" 
            
        line = plot_fn(k,q,lw=3,**kwargs,zorder=10)[0]
        legend_handles.append(line)

        plt.axvline(ds.twothirds_nyquist_frequency, color='gray', **kws[l])
        plt.axvline(1/ds.m.rd, color='pink', **kws[l])

        if loglog and plot_fits:
            plot_fn(k[i-1:j+1], np.exp(np.log(k[i-1:j+1]) * lr.slope + lr.intercept)*1.2, color=line._color, ls=kws[l]['ls'], alpha=0.5)

        maxes.append(q.max())

    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        ax.set_xlim(k.min(), 2e-4)
    if loglog: ax.set_ylim(min(maxes)/1000, max(maxes)*2)
    if leg:
        legend_handles += [
            Patch(facecolor='pink', label='Deformation frequency', alpha=0.5),
            Patch(facecolor='gray', label='65% Nyquist frequency', alpha=0.5),
        ]
        plt.legend(handles=legend_handles, loc='best',fontsize=legend_fontsize).set_zorder(11)
    ax.grid(alpha=0.25)

    prefix = ''
    if z == 0:
        prefix = "Upper "
    if z == 1: 
        prefix = "Lower "
    if z == 'sum':
        prefix = "Barotropic "
    
    ylabel = key
        
    if key in ['KEspec']:
        ylabel = "KE spectrum [$m^{2} s^{-2}$]"
    if key in ['Ensspec','entspec']:
        ylabel = "Enstrophy spectrum [$s^{-2}$]"
    if key in ['APEflux','KEflux','APEgenspec','Dissipation']:
        units = "$m^{2} s^{-3}$"
        ylabel = f"{prefix}Normalized {key.replace('flux', ' flux').replace('genspec', ' generation')} [{units}]"
        
    ax.set_ylabel(prefix+ylabel, fontsize=fontsize)
    ax.set_xlabel("Radial wavenumber $k$ [$m^{-1}$]", fontsize=fontsize)
