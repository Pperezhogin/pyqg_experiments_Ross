import os
import matplotlib
if 'USE_AGG' in os.environ:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation
import xarray as xr
import numpy as np
from scipy.stats import linregress
from pyqg.errors import DiagnosticNotFilledError
from pyqg.diagnostic_tools import calc_ispec
from pyqg.particles import GriddedLagrangianParticleArray2D

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

    def __init__(self, ax, m, spec, logy=True):
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
        self.mean_line = AnimatedLine(ax, get_mean, lw=3, logy=logy, show_best_fit=logy, color=self.curr_line.color, label=spec)
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
    def __init__(self, ax, func, logx=True, logy=True, show_best_fit=False, **kw):
        super().__init__(ax, func)
        x, y = self.x
        self.line = ax.plot(x, y, **kw)[0]
        self.best_fit = None
        self.fit_text = None
        self.show_best_fit = show_best_fit
        self.logx = logx
        self.logy = logy

        if logx:
            ax.set_xscale('log')
        if logy:
            ax.set_yscale('log')

        if self.show_best_fit:
            self.loglog_fit(x, y)

    def set_alpha(self, alpha):
        self.line.set_alpha(alpha)
        if self.best_fit is not None:
            self.best_fit.set_alpha(alpha)
            self.fit_text.set_alpha(alpha)

    @property
    def color(self):
        return self.line._color

    def loglog_fit(self, x, y, fudge=1.5):
        i = np.argmax(y) + 4
        j = np.argmin(np.abs(np.log(x)-np.log(x[i]*5)))
        line_x = x[i:]
        if y.min() <= 0:
            line_y = y
            text_t = ""
        else:
            lr = linregress(np.log(x[i:j]), np.log(y[i:j]))
            self.ax.set_xlim(x[i]/10,x[j]*10)
            self.ax.set_ylim(y[j]/10,y[i]*10)
            line_y = np.exp(np.log(x[i:]) * lr.slope + lr.intercept)*fudge
            text_t = "$\propto k^{"+f"{lr.slope:.2f}"+"}$"
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
            self.loglog_fit(x, y)
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
    def __init__(self, ax, func, min_vmin=-float('inf'), max_vmax=float('inf')):
        super().__init__(ax, func)
        x = self.x
        ax.set_xticks([])
        ax.set_yticks([])
        self.max_vmax = max_vmax
        self.vmax = min(np.percentile(np.abs(x), 99)*1.01, self.max_vmax)
        if min_vmin == 0:
            self.vmin = 0
            self.cmap = 'inferno'
        else:
            self.vmin = -self.vmax
            self.cmap = 'bwr'
        self.im = ax.imshow(x, cmap=self.cmap, vmin=self.vmin, vmax=self.vmax)
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

def animate_simulation(m, n_frames=100, steps_per_frame=100, label=None, suptitle_y=0.95, fs=16):
    #mp = ModelWithParticles(m)
    b = 4
    year = 24*60*60*360.
    fig = plt.figure(figsize=(4.5*b, 2*b))
    gs = fig.add_gridspec(2, 4)
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
        AnimatedSpectrum(axes['kespec'], m, 'KEspec'),
        AnimatedSpectrum(axes['enspec'], m, 'entspec'),
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

    def title(self, title, fontsize=16, y=1.0, **kwargs):
        self.fig.suptitle(title, y=y, fontsize=fontsize, va='bottom', **kwargs)

    def __init__(self, rows, cols, rowheight=3, rowwidth=12, after_each=lambda: None, filename=None):
        self.rows = rows
        self.cols = cols
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
