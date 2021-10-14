import numpy as np
import xarray as xr
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler

class SymbolicRegressionParameterization(object):
    def __init__(self, linear_model, feature_names, x_scaler, y_sd):
        self.linear_model = linear_model
        self.x_scaler = x_scaler
        self.y_sd = y_sd
        self.feature_names = feature_names
        
    def predict(self, m):
        X = self.input_data(m)
        yhat = self.linear_model.predict(X) * self.y_sd
        return yhat.reshape(64, 64)
        
    def input_data(self, m):
        ds = m.to_dataset()
        import numpy.fft as npfft
        from pyqg.xarray_output import spatial_dims
        ds['dqdt'] = xr.DataArray(
            npfft.irfftn(m.dqhdt,axes=(-2,-1))[np.newaxis],
            coords=[ds.coords[d] for d in spatial_dims]
        )
        
        for f_ in self.feature_names:
            for f in f_.split('_times_'):
                name = f[:-1]
                if name not in ds and '_' in name:
                    orig, diffs = name.split('_')
                    if len(diffs)==1:
                        ds[name] = ds[orig].differentiate(diffs)
                    elif len(diffs)==2:
                        intermed = f"{orig}_{diffs[0]}"
                        if intermed not in ds:
                            ds[intermed] = ds[orig].differentiate(diffs[0])
                        ds[name] = ds[intermed].differentiate(diffs[1])
                    else:
                        assert(False)
                        
        X = []
                        
        for f in self.feature_names:
            if '_times_' in f:
                fi, fj = f.split('_times_')
                fi_name = fi[:-1]
                fj_name = fj[:-1]
                fi_lev = int(fi[-1])-1
                fj_lev = int(fj[-1])-1
                X.append(ds[fi_name].isel(lev=fi_lev) * ds[fj_name].isel(lev=fj_lev))
            else:
                X.append(ds[f[:-1]].isel(lev=int(f[-1])-1))
                
            
        X = np.array([x.data.ravel() for x in X]).T
        
        return self.x_scaler.transform(X)
