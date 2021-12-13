import os
import sys
import glob
import json
import numpy as np
import xarray as xr
import pyqg
import pyqg_subgrid_experiments as pse
from pyqg_subgrid_experiments.models import FullyCNN
from pyqg_subgrid_experiments.simulate import generate_dataset, time_until_uncorrelated

class Parameterization(object):
    @property
    def targets(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError
        
    @property
    def nx(self):
        return 64

    @property
    def parameterization_type(self):
        if any(q in self.targets[0] for q in ['q_forcing', 'q_subgrid']):
            return 'q_parameterization'
        else:
            return 'uv_parameterization'

    def __call__(self, m):
        def arr(x):
            if isinstance(x, xr.DataArray): x = x.data
            return x.astype(m.q.dtype)

        preds = self.predict(m)
        keys = list(sorted(preds.keys()))
        assert keys == self.targets
        if len(keys) == 1:
            return arr(preds[keys[0]])
        elif keys == ['u_forcing_advection', 'v_forcing_advection']:
            return tuple(arr(preds[k]) for k in keys)
        elif keys == ['uq_subgrid_flux', 'vq_subgrid_flux']:
            ds = pse.Dataset.wrap(m)
            return arr(ds.ddx(preds['uq_subgrid_flux']) + ds.ddy(preds['vq_subgrid_flux']))
        elif 'uu_subgrid_flux' in keys and len(keys) == 3:
            ds = m if isinstance(m, pse.Dataset) else pse.Dataset(m)
            return (arr(ds.ddx(preds['uu_subgrid_flux']) + ds.ddy(preds['uv_subgrid_flux'])),
                    arr(ds.ddx(preds['uv_subgrid_flux']) + ds.ddy(preds['vv_subgrid_flux'])))
        else:
            raise ValueError(f"unknown targetset {keys}")

    def test_on(self, dataset, artifact_dir=None, **kw):
        if artifact_dir is not None:
            os.system(f"mkdir -p {artifact_dir}")
            offline_path = os.path.join(artifact_dir, "offline_metrics.nc")
            online_dir = os.path.join(artifact_dir, "online_simulations")
        else:
            offline_path = None
            online_dir = None

        # Compute offline metrics
        preds = self.test_offline(dataset, offline_path=offline_path)

        # Run online simulations
        sims, distances = self.test_online(dataset, online_dir=online_dir, **kw)

        return preds, sims, distances

    def run_online(self, **kw):
        params = dict(kw)
        params[self.parameterization_type] = self
        params['nx'] = self.nx
        return generate_dataset(**params)
    
    def test_online(self, dataset, n_simulations=5, online_dir=None, **kw):
        if online_dir is not None:
            os.system(f"mkdir -p {online_dir}")
            distance_path = os.path.join(online_dir, "distances.json")
        else:
            distance_path = None
        
        # Run online simulations
        sim_params = dict(dataset.pyqg_params)
        sim_params.update(kw)
        sim_params['nx'] = self.nx
        sims = []
        for i in range(n_simulations):
            if online_dir is not None and os.path.exists(os.path.join(online_dir, f"{i}.nc")):
                sim = xr.open_dataset(os.path.join(online_dir, f"{i}.nc"))
            else:
                sim = self.run_online(**sim_params)
                if online_dir is not None:
                    sim.to_netcdf(os.path.join(online_dir, f"{i}.nc"))
            sims.append(sim)
        sims = xr.concat(sims, dim='run')

        # Compute online metrics
        if distance_path is not None and os.path.exists(distance_path):
            with open(distance_path, 'r') as f:
                distances = json.loads(f.read())
        else:
            distances = dataset.distributional_distances(sims)
            if distance_path is not None:
                with open(distance_path, 'w') as f:
                    f.write(json.dumps(distances))

        return sims, distances

    def test_offline(self, dataset, offline_path=None):
        if offline_path is not None and os.path.exists(offline_path):
            return xr.open_dataset(offline_path)
        
        test = dataset[self.targets]
        
        for key, val in self.predict(dataset).items():
            truth = test[key]
            test[f"{key}_predictions"] = truth*0 + val
            preds = test[f"{key}_predictions"]
            error = (truth - preds)**2

            true_centered = (truth - truth.mean())
            pred_centered = (preds - preds.mean())
            true_var = true_centered**2
            pred_var = pred_centered**2
            true_pred_cov = true_centered * pred_centered

            def dims_except(*dims):
                return [d for d in test[key].dims if d not in dims]
            
            time = dims_except('x','y','lev')
            space = dims_except('time','lev')
            both = dims_except('lev')

            test[f"{key}_spatial_mse"] = error.mean(dim=time)
            test[f"{key}_temporal_mse"] = error.mean(dim=space)
            test[f"{key}_mse"] = error.mean(dim=both)

            test[f"{key}_spatial_skill"] = 1 - test[f"{key}_spatial_mse"] / true_var.mean(dim=time)
            test[f"{key}_temporal_skill"] = 1 - test[f"{key}_temporal_mse"] / true_var.mean(dim=space)
            test[f"{key}_skill"] = 1 - test[f"{key}_mse"] / true_var.mean(dim=both)

            test[f"{key}_spatial_correlation"] = xr.corr(truth, preds, dim=time)
            test[f"{key}_temporal_correlation"] = xr.corr(truth, preds, dim=space)
            test[f"{key}_correlation"] = xr.corr(truth, preds, dim=both)

        for metric in ['correlation', 'mse', 'skill']:
            test[metric] = sum(
                test[f"{key}_{metric}"] for key in self.targets
            ) / len(self.targets)
            
        if offline_path is not None:
            os.system(f"mkdir -p {os.path.dirname(offline_path)}")
            test.to_netcdf(offline_path)

        return test
    
    def decorrelation_timescale(self, dataset, **kw):
        ds = pse.Dataset.wrap(dataset)
        params1 = dict(ds.pyqg_params)
        
        params2 = dict(params1)
        params2[self.parameterization_type] = self
        params2['nx'] = self.nx
        params2['ny'] = self.nx
        
        m1 = pyqg.QGModel(**params1)
        m1.set_q1q2(*ds.final_q)
        m1._invert()
        
        m2 = pyqg.QGModel(**params2)
        
        return time_until_uncorrelated(m1, m2, **kw)

class ZB2020Parameterization(Parameterization):
    def __init__(self, factor=-46761284, factor_mult=1.0):
        self.factor = factor * factor_mult

    @property
    def targets(self):
        return ['u_forcing_advection', 'v_forcing_advection']

    def predict(self, m):
        if isinstance(m, pyqg.QGModel):
            ik = 1j * m.k
            il = 1j * m.l
            
            # Compute relative velocity derivatives in spectral space
            uh = m.fft(m.u)
            vh = m.fft(m.v)
            vx = m.ifft(vh * ik)
            vy = m.ifft(vh * il)
            uy = m.ifft(uh * il)
            ux = m.ifft(uh * ik)
            
            # Compute ZB2020 basis functions
            rel_vort = vx - uy
            shearing = vx + uy
            stretching = ux - vy
            
            # Combine them in real space and take their FFT
            rv_stretch = m.fft(rel_vort * stretching)
            rv_shear = m.fft(rel_vort * shearing)
            sum_sqs = m.fft(rel_vort**2 + shearing**2 + stretching**2) / 2.0
            
            # Take spectral-space derivatives and multiply by the scaling factor
            Su = self.factor * m.ifft(ik*(sum_sqs - rv_shear) + il*rv_stretch)
            Sv = self.factor * m.ifft(il*(sum_sqs + rv_shear) + ik*rv_stretch)
        else:
            Su, Sv = pse.Dataset.wrap(m).zb2020_parameterization
            Su = Su.data * self.factor
            Sv = Sv.data * self.factor
        return dict(u_forcing_advection=Su, v_forcing_advection=Sv)

class AG7531Parameterization(Parameterization):
    def __init__(self, factor=1.0):
        agpath = '/scratch/zanna/code/ag7531'
        mpath = '/scratch/zanna/data/pyqg/models/ag7531/1/dc74cea68a7f4c7e98f9228649a97135'
        if not os.path.exists(agpath):
            raise ValueError("This method can currently only be run on NYU HPC")
        if not os.path.exists(mpath):
            raise ValueError(f"This method expects a trained model file at {mpath}")
        sys.path.append(agpath)
        sys.path.append(f"{agpath}/pyqgparamexperiments")
        import torch
        from subgrid.models.utils import load_model_cls
        from subgrid.models.transforms import SoftPlusTransform
        import pyqgparamexperiments.parameterization
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model_cls = load_model_cls('subgrid.models.models1', 'FullyCNN')
        net = model_cls(2, 4)
        net.final_transformation = SoftPlusTransform()
        net.final_transformation.indices = [1,3]
        net.load_state_dict(torch.load(f"{mpath}/artifacts/models/trained_model.pth"),)
        net.to(device)
        self.param = pyqgparamexperiments.parameterization.Parameterization(net, device)
        self.factor = factor
        
    @property
    def targets(self):
        return ['u_forcing_advection', 'v_forcing_advection']
    
    def predict(self, m):
        factor = self.factor
        du, dv = self.param(m.ufull, m.vfull, m.t)
        return dict(u_forcing_advection=du*factor, v_forcing_advection=dv*factor)

class CNNParameterization(Parameterization):
    def __init__(self, directory, models=None, model_class=FullyCNN):
        self.directory = directory
        self.models = models if models is not None else [
            model_class.load(f)
            for f in sorted(glob.glob(os.path.join(directory, "models/*")))
        ]

    @property
    def targets(self):
        targets = set()
        for model in self.models:
            for target, z in model.targets:
                targets.add(target)
        return list(sorted(list(targets)))

    def predict(self, m):
        preds = {}

        for model in self.models:
            pred = model.predict(m)
            assert len(pred.shape) == len(m.q.shape)
            for channel in range(pred.shape[-3]):
                target, z = model.targets[channel]
                if target not in preds:
                    preds[target] = np.zeros_like(m.q)
                out_indices = [slice(None) for _ in m.q.shape]
                out_indices[-3] = slice(z,z+1)
                in_indices = [slice(None) for _ in m.q.shape]
                in_indices[-3] = slice(channel,channel+1)
                preds[target][tuple(out_indices)] = pred[tuple(in_indices)]

        return preds

    @classmethod
    def train_on(cls, dataset, directory,
            inputs=['q','u','v'],
            targets=['q_forcing_advection'],
            layerwise_inputs=None,
            layerwise_targets=None,
            num_epochs=50,
            zero_mean=True,
            model_class=FullyCNN, **kw):

        layers = range(len(dataset.lev))

        # Initialize models based on arguments
        if layerwise_targets and layerwise_inputs:
            # Each layer has its own model that uses that layer's data to
            # predict layer-specific targets
            models = [
                model_class(
                    [(feat, z) for feat in inputs],
                    [(feat, z) for feat in targets],
                    zero_mean=zero_mean
                ) for z in layers
            ]
        elif layerwise_targets:
            # Each layer has its own model that uses every layer's data to
            # predict layer-specific targets
            models = [
                model_class(
                    [(feat, zi) for feat in inputs for zi in layers],
                    [(feat, z) for feat in targets],
                    zero_mean=zero_mean

                ) for z in layers
            ]
        else:
            # There is a single model that predicts all layers' targets from
            # all layers' data
            models = [
                model_class(
                    [(feat, z) for feat in inputs for z in layers],
                    [(feat, z) for feat in targets for z in layers],
                    zero_mean=zero_mean
                )
            ]

        # Train models on dataset and save them
        models2 = []
        for z, model in enumerate(models):
            model_dir = os.path.join(directory, f"models/{z}")
            if os.path.exists(model_dir):
                models2.append(model_class.load(model_dir))
            else:
                X = model.extract_inputs(dataset)
                Y = model.extract_targets(dataset)
                model.fit(X, Y, num_epochs=num_epochs, **kw)
                model.save(os.path.join(directory, f"models/{z}"))
                models2.append(model)

        # Return the trained parameterization
        return cls(directory, models=models2)
