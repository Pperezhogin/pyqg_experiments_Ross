import os
import glob
import numpy as np
import pyqg_subgrid_experiments as pse
from pyqg_subgrid_experiments.models import FullyCNN
from pyqg_subgrid_experiments.simulate import generate_dataset

class CNNParameterization(object):
    def __init__(self, directory, models=None, model_class=FullyCNN):
        self.directory = directory
        self.models = models if models is not None else [
            model_class.load(f)
            for f in glob.glob(os.path.join(directory, "models/*"))
        ]

    @property
    def targets(self):
        targets = set()
        for model in self.models:
            for target, z in model.targets:
                targets.add(target)
        return list(sorted(list(targets)))

    @property
    def parameterization_type(self):
        if any(q in self.targets[0] for q in ['q_forcing', 'q_subgrid']):
            return 'q_parameterization'
        else:
            return 'uv_parameterization'

    def predict(self, m):
        preds = {}

        for model in self.models:
            pred = model.predict(m)
            assert len(pred.shape) == 4
            assert len(model.targets) == pred.shape[1]
            for channel in range(pred.shape[1]):
                target, z = model.targets[channel]
                if target not in preds:
                    preds[target] = np.zeros_like(m.q)
                m_indices = [slice(None) for _ in m.q.shape]
                m_indices[-3] = slice(z,z+1)
                m_shape = [s for s in m.q.shape]
                m_shape[-3] = 1
                preds[target][m_indices] = pred[:,channel,:,:].reshape(m_shape)

        return preds

    def __call__(self, m):
        preds = self.predict(m)
        keys = list(sorted(preds.keys()))
        assert keys == self.targets
        if len(keys) == 1:
            return preds[keys[0]]
        elif keys == ['u_forcing_advection', 'v_forcing_advection']:
            return tuple(preds[k] for k in keys)
        elif keys == ['uq_subgrid_flux', 'vq_subgrid_flux']:
            ds = m if isinstance(m, pse.Dataset) else pse.Dataset(m)
            return ds.ddx(preds['uq_subgrid_flux']) + ds.ddy(preds['vq_subgrid_flux'])
        elif 'uu_subgrid_flux' in keys and len(keys) == 3:
            ds = m if isinstance(m, pse.Dataset) else pse.Dataset(m)
            return (ds.ddx(preds['uu_subgrid_flux']) + ds.ddy(preds['uv_subgrid_flux']),
                    ds.ddx(preds['uv_subgrid_flux']) + ds.ddy(preds['vv_subgrid_flux']))
        else:
            raise ValueError(f"unknown targetset {keys}")

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
        for z, model in enumerate(models):
            X = model.extract_inputs(dataset)
            Y = model.extract_targets(dataset)
            model.fit(X, Y, num_epochs=num_epochs, **kw)
            model.save(os.path.join(directory, f"models/{z}"))

        # Return the trained parameterization
        return cls(directory, models=models)

    def run_online(self, **kw):
        params = dict(kw)
        params[self.parameterization_type] = self
        return generate_dataset(**params)

    def test_offline(self, dataset):
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

            test[f"{key}_spatial_correlation"] = (
                true_pred_cov.sum(dim=time)
                / (true_var.sum(dim=time) * pred_var.sum(dim=time))**0.5
            )
            test[f"{key}_temporal_correlation"] = (
                true_pred_cov.sum(dim=space)
                / (true_var.sum(dim=space) * pred_var.sum(dim=space))**0.5
            )
            test[f"{key}_correlation"] = (
                true_pred_cov.sum(dim=both)
                / (true_var.sum(dim=both) * pred_var.sum(dim=both))**0.5
            )

        for metric in ['correlation', 'mse', 'skill']:
            test[metric] = sum(
                test[f"{key}_{metric}"] for key in self.targets
            ) / len(self.targets)

        return test
