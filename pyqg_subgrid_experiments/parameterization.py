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
            for f in glob.glob(os.path.join(directory, "models/*.pt"))
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


    def __call__(self, m):
        return self.predict(m)

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
            model.save(os.path.join(directory, f"models/{z}.pt"))

        # Return the trained parameterization
        return cls(directory, models=models)

    def test_offline(self, dataset):
        preds = self.predict(dataset)

        results = dataset[self.targets]

        import pdb; pdb.set_trace()

    def run_online(self, **kw):
        params = dict(kw)
        params[self.parameterization_type] = self
        return generate_dataset(**params)


