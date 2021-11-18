import os
import glob
import numpy as np
from pyqg_subgrid_experiments.models import FullyCNN
from pyqg_subgrid_experiments.simulate import initialize_pyqg_model

class Parameterization(object):
    def __call__(self, m):
        pass

    def initialize_pyqg_model(self, **params):
        pass

class CNNParameterization(object):
    def __init__(self, directory, models=None, model_class=FullyCNN):
        self.directory = directory
        self.models = models if models is not None else [
            model_class.load(f)
            for f in glob.glob(os.path.join(directory, "models/*"))
        ]

    def initialize_pyqg_model(self, **params):
        pass

    def predict(self, m):
        preds = []
        for model in self.models:
            pred = model.predict(m)
            assert len(pred.shape) == 4
            assert len(model.targets) == pred.shape[1]
            for channel in range(pred.shape[1]):
                z = model.targets[channel][1]
                assert z == len(preds)
                preds.append(pred[:,channel,:,:])
        preds = np.swapaxes(np.array(preds), 0, 1)
        if len(preds) == 1: preds = preds[0]
        return preds

    def __call__(self, m):
        return self.predict(m)

    @classmethod
    def train_on(cls, dataset, directory,
            inputs=['q','u','v'],
            targets=['q_forcing_advection'],
            layerwise_inputs=None,
            layerwise_targets=None,
            num_epochs=50,
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
                    zero_mean=True
                ) for z in layers
            ]
        elif layerwise_targets:
            # Each layer has its own model that uses every layer's data to
            # predict layer-specific targets
            models = [
                model_class(
                    [(feat, zi) for feat in inputs for zi in layers],
                    [(feat, z) for feat in targets],
                    zero_mean=True
                ) for z in layers
            ]
        else:
            # There is a single model that predicts all layers' targets from
            # all layers' data
            models = [
                model_class(
                    [(feat, z) for feat in inputs for z in layers],
                    [(feat, z) for feat in targets for z in layers],
                    zero_mean=True
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

    def test_on(self, dataset):
        preds = self.predict(dataset)

        import pdb; pdb.set_trace()



