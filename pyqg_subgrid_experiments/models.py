import os
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
import numpy.fft as npfft
import pyqg
import xarray as xr
from collections import OrderedDict
from sklearn.isotonic import IsotonicRegression
import pyqg_subgrid_experiments as pse

def minibatch(*arrays, batch_size=64, as_tensor=True):
    assert len(set([len(a) for a in arrays])) == 1
    order = np.arange(len(arrays[0]))
    np.random.shuffle(order)
    steps = int(np.ceil(len(arrays[0]) / batch_size))
    xform = torch.as_tensor if as_tensor else lambda x: x
    for step in range(steps):
        idx = order[step*batch_size:(step+1)*batch_size]
        yield tuple(xform(array[idx]) for array in arrays)

def train(net, inputs, targets, num_epochs=50, batch_size=64, learning_rate=0.001, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(num_epochs/2), int(num_epochs*3/4), int(num_epochs*7/8)], gamma=0.1)
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        for x, y in minibatch(inputs, targets, batch_size=batch_size):
            optimizer.zero_grad()
            yhat = net.forward(x.to(device))
            ytrue = y.to(device)
            loss = criterion(yhat, ytrue)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_steps += 1
        print(f"Loss after Epoch {epoch+1}: {epoch_loss/epoch_steps}")
        scheduler.step()

class ScaledModel(object):
    @property
    def is_zero_mean(self):
        return hasattr(self, 'zero_mean') and getattr(self, 'zero_mean')

    def set_zero_mean(self, zero_mean=True):
        self.zero_mean = zero_mean

    def set_scales(self, input_scale, output_scale):
        self.input_scale = input_scale
        self.output_scale = output_scale

    def set_inputs(self, inputs):
        self.inputs = inputs

    def set_targets(self, targets):
        self.targets = targets

    def extract_vars(self, m, features, spatially_flatten=False, dtype=np.float32):
        if not isinstance(m, pse.Dataset):
            m = pse.Dataset(m)

        arr = np.array([
            m.extract_feature(feat).isel(lev=z).data
            for feat, z in features
        ])

        arr = np.moveaxis(arr, 0, -3)

        arr = arr.reshape(-1, len(features), m.nx, m.nx)

        if spatially_flatten:
            arr = arr.reshape(-1, len(features), m.nx**2)

        return arr.astype(dtype)

    def extract_inputs(self, m):
        return self.extract_vars(m, self.inputs)

    def extract_targets(self, m):
        return self.extract_vars(m, self.targets)

    def predict(self, inputs, device=None):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.to(device)
        preds = []
        for x, in minibatch(self.input_scale.transform(self.extract_inputs(inputs))):
            x = x.to(device)
            with torch.no_grad():
                preds.append(self.forward(x).cpu().numpy())
        preds = self.output_scale.inverse_transform(np.vstack(preds))
        if isinstance(inputs, pyqg.Model):
            return preds.astype(inputs.q.dtype)
        else:
            return preds

    def mse(self, inputs, targets, **kw):
        y_true = targets.reshape(-1, np.prod(targets.shape[1:]))
        y_pred = self.predict(inputs).reshape(-1, np.prod(targets.shape[1:]))
        return np.mean(np.sum((y_pred - y_true)**2, axis=1))

    def fit(self, inputs, targets, rescale=False, **kw):
        if rescale or not hasattr(self, 'input_scale') or self.input_scale is None:
            self.input_scale = ChannelwiseScaler(inputs)
        if rescale or not hasattr(self, 'output_scale') or self.output_scale is None:
            self.output_scale = ChannelwiseScaler(targets, zero_mean=self.is_zero_mean)
        train(self,
              self.input_scale.transform(inputs),
              self.output_scale.transform(targets),
              **kw)

    def save(self, path):
        os.system(f"mkdir -p {path}")
        torch.save(self.state_dict(), f"{path}/weights.pt")
        with open(f"{path}/input_scale.pkl", 'wb') as f:
            pickle.dump(self.input_scale, f)
        with open(f"{path}/output_scale.pkl", 'wb') as f:
            pickle.dump(self.output_scale, f)
        with open(f"{path}/inputs.pkl", 'wb') as f:
            pickle.dump(self.inputs, f)
        with open(f"{path}/targets.pkl", 'wb') as f:
            pickle.dump(self.targets, f)
        if self.is_zero_mean:
            open(f"{path}/zero_mean", 'a').close()

    @classmethod
    def load(cls, path):
        with open(f"{path}/inputs.pkl", 'rb') as f:
            inputs = pickle.load(f)
        with open(f"{path}/targets.pkl", 'rb') as f:
            targets = pickle.load(f)
        model = cls(inputs, targets)
        model.load_state_dict(torch.load(f"{path}/weights.pt"))
        with open(f"{path}/input_scale.pkl", 'rb') as f:
            model.input_scale = pickle.load(f)
        with open(f"{path}/output_scale.pkl", 'rb') as f:
            model.output_scale = pickle.load(f)
        if os.path.exists(f"{path}/zero_mean"):
            model.set_zero_mean()
        return model

class BasicScaler(object):
    def __init__(self, mu=0, sd=1):
        self.mu = mu
        self.sd = sd
        
    def transform(self, x):
        return (x - self.mu) / self.sd
    
    def inverse_transform(self, z):
        return z * self.sd + self.mu

class ChannelwiseScaler(BasicScaler):
    def __init__(self, x, zero_mean=False):
        assert len(x.shape) == 4
        if zero_mean:
            mu = 0
        else:
            mu = np.array([x[:,i].mean()
                for i in range(x.shape[1])])[np.newaxis,:,np.newaxis,np.newaxis]
        sd = np.array([x[:,i].std()
            for i in range(x.shape[1])])[np.newaxis,:,np.newaxis,np.newaxis]
        super().__init__(mu, sd)

class FullyCNN(nn.Sequential, ScaledModel):
    def __init__(self, inputs, targets, padding='same', batch_norm=True, zero_mean=False):
        if padding is None:
            padding_5 = 0
            padding_3 = 0
        elif padding == 'same':
            padding_5 = 2
            padding_3 = 1
        else:
            raise ValueError('Unknow value for padding parameter.')
        self.inputs = inputs
        self.targets = targets
        n_in = len(inputs)
        n_out = len(targets)
        self.n_in = n_in
        self.batch_norm = batch_norm
        block1 = self._make_subblock(nn.Conv2d(n_in, 128, 5, padding=padding_5))
        block2 = self._make_subblock(nn.Conv2d(128, 64, 5, padding=padding_5))
        block3 = self._make_subblock(nn.Conv2d(64, 32, 3, padding=padding_3))
        block4 = self._make_subblock(nn.Conv2d(32, 32, 3, padding=padding_3))
        block5 = self._make_subblock(nn.Conv2d(32, 32, 3, padding=padding_3))
        block6 = self._make_subblock(nn.Conv2d(32, 32, 3, padding=padding_3))
        block7 = self._make_subblock(nn.Conv2d(32, 32, 3, padding=padding_3))
        conv8 = nn.Conv2d(32, n_out, 3, padding=padding_3)
        super().__init__(*block1, *block2, *block3, *block4, *block5,
                            *block6, *block7, conv8)
        self.set_zero_mean(zero_mean)

    def forward(self, x):
        r = super().forward(x)
        if self.is_zero_mean:
            return r - r.mean(dim=(1,2,3), keepdim=True)
        else:
            return r
        
    def _make_subblock(self, conv):
        subbloc = [conv, nn.ReLU()]
        if self.batch_norm:
            subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc
