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
import pyqg_subgrid_experiments as pse
import time

def minibatch(*arrays, batch_size=64, as_tensor=True, shuffle=True):
    assert len(set([len(a) for a in arrays])) == 1
    order = np.arange(len(arrays[0]))
    if shuffle:
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

class custom_loss():
    def __init__(self, loss_type, det_ch):
        self.loss_type = loss_type
        self.det_ch = det_ch

        if loss_type in ('std', 'var'):
            self.criterion = nn.GaussianNLLLoss()
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss()
        else:
            print('error: loss is not chosen')

    def loss(self, yhat, ytrue, regularization = 0):
        if self.loss_type == 'std':
            mean_tensor = yhat[:,:self.det_ch,:,:]
            var_tensor = torch.square(yhat[:,self.det_ch:,:,:])
            
            r = self.criterion(mean_tensor, ytrue, var_tensor) + \
                regularization * \
                torch.square(torch.square(mean_tensor-ytrue).mean()-var_tensor.mean())
            return r
        elif self.loss_type == 'var':
            mean_tensor = yhat[:,:self.det_ch,:,:]
            var_tensor = yhat[:,self.det_ch:,:,:]
            r = self.criterion(mean_tensor, ytrue, var_tensor) + \
                regularization * \
                torch.square(torch.square(mean_tensor-ytrue).mean()-var_tensor.mean())
            return r
        elif self.loss_type == 'mse':
            mean_tensor = yhat[:,:self.det_ch,:,:]
            return self.criterion(mean_tensor, ytrue)

def train_probabilistic(net, inputs, targets, num_epochs=50, batch_size=64, 
    learning_rate=0.001, device=None, inputs_test = None, targets_test = None, regularization = 0.,
    cosine_annealing = False):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)

    net.inference_stochastic = True
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    if cosine_annealing:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 8, T_mult=2, last_epoch=-1)
    else:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(num_epochs/2), int(num_epochs*3/4), int(num_epochs*7/8)], gamma=0.1)
    
    criterion = custom_loss(net.channel_type, net[-1].out_channels//2)

    net.loss_history = {'train_gauss': [], 'train_mse': [], 'train_noise': [],
                        'test_gauss': [], 'test_mse': [], 'test_noise': []}

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        t = time.time()
        for x, y in minibatch(inputs, targets, batch_size=batch_size):
            optimizer.zero_grad()
            yhat = net.forward(x.to(device))
            ytrue = y.to(device)

            # last argument is the variance
            loss = criterion.loss(yhat, ytrue, regularization)

            loss.backward()
            optimizer.step()
            weight = yhat.shape[0]
            epoch_loss += weight * loss.item()
            epoch_steps += weight
        
        test_loss(net, inputs, targets, inputs_test, targets_test)
        ETA = (time.time() - t)/60*(num_epochs-epoch-1)
        print(f"Gauss Loss after Epoch {epoch+1}: {epoch_loss/epoch_steps}, remaining min:{ETA}")
        scheduler.step()

def test_loss(net, inputs, targets, inputs_test, targets_test, batch_size=64):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

    net.eval()
    net.inference_stochastic = True

    mse_loss   = custom_loss('mse', net[-1].out_channels//2)
    gauss_loss = custom_loss(net.channel_type, net[-1].out_channels//2)

    epoch_gauss = 0.
    epoch_mse = 0.
    epoch_noise = 0.
    epoch_steps = 0
    for x,y in minibatch(inputs, targets, batch_size=batch_size, shuffle=False):
        with torch.no_grad():
            yhat  = net.forward(x.to(device))
        ytrue = y.to(device)

        noise = net.generate_noise(yhat)     

        weight = yhat.shape[0]
        epoch_gauss += weight * gauss_loss.loss(yhat, ytrue).item()
        epoch_mse   += weight * mse_loss.loss(yhat, ytrue).item()
        epoch_noise += weight * noise.var().item()
        epoch_steps += weight

    net.loss_history['train_gauss'].append(epoch_gauss/epoch_steps)
    net.loss_history['train_mse'].append(epoch_mse/epoch_steps)
    net.loss_history['train_noise'].append(epoch_noise/epoch_steps)

    epoch_gauss = 0.
    epoch_mse = 0.
    epoch_noise = 0.
    epoch_steps = 0
    for x,y in minibatch(inputs_test, targets_test, batch_size=batch_size, shuffle=False):
        with torch.no_grad():
            yhat  = net.forward(x.to(device))
        ytrue = y.to(device)

        noise = net.generate_noise(yhat)     

        weight = yhat.shape[0]
        epoch_gauss += weight * gauss_loss.loss(yhat, ytrue).item()
        epoch_mse   += weight * mse_loss.loss(yhat, ytrue).item()
        epoch_noise += weight * noise.var().item()
        epoch_steps += weight

    net.loss_history['test_gauss'].append(epoch_gauss/epoch_steps)
    net.loss_history['test_mse'].append(epoch_mse/epoch_steps)
    net.loss_history['test_noise'].append(epoch_noise/epoch_steps)

    net.train()

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

        arr = np.stack([
            np.array(m.extract_feature(feat).isel(lev=z).data)
            for feat, z in features
        ], axis=-3)

        #arr = np.moveaxis(arr, 0, -3)

        arr = arr.reshape((-1, len(features), m.nx, m.nx))

        if spatially_flatten:
            arr = arr.reshape((-1, len(features), m.nx**2))

        return arr.astype(dtype)

    def extract_inputs(self, m):
        return self.extract_vars(m, self.inputs)

    def extract_targets(self, m):
        return self.extract_vars(m, self.targets)

    def predict(self, inputs, device=None):
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.to(device)

        print('Current mode:', self.training)
        self.eval()
        print('Mode after eval():', self.training)
        self.inference_stochastic = False
        
        X = self.input_scale.transform(self.extract_inputs(inputs))

        preds = []
        for x, in minibatch(X, shuffle=False):
            x = x.to(device)
            with torch.no_grad():
                preds.append(self.forward(x).cpu().numpy())

        preds = self.output_scale.inverse_transform(np.vstack(preds))

        s = list(inputs.q.shape)
        preds = np.stack([
            preds[:,i].reshape(s[:-3] + s[-2:])
            for i in range(len(self.targets))
        ], axis=-3)

        if isinstance(inputs, pyqg.Model):
            return preds.astype(inputs.q.dtype)
        else:
            return preds

    def mse(self, inputs, targets, **kw):
        y_true = targets.reshape(-1, np.prod(targets.shape[1:]))
        y_pred = self.predict(inputs).reshape(-1, np.prod(targets.shape[1:]))
        return np.mean(np.sum((y_pred - y_true)**2, axis=1))

    def fit(self, inputs, targets, rescale=False, inputs_test = None, targets_test = None, **kw):
        if rescale or not hasattr(self, 'input_scale') or self.input_scale is None:
            self.input_scale = ChannelwiseScaler(inputs)
        if rescale or not hasattr(self, 'output_scale') or self.output_scale is None:
            self.output_scale = ChannelwiseScaler(targets, zero_mean=self.is_zero_mean)

        print('Current mode:', self.training)
        self.train()
        print('Mode after train():', self.training)
        
        if type(self).__name__ == 'FullyCNN':
            train(self,
                  self.input_scale.transform(inputs),
                  self.output_scale.transform(targets),
                  **kw)
        elif type(self).__name__ == 'ProbabilisticCNN':
            train_probabilistic(self,
                  self.input_scale.transform(inputs),
                  self.output_scale.transform(targets),
                  inputs_test = self.input_scale.transform(inputs_test),
                  targets_test = self.output_scale.transform(targets_test),
                  **kw)

    def save(self, path):
        os.system(f"mkdir -p {path}")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cpu()
        torch.save(self.state_dict(), f"{path}/weights.pt")
        self.to(device)
        with open(f"{path}/input_scale.pkl", 'wb') as f:
            pickle.dump(self.input_scale, f)
        with open(f"{path}/output_scale.pkl", 'wb') as f:
            pickle.dump(self.output_scale, f)
        with open(f"{path}/inputs.pkl", 'wb') as f:
            pickle.dump(self.inputs, f)
        with open(f"{path}/targets.pkl", 'wb') as f:
            pickle.dump(self.targets, f)
        try:
            with open(f"{path}/loss_history.pkl", 'wb') as f:
                pickle.dump(self.loss_history, f)
        except:
            pass
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
        try:
            with open(f"{path}/loss_history.pkl", 'rb') as f:
                model.loss_history = pickle.load(f)
        except:
            pass
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
    def __init__(self, inputs, targets, padding='same', batch_norm=True, zero_mean=False, channel_type = None):
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

class ProbabilisticCNN(nn.Sequential, ScaledModel):
    def __init__(self, inputs, targets, zero_mean = True, channel_type = None):
        """
        Inputs and targets are lists of tuples, for example:

            input = [('q',0)]
            
        Here 'q' represents dynamic variable, 0 - ocean layer. 
        This data will be extracted from dataset type.

        Method forward works on NORMALIZED torch.tensor of size:

            batch x channels x height x width

        First half of output channels are for mean,
        second half for stds. 
        
        """
        self.inputs = inputs
        self.targets = targets

        n_in = len(inputs)
        n_out = len(targets) * 2 # mean + std
        self.channel_type = channel_type # type of stochastic channels, var or std

        self.padding_mode = 'circular' # fast pass of parameter

        blocks = []
        blocks.extend(self._make_subblock(n_in,128,5))               #1
        blocks.extend(self._make_subblock(128,64,5))                 #2
        blocks.extend(self._make_subblock(64,32,3))                  #3
        blocks.extend(self._make_subblock(32,32,3))                  #4
        blocks.extend(self._make_subblock(32,32,3))                  #5
        blocks.extend(self._make_subblock(32,32,3))                  #6
        blocks.extend(self._make_subblock(32,32,3))                  #7
        blocks.append(nn.Conv2d(32, n_out, 3,                        #8
            padding='same', padding_mode=self.padding_mode))

        super().__init__(*blocks)
        
        self.set_zero_mean(zero_mean)       # for proper scaling of targets
        self.inference_stochastic = True    # trigger for method forward

    def forward(self, x):
        # number of channels predicting deterministic part
        det_ch = self[-1].out_channels//2

        x = super().forward(x)

        if self.inference_stochastic:
            r = torch.zeros_like(x)
            r[:,:det_ch,:,:] = x[:,:det_ch,:,:]
            r[:,det_ch:,:,:] = nn.functional.softplus(x[:,det_ch:,:,:]).mean()
            return r
        else:
            return x[:,:det_ch,:,:]
    
    def generate_noise(self, yhat):        
        """
        Takes full prediction of network
        and generates noise
        """
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        det_ch = self[-1].out_channels//2

        stds = yhat[:,det_ch:,:,:]
        if (self.channel_type == 'var'):
            stds = torch.sqrt(stds)

        return stds * torch.randn(stds.shape).to(device)

    def _make_subblock(self, input_channels, output_channels, filter_size):
        conv = nn.Conv2d(input_channels, output_channels, 
            filter_size, padding='same', padding_mode=self.padding_mode)
        subbloc = [conv, nn.ReLU()]
        subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc

    def check_channels(self):
        n_in = self[0].in_channels
        det_ch = self[-1].out_channels//2
        self.inference_stochastic = True

        self.to('cpu')
        print('Current mode:', self.training)
        self.eval()
        print('Mode after eval():', self.training)
        x = torch.rand(10,n_in,64,64).to('cpu')
        y = self.forward(x).to('cpu')
        print('min, max mean:', y[:,:det_ch,:,:].min().item(), y[:,:det_ch,:,:].max().item())
        print('min, max std :', y[:,det_ch:,:,:].min().item(), y[:,det_ch:,:,:].max().item())
