import torch
import torch.nn as nn
import torch.optim as optim
import pickle
import numpy as np
from collections import OrderedDict

def minibatch(inputs, targets, batch_size=64, as_tensor=True):
    assert len(inputs) == len(targets)
    order = np.arange(len(inputs))
    np.random.shuffle(order)
    steps = int(np.ceil(len(inputs) / batch_size))
    xform = torch.as_tensor if as_tensor else lambda x: x
    for step in range(steps):
        idx = order[step*batch_size:(step+1)*batch_size]
        x = xform(inputs[idx])
        y = xform(targets[idx])
        yield x, y

def train(net, inputs, targets, num_epochs=50, batch_size=64, learning_rate=0.001, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        for x, y in minibatch(inputs, targets, batch_size=batch_size):
            optimizer.zero_grad()
            yhat = net.forward(x.to(device))
            loss = criterion(yhat, y.to(device))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_steps += 1
        print(f"Loss after Epoch {epoch+1}: {epoch_loss/epoch_steps}")

class ScaledModel(object):
    def forward(self, x):
        r = super().forward(x)
        if hasattr(self, 'zero_mean') and getattr(self, 'zero_mean'):
            return r - r.mean(dim=(1,2,3), keepdim=True)
        else:
            return r

    def set_zero_mean(self, zero_mean=True):
        self.zero_mean = zero_mean

    def set_scales(self, input_scale, output_scale):
        self.input_scale = input_scale
        self.output_scale = output_scale

    def predict(self, inputs, device=None):
        assert isinstance(inputs, np.ndarray)
        scaled = self.input_scale.transform(inputs)
        tensor = torch.as_tensor(scaled)
        if device is not None:
            tensor = tensor.to(device)
        with torch.no_grad():
            output = self.forward(tensor).cpu().numpy()
        return self.output_scale.inverse_transform(output)

    def mse(self, inputs, targets, device=None):
        mse = nn.MSELoss()
        mses = []
        for x, y in minibatch(inputs, targets, as_tensor=False):
            yhat = self.predict(x)
            errs = np.sum((y-yhat)**2, axis=1)
            mses.append(errs.mean())
        return np.mean(mses)

    def fit(self, inputs, targets, **kw):
        train(self,
              self.input_scale.transform(inputs),
              self.output_scale.transform(targets),
              **kw)

    def save(self, path):
        torch.save(self.state_dict(), path)
        print(f"saved model to {path}")
        with open(f"{path}.input_scale.pkl", 'wb') as f:
            pickle.dump(self.input_scale, f)
        with open(f"{path}.output_scale.pkl", 'wb') as f:
            pickle.dump(self.output_scale, f)
        print(f"saved scales")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        with open(f"{path}.input_scale.pkl", 'rb') as f:
            self.input_scale = pickle.load(f)
        with open(f"{path}.output_scale.pkl", 'rb') as f:
            self.output_scale = pickle.load(f)

class BasicScaler(object):
    def __init__(self, mu=0, sd=1):
        self.mu = mu
        self.sd = sd
    
    def fit(self, x):
        assert(len(x.shape)==2)
        self.mu = np.mean(x)
        self.sd = np.std(x)
        
    def transform(self, x):
        return (x - self.mu) / self.sd
    
    def inverse_transform(self, z):
        return z * self.sd + self.mu

class BasicCNN(nn.Sequential, ScaledModel):
    def __init__(self, input_shape, output_shape, pad='circular'):
        conv = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(input_shape[0], 64, 5, padding_mode=pad)),
            ('norm1', nn.BatchNorm2d(64)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(2)),

            ('conv2', nn.Conv2d(64, 32, 5, padding_mode=pad)),
            ('norm2', nn.BatchNorm2d(32)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(2)),

            ('flat', nn.Flatten())
        ]))
        fc_size = conv(torch.rand(2,*input_shape)).shape[1]
        super().__init__(OrderedDict([
            ('conv', conv),
            ('fc1', nn.Linear(fc_size, 256)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(256, np.product(output_shape))),
            ('unflatten', nn.Unflatten(1, output_shape))
        ]))

class FullyCNN(nn.Sequential, ScaledModel):
    def __init__(self, n_in: int = 3, n_out: int = 1, padding='same', batch_norm=True):
        if padding is None:
            padding_5 = 0
            padding_3 = 0
        elif padding == 'same':
            padding_5 = 2
            padding_3 = 1
        else:
            raise ValueError('Unknow value for padding parameter.')
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

    def _make_subblock(self, conv):
        subbloc = [conv, nn.ReLU()]
        if self.batch_norm:
            subbloc.append(nn.BatchNorm2d(conv.out_channels))
        return subbloc
