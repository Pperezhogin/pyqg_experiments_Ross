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
    steps = len(inputs) // batch_size
    xform = torch.as_tensor if as_tensor else lambda x: x
    for step in range(steps):
        idx = order[step*batch_size:(step+1)*batch_size]
        x = xform(inputs[idx])
        y = xform(targets[idx])
        yield x, y

def train(net, inputs, targets, num_epochs=50, batch_size=64, learning_rate=0.001):
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0
        for x, y in minibatch(inputs, targets, batch_size=batch_size):
            optimizer.zero_grad()
            yhat = net(x)
            loss = criterion(yhat, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_steps += 1
        print(f"Loss after Epoch {epoch+1}: {epoch_loss/epoch_steps}")

class ScaledModel(object):
    def set_scales(self, input_scale, output_scale):
        self.input_scale = input_scale
        self.output_scale = output_scale

    def predict(self, inputs):
        assert isinstance(inputs, np.ndarray)
        scaled = self.input_scale.transform(inputs)
        tensor = torch.as_tensor(scaled)
        with torch.no_grad():
            output = self.forward(tensor).numpy()
        return self.output_scale.inverse_transform(output)

    def mse(self, inputs, targets):
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
        with open(f"{path}.input_scale.pkl", 'wb') as f:
            pickle.dump(self.input_scale, f)
        with open(f"{path}.output_scale.pkl", 'wb') as f:
            pickle.dump(self.output_scale, f)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        with open(f"{path}.input_scale.pkl", 'rb') as f:
            self.input_scale = pickle.load(f)
        with open(f"{path}.output_scale.pkl", 'rb') as f:
            self.output_scale = pickle.load(f)

class BasicCNN(nn.Sequential, ScaledModel):
    def __init__(self, input_shape, output_shape):
        conv = nn.Sequential(OrderedDict([
            ('unflatten', nn.Unflatten(1, input_shape)),
            ('conv1', nn.Conv2d(input_shape[0], 32, 5)),
            ('relu1', nn.ReLU()),
            ('pool1', nn.MaxPool2d(2)),
            ('conv2', nn.Conv2d(32, 32, 5)),
            ('relu2', nn.ReLU()),
            ('pool2', nn.MaxPool2d(2)),
            ('flat', nn.Flatten())
        ]))
        fc_size = conv(torch.rand(1,np.product(input_shape))).shape[1]
        super().__init__(OrderedDict([
            ('conv', conv),
            ('fc1', nn.Linear(fc_size, 256)),
            ('relu', nn.ReLU()),
            ('fc2', nn.Linear(256, np.product(output_shape)))
        ]))
