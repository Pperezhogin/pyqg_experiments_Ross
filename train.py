import sys; sys.path.append('.')
import argparse
import numpy as np
import xarray as xr
import torch
import os
import csv
from sklearn.preprocessing import StandardScaler

from pyqg_subgrid_dataset import PYQGSubgridDataset
from models import *

pyqg_dir = '/scratch/zanna/data/pyqg'
datasets = [
    PYQGSubgridDataset(os.path.join(pyqg_dir, ds))
    for ds in os.listdir(pyqg_dir)
]

with open('basic_cnn_results.csv', 'w') as f:
    f.write("Train,Test,MSE\n")

def write_result(s):
    print(s)
    with open('basic_cnn_results.csv', 'a+') as f:
        f.write(s + "\n")

for ds in datasets:
    print(ds.name)
    X_train, X_test, Y_train, Y_test = ds.train_test_split()
    X_scale = StandardScaler().fit(X_train)
    Y_scale = StandardScaler().fit(Y_train)
    shape = (1, ds.resolution, ds.resolution)
    model = BasicCNN(shape, shape)
    model.set_scales(X_scale, Y_scale)
    model.fit(X_train, Y_train, num_epochs=25)
    model.save(ds.path('basic_cnn'))
    for ds2 in datasets:
        _, X_test2, __, Y_test2 = ds2.train_test_split()
        mse = model.mse(X_test2, Y_test2)
        write_result(f"{ds.name},{ds2.name},{mse}")
