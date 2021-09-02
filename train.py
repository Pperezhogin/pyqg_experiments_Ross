import sys; sys.path.append('.')
import argparse
import numpy as np
import xarray as xr
import torch
import os
import csv
from pyqg_subgrid_dataset import PYQGSubgridDataset
from models import *
from sklearn.preprocessing import StandardScaler

datasets = [
    PYQGSubgridDataset(ds)
    for ds in os.listdir('/scratch/zanna/data/pyqg')
]

results = open('basic_cnn_results.csv', 'w')

def write_result(s):
    print(s)
    results.write(s)

write_result("Train,Test,MSE")

for ds in datasets:
    X_train, X_test, Y_train, Y_test = ds.train_test_split()
    X_scale = StandardScaler().fit(X_train)
    Y_scale = StandardScaler().fit(Y_train)
    shape = (1, ds.resolution, ds.resolution)
    model = BasicCNN(shape, shape)
    model.set_scales(X_scale, Y_scale)
    model.fit(X_train, Y_train)
    model.save(ds.path('basic_cnn'))

    for ds2 in datasets:
        _, X_test2, __, Y_test2 = ds2.train_test_split()
        mse = model.mse(X_test2, Y_test2)
        write_result(f"{ds.name},{ds2.name},{mse}")
