import os
import json
import pyqg_subgrid_experiments as pse
from pyqg_subgrid_experiments.models import ProbabilisticCNN
import argparse

import torch
print('Cuda is avaliable = ', torch.cuda.is_available())


parser = argparse.ArgumentParser()
parser.add_argument('--train_set', type=str, default="/scratch/zanna/data/pyqg/data/train/*.nc")
parser.add_argument('--test_set', type=str, default="/scratch/zanna/data/pyqg/data/test/*.nc")
parser.add_argument('--transfer_set', type=str, default="/scratch/zanna/data/pyqg/data/transfer/*.nc")
parser.add_argument('--save_dir', type=str, default="PCNN")
parser.add_argument('--inputs', type=str, default="u,v,q")
parser.add_argument('--targets', type=str, default="q_forcing_advection")
parser.add_argument('--zero_mean', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--learning_rate', type=float, default=0.0001)
parser.add_argument('--channel_type', type=str, default='var')
parser.add_argument('--regularization', type=float, default=1.)
parser.add_argument('--epoch_var', type=int, default=0)
args = parser.parse_args()

print(args)


train = pse.Dataset(args.train_set)
test = pse.Dataset(args.test_set)
xfer = pse.Dataset(args.transfer_set)

test_dir = os.path.join(args.save_dir, "test")
xfer_dir = os.path.join(args.save_dir, "transfer")
for d in [args.save_dir, test_dir, xfer_dir]:
    os.system(f"mkdir -p {d}") 

with open(f"{args.save_dir}/model_config.json", 'w') as f:
    f.write(json.dumps(args.__dict__))

param = pse.CNNParameterization.train_on(train, args.save_dir,
            inputs=args.inputs.split(","),
            targets=args.targets.split(","),
            layerwise_inputs=False,
            layerwise_targets=True,
            zero_mean=args.zero_mean,
            num_epochs=args.num_epochs,
            model_class=ProbabilisticCNN,
            learning_rate=args.learning_rate,
            dataset_test=test,
            channel_type=args.channel_type,
            cosine_annealing=False,
            regularization = args.regularization,
            epoch_var = args.epoch_var
            )

param.test_offline(test, os.path.join(test_dir, "offline_metrics.nc"))
param.test_offline(xfer, os.path.join(xfer_dir, "offline_metrics.nc"))
