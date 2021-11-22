import os
import json
import pyqg_subgrid_experiments as pse
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_set', type=str, default="/scratch/zanna/data/pyqg/data/train/*.nc")
parser.add_argument('--test_set', type=str, default="/scratch/zanna/data/pyqg/data/test/*.nc")
parser.add_argument('--transfer_set', type=str, default="/scratch/zanna/data/pyqg/data/transfer/*.nc")
parser.add_argument('--save_dir', type=str)
parser.add_argument('--inputs', type=str, default="u,v,q")
parser.add_argument('--targets', type=str, default="q_forcing_advection")
parser.add_argument('--zero_mean', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--layerwise_inputs', type=int, default=0)
parser.add_argument('--layerwise_targets', type=int, default=0)
args = parser.parse_args()

save_dir = args.save_dir
os.system(f"mkdir -p {save_dir}") 

with open(f"{save_dir}/model_config.json", 'w') as f:
    f.write(json.dumps(args.__dict__))

train = pse.Dataset(args.train_set)

param = pse.CNNParameterization.train_on(train, save_dir,
            inputs=args.inputs.split(","),
            targets=args.target.split(","),
            layerwise_inputs=args.layerwise_inputs,
            layerwise_targets=args.layerwise_targets,
            zero_mean=args.zero_mean,
            num_epochs=args.num_epochs)

param.test_on(
    pse.Dataset(args.test_set),
    os.path.join(save_dir, "test")
)

param.test_on(
    pse.Dataset(args.transfer_set),
    os.path.join(save_dir, "transfer")
)
