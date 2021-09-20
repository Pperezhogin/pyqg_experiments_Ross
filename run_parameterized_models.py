template = """#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --job-name=generate_datasets
#SBATCH --mail-type=END
#SBATCH --mail-user=asross@nyu.edu

module purge

singularity exec --nv --overlay /scratch/asr9645/envs/m2lines.ext3:ro /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "source /ext3/env.sh; python -u run_model_in_pyqg.py --model={model} --inputs={inputs} --target={target} --run_idx={idx}"
"""

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--inputs', type=str, default="u,v,q")
parser.add_argument('--target', type=str, default="q_forcing_advection")
parser.add_argument('--model', type=str, default="fully_cnn")
parser.add_argument('--n_runs', type=int, default=16)
parser.add_argument('--start_idx', type=int, default=0)
args = parser.parse_args()

for i in range(args.n_runs):
    cmd = template.format(
            inputs=args.inputs,
            target=args.target,
            model=args.model,
            idx=i + args.start_idx,
        )

    with open('tmp.slurm', 'w') as f:
        f.write(cmd)

    os.system("cat tmp.slurm | sbatch")
