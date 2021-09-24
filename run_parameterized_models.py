template = """#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=generate_datasets
#SBATCH --mail-type=END
#SBATCH --mail-user=asross@nyu.edu

module purge

singularity exec --nv --overlay /scratch/asr9645/envs/m2lines.ext3:ro /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "source /ext3/env.sh; python -u run_model_in_pyqg.py {args}"
"""

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str)
parser.add_argument('--inputs', type=str, default="u,v,q")
parser.add_argument('--n_runs', type=int, default=16)
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--divide_by_dt', action='store_true', default=False)
args = parser.parse_args()

for i in range(args.n_runs):
    cmd_args = f"--inputs={args.inputs} --run_idx={i + args.start_idx} --save_dir={args.save_dir}"
    if args.divide_by_dt:
        cmd_args += " --divide_by_dt"

    cmd = template.format(args=cmd_args)

    with open('tmp.slurm', 'w') as f:
        f.write(cmd)

    os.system("cat tmp.slurm | sbatch")
