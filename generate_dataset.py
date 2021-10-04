template = """#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=generate_datasets
#SBATCH --mail-type=END
#SBATCH --mail-user=asross@nyu.edu

module purge

singularity exec --nv --overlay /scratch/asr9645/envs/m2lines.ext3:ro /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "source /ext3/env.sh; python -u /home/asr9645/pyqg_experiments/pyqg_subgrid_dataset.py --data_dir={data_dir} --run_idx={idx} --control={control} --sampling_dist={dist} --physical={physical} --transfer_test={transfer_test}"
"""

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str)
parser.add_argument('--n_runs', type=int)
parser.add_argument('--control', type=int, default=0)
parser.add_argument('--physical', type=int, default=0)
parser.add_argument('--sampling_dist', type=str, default='uniform')
parser.add_argument('--start_idx', type=int, default=0)
parser.add_argument('--transfer_test', type=int, default=0)
args = parser.parse_args()

for i in range(args.n_runs):
    cmd = template.format(data_dir=args.data_dir, idx=i + args.start_idx, control=args.control, dist=args.sampling_dist, physical=args.physical, transfer_test=args.transfer_test)

    with open('tmp.slurm', 'w') as f:
        f.write(cmd)

    os.system("cat tmp.slurm | sbatch")
