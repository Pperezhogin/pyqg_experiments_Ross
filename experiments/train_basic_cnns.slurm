#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu
#SBATCH --job-name=train_cnns
#SBATCH --mail-type=END
#SBATCH --mail-user=asross@nyu.edu

module purge

singularity exec --nv --overlay /scratch/asr9645/envs/m2lines.ext3:ro /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "source /ext3/env.sh; python -u experiments/train_basic_cnns.py"
