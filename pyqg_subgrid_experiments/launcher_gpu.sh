#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu
#SBATCH --job-name=torch

module purge

singularity exec --nv \
	    --overlay /scratch/pp2681/python-container/python-overlay.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; time python -u train_pnn.py --num_epochs=1015 --learning_rate=0.0001 --channel_type='var' --save_dir='PCNN_var_0_0001_annealing' "
