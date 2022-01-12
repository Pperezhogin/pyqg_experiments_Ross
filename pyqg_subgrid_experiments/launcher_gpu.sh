#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu
#SBATCH --job-name=torch

module purge

singularity exec --nv \
	    --overlay /scratch/pp2681/python-container/python-overlay.ext3:ro \
	    /scratch/work/public/singularity/cuda11.2.2-cudnn8-devel-ubuntu20.04.sif \
	    /bin/bash -c "source /ext3/env.sh; time python -u train_pnn.py --regularization=0.0 --num_epochs=200 --learning_rate=0.0001 --channel_type='var' --epoch_var=50 --save_dir='PCNN_var_0_0001_epoch_var_50_mean_linear' "
