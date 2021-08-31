#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=generate_datasets
#SBATCH --mail-type=END
#SBATCH --mail-user=asross@nyu.edu

module purge

singularity exec --nv --overlay /scratch/asr9645/envs/m2lines.ext3:ro /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "source /ext3/env.sh; python -u /home/asr9645/pyqg_experiments/pyqg_subgrid_dataset.py --data_dir=/scratch/zanna/data/pyqg/single_run_all_steps --n_runs=1 --sampling_freq=1"

singularity exec --nv --overlay /scratch/asr9645/envs/m2lines.ext3:ro /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "source /ext3/env.sh; python -u /home/asr9645/pyqg_experiments/pyqg_subgrid_dataset.py --data_dir=/scratch/zanna/data/pyqg/many_runs_few_steps_uniform --n_runs=1000 --sampling_freq=1000 --sampling_mode=uniform"

singularity exec --nv --overlay /scratch/asr9645/envs/m2lines.ext3:ro /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "source /ext3/env.sh; python -u /home/asr9645/pyqg_experiments/pyqg_subgrid_dataset.py --data_dir=/scratch/zanna/data/pyqg/many_runs_few_steps_irregular --n_runs=1000 --sampling_freq=1000 --sampling_mode=exponential"

