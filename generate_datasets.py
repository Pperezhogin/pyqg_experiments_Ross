cmd_template = """singularity exec --nv --overlay /scratch/asr9645/envs/m2lines.ext3:ro /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "source /ext3/env.sh; python -u /home/asr9645/pyqg_experiments/pyqg_subgrid_dataset.py {arguments}"
"""

slurm_template = """#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=generate_datasets
#SBATCH --mail-type=END
#SBATCH --mail-user=asross@nyu.edu

module purge

{commands}
"""

data_dir = '/scratch/zanna/data/pyqg'
settings = [
    dict(data_dir=f"{data_dir}/single_run_all_steps", n_runs=1, sampling_freq=1),
    dict(data_dir=f"{data_dir}/many_runs_few_steps_uniform", n_runs=1000, sampling_freq=1000, sampling_mode='uniform'),
    dict(data_dir=f"{data_dir}/many_runs_few_steps_irregular", n_runs=1000, sampling_freq=1000, sampling_mode='exponential'),
    #dict(data_dir=f"{data_dir}/test", n_runs=1, sampling_freq=100)
]

commands = []
for s in settings:
    args = " ".join([f"--{k}={v}" for k, v in s.items()])
    commands.append(cmd_template.format(arguments=args))

slurm = slurm_template.format(commands="\n".join(commands))

with open('generate_datasets.slurm', 'w') as f:
    f.write(slurm)



