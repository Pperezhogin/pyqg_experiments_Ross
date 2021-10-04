import re
import os

template = """#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:rtx8000:1
#SBATCH --job-name=train_model
#SBATCH -o {outfile}

module purge

singularity exec --nv --overlay /scratch/asr9645/envs/m2lines.ext3:ro /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "source /ext3/env.sh; python -u train_model.py {args}"
"""

model_dir = "/scratch/zanna/data/pyqg/models"

argsets = [
    "--save_dir={model_dir}/fullycnn_skip_0.0/{i}",
    "--save_dir={model_dir}/fullycnn_skip_0.5/{i} --skip_datasets=250",
    "--save_dir={model_dir}/fullycnn_skip_0.8/{i} --skip_datasets=400",
    "--save_dir={model_dir}/fullycnn_skip_0.9/{i} --skip_datasets=450",
    "--save_dir={model_dir}/fullycnn_skip_0.95/{i} --skip_datasets=475",
    "--save_dir={model_dir}/fullycnn_skip_0.99/{i} --skip_datasets=495",
    #"--save_dir={model_dir}/fullycnn_no_constraints/{i} --zero_mean=0",
    #"--save_dir={model_dir}/fullycnn_rescale_loss/{i} --scaler=logpow",
    #"--save_dir={model_dir}/fullycnn_sparse_grads/{i} --l1_grads=0.0001",
    #"--save_dir={model_dir}/fullycnn_way2_forcing/{i} --target=q_forcing_model",
    #"--save_dir={model_dir}/fullycnn_q_only/{i} --inputs=q",
    #"--save_dir={model_dir}/fullycnn_uv_only/{i} --inputs=u,v",
]

for i in range(3):
    for argset in argsets:
        args = argset.format(model_dir=model_dir, i=i)
        mdir = re.search(r'--save_dir=([^\s]+)', args).group(1)
        jobfile = f"{mdir}/job.slurm"
        outfile = f"{mdir}/job.out"
        cmd = template.format(args=args, outfile=outfile)

        print(mdir)

        os.system(f"mkdir -p {mdir}")

        with open(jobfile, 'w') as f:
            f.write(cmd)

        os.system(f"cat {jobfile} | sbatch")
