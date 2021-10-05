import os
import re

class SlurmJob(object):
    def __init__(self, python_file, time="12:00:00", mem="32GB", gpu="rtx8000:1", job_name=None, save_command=True, **kwargs):
        self.time = time
        self.mem = mem
        self.gpu = gpu
        self.kwargs = kwargs
        self.python_file = python_file
        self.job_name = job_name or python_file.split('/')[-1].split('.')[0]
        self.save_command = int(save_command)

    @property
    def save_dir(self):
        return self.kwargs.get('save_dir', None)

    @property
    def command(self):
        return ('singularity exec --nv --overlay ' + 
                self.singularity_env +
                ' /scratch/work/public/singularity/cuda11.1-cudnn8-devel-ubuntu18.04.sif /bin/bash -c "source /ext3/env.sh; python -u ' +
                self.python_file + ' ' + 
                self.args+'"')

    @property
    def args(self):
        args = []
        for k, v in self.kwargs.items():
            args.append(f"--{k}={v}")
        return " ".join(args)

    @property
    def singularity_env(self):
        if 'SINGULARITY_ENV' in os.environ:
            return os.environ['SINGULARITY_ENV']
        else:
            return "/scratch/asr9645/envs/m2lines.ext3:ro"

    @property
    def lines(self):
        lines = [
            "#!/bin/bash",
            "",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks-per-node=1",
            "#SBATCH --cpus-per-task=1",
            f"#SBATCH --mem={self.mem}",
            f"#SBATCH --time={self.time}",
            f"#SBATCH --job-name={self.job_name}",
        ]

        if self.gpu:
            lines += [f"#SBATCH --gres=gpu:{self.gpu}"]

        if self.save_dir and self.save_command:
            lines += [f"#SBATCH -o {os.path.join(self.save_dir, 'job.out')}"]

        lines += ["", "module purge", "", self.command]

        return lines

    @property
    def text(self):
        return "\n".join(self.lines)

    def __repr__(self):
        return self.text

    def launch(self):
        if self.save_dir and self.save_command:
            os.system(f"mkdir -p {self.save_dir}")
            command_filename = os.path.join(self.save_dir, "job.slurm")
        else:
            command_filename = "tmp.slurm"

        with open(command_filename, 'w') as f:
            f.write(self.text)

        os.system(f"cat {command_filename} | sbatch")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--python_file', type=str)
    parser.add_argument('--launch_job', type=int, default=1)
    args, extra = parser.parse_known_args()
    cmd_args = {}
    for param in extra:
        key, val = param.split('=')
        cmd_args[key.replace('--', '')] = val
    job = SlurmJob(args.python_file, **cmd_args)
    if args.launch_job:
        job.launch()
    else:
        print(job)
