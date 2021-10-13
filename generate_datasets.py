import os
import sys
import argparse
dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dirname)
from slurm_job import SlurmJob

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/scratch/zanna/data/pyqg/datasets')
args = parser.parse_args()
data_dir = args.data_dir
job_script = os.path.join(dirname, 'generate_dataset.py')

def launch_job(**kwargs):
    SlurmJob(job_script, time="01:00:00", mem="32GB", gpu="", **kwargs).launch()

for i in range(250):
    launch_job(save_dir=f"{data_dir}/train/{i}")

for i in range(25):
    launch_job(save_dir=f"{data_dir}/test/{i}")
    launch_job(save_dir=f"{data_dir}/test/{i}", control=1)
    launch_job(save_dir=f"{data_dir}/test/{i}", physical=1)

for i in range(25):
    launch_job(transfer_test=1, save_dir=f"{data_dir}/transfer/{i}")
    launch_job(transfer_test=1, save_dir=f"{data_dir}/transfer/{i}", control=1)
    launch_job(transfer_test=1, save_dir=f"{data_dir}/transfer/{i}", physical=1)
