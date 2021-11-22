import os
import sys
import argparse
dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dirname)
from slurm_job import SlurmJob

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/scratch/zanna/data/pyqg/data2')
args = parser.parse_args()
data_dir = args.data_dir
job_script = os.path.join(dirname, '../pyqg_subgrid_experiments/simulate.py')

def launch_job(**kwargs):
    SlurmJob(job_script, time="01:00:00", mem="32GB", gpu="", **kwargs).launch()

for i in range(1):
    launch_job(save_to=f"{data_dir}/train/{i}.nc")

for i in range(1):
    launch_job(save_to=f"{data_dir}/test/{i}.nc")
    launch_job(transfer_test=1, save_to=f"{data_dir}/transfer/{i}.nc")
