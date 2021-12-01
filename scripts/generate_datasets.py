import os
import sys
import argparse
dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dirname)
from slurm_job import SlurmJob

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type=str, default='/scratch/zanna/data/pyqg/data')
args = parser.parse_args()
data_dir = args.data_dir
job_script = os.path.join(dirname, '../pyqg_subgrid_experiments/simulate.py')

def launch_job(**kwargs):
    SlurmJob(job_script, time="01:00:00", mem="32GB", gpu="", **kwargs).launch()

for i in range(250):
    launch_job(save_to=f"{data_dir}/train/{i}.nc")

for i in range(25):
    launch_job(save_to=f"{data_dir}/test/{i}.nc")
    launch_job(transfer_test=1, save_to=f"{data_dir}/transfer/{i}.nc")

for i in range(5):
    launch_job(save_to=f"{data_dir}/test/hires/{i}.nc", control=1, nx=256)
    launch_job(save_to=f"{data_dir}/test/lores/{i}.nc", control=1)
    
    launch_job(transfer_test=1, save_to=f"{data_dir}/transfer/hires/{i}.nc", control=1, nx=256)
    launch_job(transfer_test=1, save_to=f"{data_dir}/transfer/lores/{i}.nc", control=1)
