import os
import sys
import argparse
dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dirname)
from slurm_job import SlurmJob

model_dir = "/scratch/zanna/data/pyqg/models"
job_script = os.path.join(dirname, 'train_model.py')

def launch_job(**kwargs):
    job = SlurmJob(job_script, time="12:00:00", mem="32GB", gpu="rtx8000:1", **kwargs)
    job.launch()

argsets = {}
argsets["fullycnn"] = dict()
argsets["fullycnn_no_constraints"] = dict(zero_mean=0)
argsets["fullycnn_rescale_loss"] = dict(scaler='logpow')
argsets["fullycnn_way2_forcing"] = dict(target='q_forcing_model')
argsets["fullycnn_q_only"] = dict(inputs="q")
argsets["fullycnn_uv_only"] = dict(inputs="u,v")
for pct in [0.5,0.8,0.9,0.95,0.99]:
    argsets[f"fullycnn_skip_{pct}"] = dict(skip_datasets=int(500*pct))

for restart in range(3):
    for name, kw in argsets.items():
        launch_job(save_dir=f"{model_dir}/{name}/{restart}", **kw)
