import os
import sys
import argparse
dirname = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dirname)
from slurm_job import SlurmJob

model_dir = "/scratch/zanna/data/pyqg/models"
job_script = os.path.join(dirname, '../pyqg_subgrid_experiments/train.py')

def launch_job(**kwargs):
    job = SlurmJob(job_script, time="12:00:00", mem="32GB", gpu="rtx8000:1", **kwargs)
    job.launch()

argsets = {}

inputs = [
    'ddx_u,ddx_v',
    'q',
    'u,v',
    'q,u,v',
    'dqdt_through_lores',
    'q,dqdt_through_lores',
]

outputs = [
    'q_forcing_total',
    'q_forcing_advection',
    'uq_subgrid_flux,vq_subgrid_flux',
    'u_forcing_advection,v_forcing_advection',
    'uu_subgrid_flux,uv_subgrid_flux,vv_subgrid_flux',
]

for inp in inputs:
    for outp in outputs:
        for zero_mean in [0,1]:
            for layer_in, layer_out in [(0,0), (1,1), (0,1)]:
                key = '_'.join([
                    'fcnn',
                    inp.replace(',','-'),
                    outp.replace(',','-'),
                    f"zeromean{zero_mean}",
                    f"layerwise{layer_in}{layer_out}"
                ])
                argsets[key] = dict(
                    inputs=inp,
                    targets=outp,
                    zero_mean=zero_mean,
                    layerwise_inputs=layer_in,
                    layerwise_targets=layer_out,
                )

for restart in range(1):
    for name, kw in argsets.items():
        launch_job(save_dir=f"{model_dir}/{name}/{restart}", **kw)
