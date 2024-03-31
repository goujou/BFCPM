# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Script to run several python files (from notebooks) one after another and store their output
#
# - check that the ipython file (``run_pre_spinup.ipynb``, ``run_simulation.ipynb``, ``run_benchmarking.ipynb``) is synchronized with its python file (via ``jupytext --set-formats ipynb,py --sync yourNotebook.ipynb``)
# - add the python file with its parameters to ``notebook_datas``

# ## ISSUES
#
# - year 2018/2019 kills trees (very low GPP): Most drastically, when only forcing from 2000 is used (delay=0)
# - if one tree dies and the year is repeated, then another tree dies, and the year is repeated, the former tree is there again and we end up in an endless loop (delay=0)
# - delay=10: cc_tree0 is emergency cut, bot action=[] because the tree is not considered alive (it was shortly replanted)

# +
import os
import dask

dask.config.set({"distributed.nanny.environ.MALLOC_TRIM_THRESHOLD_": 0})
os.environ["MALLOC_TRIM_THRESHOLD_"] = str(dask.config.get("distributed.nanny.environ.MALLOC_TRIM_THRESHOLD_"))

# +
#from dask_mpi import initialize
#initialize()


# +
from dask.distributed import Client
import socket

print("running client")
#client = Client()  # Connect this local process to remote workers

from dask.distributed import LocalCluster
cluster = LocalCluster(n_workers=20, threads_per_worker=1)
client = Client(cluster)
print("client running")

host = client.run_on_scheduler(socket.gethostname)
port = client.scheduler_info()['services']['dashboard']
login_node_address = "hrme0001@rackham.uppmax.uu.se" # Change this to the address/domain of your login node

subhost = host.split(".")[0]
s = f"ssh -L {8892}:{subhost}:{port} {login_node_address}"
print()
print("Dashboard SSH command")
print(s)
print()
client


# +
import subprocess
from pathlib import Path

from BFCPM import LOGS_PATH
from tqdm import tqdm
import numpy as np

# +
#    parser.add_argument("pre_spinup_date", type=str)
#    parser.add_argument("pre_spinup_species", type=str)
#    parser.add_argument("common_spinup_dmp_filepath", type=str) # continuous-cover (age-distributed) spinup

#    parser.add_argument("sim_date", type=str)
#    parser.add_argument("species", type=str)
#    parser.add_argument("coarseness", type=int)

#    parser.add_argument("delay", type=int)


# +
# pre_spinup_date = "2023-06-07" # corrected wood density
#pre_spinup_date = "2023-07-25"  # publication
pre_spinup_date = "2023-10-18"
pre_spinup_species = "pine"

coarseness = 12
common_spinup_dmp_filepath = f"DWC_common_spinup_clear_cut_{pre_spinup_species}_{coarseness:02d}"


#sim_cohort_name = "sensitivity_full_sim"
sim_cohort_name = "patch_transition_at_end_smooth"

# simulation data

# sim_date = "2023-05-15" # current simulation for paper
# sim_date = "2023-06-05" # other emergency strategies
# sim_date = "2023-06-08" # corrected wood density
# sim_date = "2023-06-19" # at emergency automatically thin stand to SBA = 18
# sim_date = "2023-07-11" # mixed-aged_pine
#sim_date = "2023-07-26"  # publication
#sim_date = "2023-11-26"
#sim_date = "2024-03-05"
sim_date = "2024-03-07"

species = "pine"
# -


logs_path = LOGS_PATH.joinpath(f"{sim_cohort_name}").joinpath(f"{sim_date}")
logs_path.mkdir(exist_ok=True, parents=True)
logs_path

# +
all_delays = list(range(80))
delays = all_delays[0::4] 
#delays = all_delays[2::4]
#delays = all_delays[1::4]
#delays = all_delays[3::4]

# failed ones
#delays = [52, 73, 75, 76, 78, 79]

print(len(delays))
delays
# -

# four scenarios simulation data including continuous-cover spinup
delay_data = [
    {
        "filename": "patch_transition_at_end_smooth.py",
        "params": [
            # make sure all parameters are strings!
            pre_spinup_date,
            pre_spinup_species,
            common_spinup_dmp_filepath,
            
            sim_date,
            species,
            str(coarseness), 

            str(delay)
        ],
    }
    for delay in delays
]
print(len(delay_data))
print(delay_data[:2])

# ## Run scenarios in parallel

def run_nb(filename, params, logs_path, nr):
    print("Starting:", filename)
    print(params)
    logfile_path = logs_path.joinpath(f"patch_transition_at_end_smooth_{nr}.log")
    print("Logfile path:", logfile_path)
    with open(logfile_path, "w") as f:
        process = subprocess.run(
            ["python", filename] + params, stdout=f, universal_newlines=True
        )
    #        process.wait()

    print("Finished:", filename)
    print(logfile_path.absolute())
    print(process)

    return "Done"


results = list()
for nr, nb_data in enumerate(delay_data):
    filename = nb_data["filename"]
    params = nb_data["params"]

    result = dask.delayed(run_nb)(filename, params, logs_path, nr)
    results.append(result)

# +
# %%time

dask.compute(results)
# -

print("closing client")
client.close()
print("client closed")


print("closing cluster")
cluster.close()
print("cluster closed")




