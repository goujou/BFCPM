# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.15.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# ## Script to run several python files (from notebooks) one after another and store their output
#
# - check that the ipython file (``run_pre_spinup.ipynb``, ``run_simulation.ipynb``, ``run_benchmarking.ipynb``) is synchronized with its python file (via ``jupytext --set-formats ipynb,py --sync yourNotebook.ipynb``)
# - add the python file with its parameters to ``notebook_datas``

# +
import subprocess
from pathlib import Path

from BFCPM import LOGS_PATH
from tqdm import tqdm

# +
# pre_spinup_date = "2023-06-07" # corrected wood density
pre_spinup_date = "2023-07-25"  # publication

sim_cohort_name = "sensitivity_empty_stand"

# simulation data

# sim_date = "2023-05-15" # current simulation for paper
# sim_date = "2023-06-05" # other emergency strategies
# sim_date = "2023-06-08" # corrected wood density
# sim_date = "2023-06-19" # at emergency automatically thin stand to SBA = 18
# sim_date = "2023-07-11" # mixed-aged_pine
#sim_date = "2023-07-26"  # publication
sim_date = "2023-11-24"

sim_names = [
    "mixed-aged_pine_long",
    "even-aged_pine_long",
    "even-aged_spruce_long",
    "even-aged_mixed_long",
]

# tree density of even-aged stands before pre-commercial thinning
sim_N = 2_000

# emergency actions to be taken
emergency_action_str, emergency_direction, emergency_stand_action_str = (
    "Die",
    "below",
    "ThinStandToSBA18",
)
# emergency_action_str, emergency_direction = "Cut", "above"
# emergency_action_str, emergency_direction = "CutWait3AndReplant", "above"
# emergency_action_str, emergency_direction = "Thin", "above"



# +
sensitivity_param_names = ["R_mL", "S_R", "rho_RL", "Vcmax"]
#sensitivity_qs = [0.95, 1.00, 1.05]
sensitivity_qs = [0.9, 1.0, 1.1]

sensitivity_strs = list()
for param_name in sensitivity_param_names:
    for q in sensitivity_qs:
        s = f"{param_name},{q}"
        sensitivity_strs.append(s)

print(sensitivity_strs)
# -

logs_path = LOGS_PATH.joinpath(f"{sim_cohort_name}").joinpath(f"{sim_date}")
logs_path.mkdir(exist_ok=True, parents=True)
logs_path

# four scenarios simulation data including continuous-cover spinup
four_scenarios_data = [
    {
        "filename": "run_sensitivity_empty_stand.py",
        "params": [
            pre_spinup_date,
            sim_cohort_name,
            sim_date,
            sim_name,
            str(sim_N),
            emergency_action_str,
            emergency_direction,
            emergency_stand_action_str,
            sensitivity_str
        ],
    }
    for sim_name in sim_names
    for sensitivity_str in sensitivity_strs
]
len(four_scenarios_data)

# ## Run scenarios in parallel

import os
import dask
from dask.distributed import Client, LocalCluster

dask.config.set({"distributed.nanny.environ.MALLOC_TRIM_THRESHOLD_": 0})

os.environ["MALLOC_TRIM_THRESHOLD_"] = str(dask.config.get("distributed.nanny.environ.MALLOC_TRIM_THRESHOLD_"))

cluster = LocalCluster(n_workers = 16, threads_per_worker=1)
c = Client(cluster)
c


def run_nb(filename, params, logs_path, nr):
    print("Starting:", filename)
    print(params)
    logfile_path = logs_path.joinpath(f"run_sensitivity_empty_stand_{nr}.log")
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
for nr, nb_data in enumerate(four_scenarios_data):
    filename = nb_data["filename"]
    params = nb_data["params"]

    result = dask.delayed(run_nb)(filename, params, logs_path, nr)
    results.append(result)

# +
# %%time

dask.compute(results)
# -


