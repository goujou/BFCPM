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

# +
import subprocess
from pathlib import Path

from BFCPM import LOGS_PATH
from tqdm import tqdm

# +
# pre_spinup_date = "2023-06-07" # corrected wood density
#pre_spinup_date = "2023-07-25"  # publication
pre_spinup_date = "2023-11-23_WP_long_only"  # Revision 1, wood products short and long separeted

# continuous-cover spinup
cc_spinup_species = "pine"
cc_spinup_length = 160
cc_spinup_N = 1_500

# simulation data

# sim_date = "2023-05-15" # current simulation for paper
# sim_date = "2023-06-05" # other emergency strategies
# sim_date = "2023-06-08" # corrected wood density
# sim_date = "2023-06-19" # at emergency automatically thin stand to SBA = 18
# sim_date = "2023-07-11" # mixed-aged_pine
#sim_date = "2023-07-26"  # publication
sim_date = "2023-11-23_WP_long_only"  # Revision 1, wood products short and long separeted

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

# -


logs_path = LOGS_PATH.joinpath(f"{sim_date}")
logs_path.mkdir(exist_ok=True, parents=True)
logs_path

# +
notebook_datas = []

# pre-spinup simulation data
pre_spinup_data = [
    {"filename": "run_pre_spinup_WP_long_only.py", "params": [pre_spinup_date]},
]

# four scenarios simulation data including continuous-cover spinup
four_scenarios_data = [
    {
        "filename": "run_simulation_WP_long_only.py",
        "params": [
            pre_spinup_date,
            cc_spinup_species,
            str(cc_spinup_length),
            str(cc_spinup_N),
            sim_date,
            sim_name,
            str(sim_N),
            emergency_action_str,
            emergency_direction,
            emergency_stand_action_str,
        ],
    }
    for sim_name in sim_names
]
# -

notebook_datas = (
    pre_spinup_data + four_scenarios_data
)
notebook_datas

# +
# %%time

for nr, nb_data in tqdm(enumerate(notebook_datas)):
    filename = nb_data["filename"]
    params = nb_data["params"]

    print("Starting:", filename)
    print(params)
    logfile_path = logs_path.joinpath(f"run_sims_for_paper_serial_WP_long_only_{nr}.log")
    print("Logfile path:", logfile_path)
    with open(logfile_path, "w") as f:
        process = subprocess.run(
            ["python", filename] + params, stdout=f, universal_newlines=True
        )
    #        process.wait()

    print("Finished:", filename)
    print(logfile_path.absolute())
    print(process)
