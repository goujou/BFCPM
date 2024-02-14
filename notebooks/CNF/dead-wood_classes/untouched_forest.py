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

# # Untouched forest for 160/320 yr

# %load_ext autoreload

# +
import numpy as np
import xarray as xr
import argparse

from copy import deepcopy

from bgc_md2.notebook_helpers import write_to_logfile
from CompartmentalSystems.discrete_model_run import DiscreteModelRun as DMR
from LAPM.discrete_linear_autonomous_pool_model import DiscreteLinearAutonomousPoolModel as DLAPM

from BFCPM import utils
from BFCPM.__init__ import PRE_SPINUPS_PATH, SIMULATIONS_PATH, Q_
from BFCPM.simulation import utils as sim_utils
from BFCPM.management.library import species_setting_from_sim_profile
from BFCPM.management.management_strategy import (
    ManagementStrategy,
    OnStandAges, OnSBALimit, PCT,
    Cut, CutWaitAndReplant, ThinStand, Thin
)

from BFCPM.stand import Stand
from BFCPM.simulation.library import prepare_forcing
from BFCPM.prepare_stand import load_wood_product_interface
from BFCPM.trees.single_tree_allocation import SingleTree

from BFCPM.simulation.recorded_simulation import RecordedSimulation
from BFCPM.trees.single_tree_params import species_params

# %autoreload 2
# -

# ### Custom species parameters?

# +
# #%tb

try:
    parser = argparse.ArgumentParser()

    parser.add_argument("pre_spinup_date", type=str)
    parser.add_argument("pre_spinup_species", type=str)
    
    parser.add_argument("common_spinup_dmp_filepath", type=str) # continuous-cover (age-distributed) spinup
 
    parser.add_argument("sim_date", type=str)
    parser.add_argument("sim_name", type=str)
    parser.add_argument("coarseness", type=int)

    args = parser.parse_args()

    pre_spinup_date = args.pre_spinup_date
    pre_spinup_species = args.pre_spinup_species
    
    common_spinup_dmp_filepath = args.common_spinup_dmp_filepath

    sim_date = args.sim_date
    sim_name = args.sim_name
    coarseness = args.coarseness
    
    print("Simulation settings from command line")
except SystemExit:
    print("Standard simulation settings")

    pre_spinup_date = "2023-10-18"
    pre_spinup_species = "pine"
    
    # "common" means used by all simulations
#    coarseness = 1
    coarseness = 12

    common_spinup_dmp_filepath = f"DWC_common_spinup_clear_cut_{pre_spinup_species}_{coarseness:02d}"

    sim_date = "2023-10-19"
    sim_name = f"DWC_untouched_forest_320_{pre_spinup_species}_{coarseness:02d}"
    
sim_dict = {
    "pre_spinup_date": pre_spinup_date,
    "pre_spinup_species": pre_spinup_species,
    
    "common_spinup_dmp_filepath": common_spinup_dmp_filepath,

    "sim_date": sim_date,
    "sim_name": sim_name,
    "coarseness": coarseness,

    "sim_length": 8 * 20 * 2,
    "N": 2_000
}

print(sim_dict)
# -

# ## Load spinup simulation and try to continue from here

# +
sim_cohort_name = ""
sim_cohort_path = SIMULATIONS_PATH.joinpath(sim_cohort_name)
sim_cohort_path = sim_cohort_path.joinpath(f"{sim_dict['sim_date']}")

sim_cohort_path.mkdir(exist_ok=True)
print(sim_cohort_path)
# -

filepath = sim_cohort_path.joinpath(sim_dict["common_spinup_dmp_filepath"] + ".dmp")
recorded_spinup_simulation = RecordedSimulation.from_file(filepath)
spinup_simulation = recorded_spinup_simulation.simulation

stand = deepcopy(spinup_simulation.stand)

common_spinup_length = stand.age.to("year").magnitude
total_length = common_spinup_length + sim_dict["sim_length"]
total_length

sim_name = sim_dict["sim_name"]
sim_name

wood_product_interface = load_wood_product_interface("no_harvesting", SingleTree, stand.soil_model, stand.wood_product_model)
stand.wood_product_interface = wood_product_interface

# +
light_model = "Zhao" # Zhao or Spitters

# start `spinup_length` years earlier so as to have the true start again at 2000
nr_copies = sim_dict["sim_length"] // 20
forcing = prepare_forcing(nr_copies=nr_copies, year_offset=0, coarseness=sim_dict["coarseness"])
# -

# ## Remove all management strategies

# +
# remove all management actions from the spinup first

for tree_in_stand in stand.trees:
    ms = list()  
    tree_in_stand.management_strategy = ManagementStrategy(ms)
        
#print(stand)
# -

# ### Add final felling if asked for

# +
#final_felling = True
final_felling = False

if final_felling:
    stand.add_final_felling(Q_(total_length, "yr"))
    
print(stand)
# -

sim_profile = "untouched_forest" # dummy, currently used for logging only

# +
emergency_action_str, emergency_direction, emergency_stand_action_str = "Die", "below", ""
#emergency_action_str, emergency_direction = "Thin", "below"
#emergency_action_str, emergency_direction = "CutWait3AndReplant", "above"

emergency_q = 0.95 # remaining fraction after emergency thinning (in case it is asked for)

logfile_path = sim_cohort_path.joinpath(sim_name + ".log")
print(f"log file: {logfile_path}")
# -

recorded_simulation = RecordedSimulation.from_simulation_run(
    sim_name,
    logfile_path,
    sim_profile,
    light_model,
#    "Spitters",
    forcing,
#    custom_species_params,
    species_params,
    stand,
    emergency_action_str,
    emergency_direction,
    emergency_q, # fraction to keep
    emergency_stand_action_str,
    recorded_spinup_simulation,
)

# ## Save recorded simulation and all the objects

filepath = sim_cohort_path.joinpath(sim_name + ".dmp")
recorded_simulation.save_to_file(filepath)
print(filepath)

# ## Load recorded simulation and all the objects

filepath = sim_cohort_path.joinpath(sim_name + ".dmp")
recorded_simulation = RecordedSimulation.from_file(filepath)

ds = recorded_simulation.ds
ds

# ## Compute transit time variables, carbon sequestration, add to dataset

# +
# load fake equilibrium dmr
pre_spinup_species = sim_dict["pre_spinup_species"]
coarseness = sim_dict["coarseness"]

spinups_path = PRE_SPINUPS_PATH.joinpath(sim_dict["pre_spinup_date"])
pre_spinup_name = f"DWC_{light_model}_{pre_spinup_species}_{coarseness:02d}_2nd_round"
dmr_path = spinups_path.joinpath(pre_spinup_name + ".dmr_eq")
dmr_eq = DLAPM.load_from_file(dmr_path)
# -

cache_size = 30_000
verbose = True

# +
# %%time

dmr = utils.create_dmr_from_stocks_and_fluxes(ds)

ds = sim_utils.compute_BTT_vars_and_add_to_ds(ds, dmr, dmr_eq, up_to_order=2, cache_size=cache_size, verbose=verbose)
ds = sim_utils.compute_C_balance_and_CS_and_add_to_ds(ds, dmr, cache_size=cache_size, verbose=verbose)
ds
# -

# ## Save simulation dataset, discrete model run, and spinup (fake equilibrium) model run

# +
filepath = sim_cohort_path.joinpath(sim_name + ".nc")
ds.to_netcdf(str(filepath))
print("Simulation dataset")
print(filepath)
print()

filepath = sim_cohort_path.joinpath(sim_name + ".dmr")
dmr.save_to_file(filepath)
print("Simulation discrete model run")
print(filepath)
print()

filepath = sim_cohort_path.joinpath(sim_name + ".dmr_eq")
dmr_eq.save_to_file(filepath)
print("Spinup discrete model run in equilibrium")
print(filepath)
print()
# -
# ### Make stand cross-section and simulation videos

# +
# %%time

# the base for the x-axis of the video, quite arbitray
base_N = sim_dict["N"] / 10_000
print("Creating stand cross section video")
filepath = sim_cohort_path.joinpath(sim_name + "_cs.mp4")
utils.create_stand_cross_section_video(ds, filepath, base_N)
print(filepath)
# +
# %%time

print("\nCreating simulation video")
filepath = sim_cohort_path.joinpath(sim_name + "_sim.mp4")

utils.create_simulation_video(
    ds,
    dmr_eq,
    np.array([dmr.soil_pool_nrs[-1]]),
    filepath, 
    resolution=5,
    time_index_start=0,
    clearcut_index=common_spinup_length,
    time_index_stop=len(ds.time)-2,
    year_shift=-common_spinup_length,
    cache_size=1_000
)
print(filepath)
# -
# ## Compute C balance and CS for simulation only (no spinup)

# +
start_year = common_spinup_length

ds_sim = ds.sel(time=ds.time[start_year:])
ds_sim = ds_sim.assign({"time": np.arange(len(ds_sim.time))})
ds_sim

# +
# %%time

dmr_sim = utils.create_dmr_from_stocks_and_fluxes(ds_sim, GPP_total_prepend=ds.GPP_total[start_year-1])
ds_sim = sim_utils.compute_C_balance_and_CS_and_add_to_ds(ds_sim, dmr_sim, cache_size=cache_size, verbose=verbose)

ds_sim
# -

# ## Save simulation dataset and discrete model run

filepath = sim_cohort_path.joinpath(sim_name + "_sim.nc")
ds_sim.to_netcdf(str(filepath))
print("Simulation dataset")
print(filepath)
print()
filepath = sim_cohort_path.joinpath(sim_name + "_sim.dmr")
dmr_sim.save_to_file(filepath)
print("Simulation discrete model run")
print(filepath)
print()


