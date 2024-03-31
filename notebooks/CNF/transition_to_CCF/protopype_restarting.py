# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Development notebook to re-use a spinup

# ...spinup 160 or 240 years, simulation 80 years

# %load_ext autoreload

# +
import numpy as np
import xarray as xr
import argparse

from bgc_md2.notebook_helpers import write_to_logfile
from CompartmentalSystems.discrete_model_run import DiscreteModelRun as DMR
from LAPM.discrete_linear_autonomous_pool_model import DiscreteLinearAutonomousPoolModel as DLAPM

from ACGCA import utils
from ACGCA.__init__ import DATA_PATH, Q_
from ACGCA.simulation_parameters import stand_params
from ACGCA.simulation import utils as sim_utils
from ACGCA.soil.simple_soil_c_model import SimpleSoilCModel
from ACGCA.wood_products.simple_wood_product_model import SimpleWoodProductModel
from ACGCA.management.library import species_setting_from_sim_profile

from ACGCA.productivity.stand import Stand
from ACGCA.simulation.library import (
    create_mixed_aged_sim_profile,
    load_clear_cut_sim_profiles,
    prepare_forcing
)
from ACGCA.simulation.recorded_simulation import RecordedSimulation
from ACGCA.alloc.ACGCA_marklund_tree_params import species_params

# %autoreload 2
# -

all_sims_path = DATA_PATH.joinpath("simulations")
all_sims_path.mkdir(exist_ok=True)

# ## Set up forcing and simulation length

# +
# simulation data


# start `spinup_length` years earlier so as to have the true start again at 2000
nr_copies = 1
forcing = prepare_forcing(nr_copies=nr_copies, year_offset=-nr_copies*20)

sim_cohort_name = "prototype_restart"
sim_cohort_path = all_sims_path.joinpath(sim_cohort_name)

sim_cohort_path.mkdir(exist_ok=True)
print(sim_cohort_path)

# +
management_strategy = [
    ("StandAge3", "Plant"),
    ("PCT", "T0.75"),
]

species = "pine"
N = 2_000
sim_profile =  [
    (species, 1.0, N / 10_000 / 4, management_strategy, "waiting"),
#    (species, 1.2, N / 10_000 / 4, management_strategy, "waiting"),
#    (species, 1.4, N / 10_000 / 4, management_strategy, "waiting"),
#    (species, 1.6, N / 10_000 / 4, management_strategy, "waiting"),
]

# +
sim_name = "common_spinup"

emergency_action_str, emergency_direction = "Die", "below"
emergency_q = 0.75 # remaining fraction after emergency thinning (in case it is asked for)

species_setting = species_setting_from_sim_profile(sim_profile)

logfile_path = sim_cohort_path.joinpath(sim_name + ".log")
print(f"log file: {logfile_path}")

# +
# %%time
  
stand = Stand.create_empty(stand_params)
stand.add_trees_from_setting(species_setting, custom_species_params=species_params)

print(stand)
# -

# ## Run common spinup

recorded_spinup_simulation = RecordedSimulation.from_simulation_run(
    sim_name,
    sim_profile,
#    light_model,
    "Spitters",
    forcing,
#    custom_species_params,
    species_params,
    stand,
#    final_felling,
    emergency_action_str,
    emergency_direction,
    emergency_q, # fraction to keep
    logfile_path
)

# ### Save spinup simulation

filepath = sim_cohort_path.joinpath(sim_name + ".dmp")
recorded_spinup_simulation.save_to_file(filepath)
print(filepath)

# ## Load spinup simulation and try to continue from here

recorded_spinup_simulation = RecordedSimulation.from_file(filepath)

loaded_stand = recorded_spinup_simulation.simulation.stand

# ## Set up forcing and simulation length

# +
# simulation data
sim_name = "continuation"

# start `spinup_length` years earlier so as to have the true start again at 2000
nr_copies = 1
forcing = prepare_forcing(nr_copies=nr_copies, year_offset=0)

# +
# add immediate clear cut with replanting

nr_wait = 3
actions = ["cut"] + ["wait"] * (nr_wait + 1) + ["replant"]
for tree_in_stand in loaded_stand.trees:
    # assign tree to cutting and replanting
    tree_in_stand.status_list[-1] = f"assigned to: {actions}"
    
    # set properties of newly planted tree
    tree_in_stand._new_species = tree_in_stand.species
    tree_in_stand._new_dbh = tree_in_stand.C_only_tree.tree.initial_dbh
    tree_in_stand._new_tree_age = Q_(0, "yr")
    tree_in_stand._new_N_per_m2 = tree_in_stand.base_N_per_m2

# +
final_felling = True

if final_felling:
    total_length = 2 * 20
    loaded_stand.add_final_felling(Q_(total_length, "yr"))
    
print(loaded_stand)
# -

recorded_simulation = RecordedSimulation.from_simulation_run(
    sim_name,
    sim_profile,
#    light_model,
    "Spitters",
    forcing,
#    custom_species_params,
    species_params,
    loaded_stand,
#    final_felling,
    emergency_action_str,
    emergency_direction,
    emergency_q, # fraction to keep
    logfile_path,
    recorded_spinup_simulation
)

ds = recorded_simulation.ds
ds

filepath = sim_cohort_path.joinpath(sim_name + ".nc")
print(filepath)

ds.to_netcdf(str(filepath))
print(filepath)

dmr = utils.create_dmr_from_stocks_and_fluxes(ds)





# ## Cut out last part of simulation

nr_years = sim_dict["sim_length"] + 1
ds_long = ds.copy()
ds = ds_long.sel(time=ds_long.time[-nr_years:])
ds = ds.assign({"time": np.arange(len(ds.time))})
ds

# ## Create discrete model run, load initial age data

sim_name = sim_name.replace("_long", "")
print(sim_name)

# create discrete model run from stocks and fluxes
# shorten the data time step artificially to be able to create DMR
#nr_all_pools = stand.nr_trees * stand.nr_tree_pools + stand.nr_soil_pools
dmr = utils.create_dmr_from_stocks_and_fluxes(ds, GPP_total_prepend=ds_long.GPP_total[-(nr_years+1)])

# ## Compute transit time variables, carbon sequestration, add to dataset

ds = sim_utils.compute_BTT_vars_and_CS_and_add_to_ds(ds, dmr, dmr_eq, up_to_order=2)
ds

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

# the base for the x-axis of the video, quite arbitray
base_N = sim_dict["N"] / 10_000
print("Creating stand cross section video")
filepath = sim_cohort_path.joinpath(sim_name + "_cs.mp4")
utils.create_stand_cross_section_video(ds, filepath, base_N)
print(filepath)
# +
# %%time

print("\nCreating simulation section video")
filepath = sim_cohort_path.joinpath(sim_name + "_sim.mp4")

utils.create_simulation_video(
    ds_long,
    dmr_eq,
    np.array([dmr.soil_pool_nrs[-1]]),
    filepath, 
    resolution=10,
    time_index_start=sim_dict["cc_spinup_length"]-sim_dict["sim_length"],
    clearcut_index=sim_dict["cc_spinup_length"],
    time_index_stop=len(ds_long.time)-2,
    year_shift=-sim_dict["cc_spinup_length"],
    cache_size=1_000
)
print(filepath)
# -



