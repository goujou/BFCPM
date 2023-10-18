# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# # Just continue clear-cutting every 80 years

# %load_ext autoreload

# +
import numpy as np
import xarray as xr
import argparse

from copy import deepcopy

from bgc_md2.notebook_helpers import write_to_logfile
from CompartmentalSystems.discrete_model_run import DiscreteModelRun as DMR
from LAPM.discrete_linear_autonomous_pool_model import DiscreteLinearAutonomousPoolModel as DLAPM

from ACGCA import utils
from ACGCA.__init__ import DATA_PATH, Q_
from ACGCA.simulation import utils as sim_utils
from ACGCA.management.library import species_setting_from_sim_profile
from ACGCA.management.management_strategy import (
    ManagementStrategy,
    OnStandAges, OnSBALimit, PCT,
    Cut, CutWaitAndReplant, ThinStand, Thin
)

from ACGCA.productivity.stand import Stand
from ACGCA.simulation.library import prepare_forcing

from ACGCA.simulation.recorded_simulation import RecordedSimulation
from ACGCA.alloc.ACGCA_marklund_tree_params import species_params

# %autoreload 2
# -

all_sims_path = DATA_PATH.joinpath("simulations")
all_sims_path.mkdir(exist_ok=True)

# ### Custom species parameters?

# +
# #%tb

try:
    parser = argparse.ArgumentParser()

    parser.add_argument("pre_spinup_date", type=str)
    parser.add_argument("common_spinup_dmp_filepath", type=str) # continuous-cover (age-distributed) spinup
#    parser.add_argument("cc_spinup_length", type=int)
#    parser.add_argument("cc_spinup_N", type=int)
 
    parser.add_argument("sim_date", type=str)
    parser.add_argument("sim_name", type=str)

    args = parser.parse_args()

    pre_spinup_date = args.pre_spinup_date
    common_spinup_dmp_filepath = args.common_spinup_dmp_filepath
#    cc_spinup_length = args.cc_spinup_length
#    cc_spinup_N = args.cc_spinup_N
    sim_date = args.sim_date
    sim_name = args.sim_name
    print("Simulation settings from command line")
except SystemExit:
    print("Standard simulation settings")

    pre_spinup_date = "2023-06-22"
    
    # "common" means used by all simulations
    common_spinup_dmp_filepath = f"DWC_common_spinup_pine_clear_cut"

#    sim_date = "2023-06-23"
    sim_date = "2023-06-29"
    sim_name = "DWC_BAU_320"
    
sim_dict = {
    "pre_spinup_date": pre_spinup_date,
    
    "common_spinup_dmp_filepath": common_spinup_dmp_filepath,

    "sim_date": sim_date,
    "sim_name": sim_name,

    "sim_length": 8 * 20 * 2,
    "N": 2_000
}

print(sim_dict)
# -

# ## Set up forcing and simulation length

# +
light_model = "Zhao" # Zhao or Spitters

# start `spinup_length` years earlier so as to have the true start again at 2000
nr_copies = sim_dict["sim_length"] // 20
forcing = prepare_forcing(nr_copies=nr_copies, year_offset=0)

sim_cohort_name = ""
sim_cohort_path = all_sims_path.joinpath(sim_cohort_name)
sim_cohort_path = sim_cohort_path.joinpath(f"{sim_dict['sim_date']}")

sim_cohort_path.mkdir(exist_ok=True)
print(sim_cohort_path)
# -

# ## Load spinup simulation and try to continue from here

filepath = sim_cohort_path.joinpath(sim_dict["common_spinup_dmp_filepath"] + ".dmp")
recorded_spinup_simulation = RecordedSimulation.from_file(filepath)
spinup_simulation = recorded_spinup_simulation.simulation

stand = deepcopy(spinup_simulation.stand)

common_spinup_length = stand.age.to("year").magnitude
total_length = common_spinup_length + sim_dict["sim_length"]
total_length

sim_name = sim_dict["sim_name"]
sim_name

# ## Add clear cuts after 0, 79 years of simulation

# +
# add immediate clear cut with replanting

nr_wait = 3
actions = ["cut"] + ["wait"] * (nr_wait + 1) + ["replant"]
for tree_in_stand in stand.trees:
    # assign tree to cutting and replanting
    tree_in_stand.status_list[-1] = f"assigned to: {actions}"
    
    # set properties of newly planted tree
    tree_in_stand._new_species = tree_in_stand.species
    tree_in_stand._new_dbh = tree_in_stand.C_only_tree.tree.initial_dbh
    tree_in_stand._new_tree_age = Q_(0, "yr")
    tree_in_stand._new_N_per_m2 = tree_in_stand.base_N_per_m2

# +
# remove all management actions from the spinup first
# then add clear cut after 80 years with replanting

for tree_in_stand in stand.trees:
    ms = list()
    for clear_cut_time in [80, 160, 240]:
        trigger = OnStandAges(stand_ages=[Q_(common_spinup_length+clear_cut_time-1, "yr")])
        action = CutWaitAndReplant(nr_waiting=3)
        ms_item = (trigger, action)
        ms.append(ms_item)
    
    tree_in_stand.management_strategy = ManagementStrategy(ms)
        
#print(stand)
# -

# ## Add PCT and SBA-dependent thinning fom 25 to 18

# +
for tree_in_stand in stand.trees:
    ms = tree_in_stand.management_strategy.trigger_action_list
    trigger = PCT(mth_lim=Q_("3.0 m"))
    action = Thin(q=1-0.75)
    ms_item = (trigger, action)
    ms.append(ms_item)
    
    tree_in_stand.management_strategy = ManagementStrategy(ms)

#print(stand)

# +
# add new SBA-dependent thinning with updated blocks
for tree_in_stand in stand.trees:
    trigger = OnSBALimit(
        sba_lim=Q_("25 m^2/ha"),
        blocked_stand_ages=[
            (Q_(common_spinup_length+70, "yr"), Q_(common_spinup_length+80, "yr")),
            (Q_(common_spinup_length+150, "yr"), Q_(common_spinup_length+160, "yr")),
            (Q_(common_spinup_length+230, "yr"), Q_(common_spinup_length+240, "yr")),
            (Q_(common_spinup_length+310, "yr"), Q_(common_spinup_length+320, "yr")),
        ]
    )
    action = ThinStand(f=lambda dth: 18.0)
    ms_item = (trigger, action)
    ms = tree_in_stand.management_strategy.trigger_action_list
    ms.append(ms_item)
    tree_in_stand.management_strategy = ManagementStrategy(ms)
    
#print(stand)
# -

# ### Add final felling if asked for

# +
final_felling = True

if final_felling:
    stand.add_final_felling(Q_(total_length, "yr"))
    
print(stand)
# -

sim_profile = sim_name # dummy, currently used for logging only

# +
emergency_action_str, emergency_direction, emergency_stand_action_str = "Die", "below", "ThinStandToSBA18"
#emergency_action_str, emergency_direction = "Thin", "below"
#emergency_action_str, emergency_direction = "CutWait3AndReplant", "above"

emergency_q = 0.75 # remaining fraction after emergency thinning (in case it is asked for)

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
pre_spinup_species = "pine"

spinups_path = DATA_PATH.joinpath("pre_spinups").joinpath(sim_dict["pre_spinup_date"])
pre_spinup_name = f"DWC_{light_model}_{pre_spinup_species}_2nd_round"
dmr_path = spinups_path.joinpath(pre_spinup_name + ".dmr_eq")
dmr_eq = DLAPM.load_from_file(dmr_path)
# -

cache_size = 20_000
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
# -

# create discrete model run from stocks and fluxes for simulation only
dmr_sim = utils.create_dmr_from_stocks_and_fluxes(ds_sim, GPP_total_prepend=ds.GPP_total[start_year-1])

# +
# %%time

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

