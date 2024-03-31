# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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
from BFCPM.__init__ import PRE_SPINUPS_PATH, SIMULATIONS_PATH, Q_
from ACGCA.simulation_parameters import stand_params
from ACGCA.simulation import utils as sim_utils
from ACGCA.soil.simple_soil_c_model import SimpleSoilCModel
from ACGCA.wood_products.simple_wood_product_model import SimpleWoodProductModel
from ACGCA.management.library import species_setting_from_sim_profile
from ACGCA.management.management_strategy import (
    ManagementStrategy,
    OnStandAges, OnSBALimit, PCT,
    Cut, CutWaitAndReplant, ThinStand, Thin
)

from ACGCA.productivity.stand import Stand
from ACGCA.simulation.library import (
    create_mixed_aged_sim_profile,
    load_clear_cut_sim_profiles,
    prepare_forcing
)
from ACGCA.simulation.recorded_simulation import RecordedSimulation
from ACGCA.alloc.ACGCA_marklund_tree_params import species_params
from BFCPM.params import global_tree_params

# %autoreload 2
# -

# ### Custom species parameters?

# +
# #%tb

custom_species_params = species_params.copy()
custom_global_tree_params = global_tree_params.copy()

try:
    parser = argparse.ArgumentParser()

    parser.add_argument("common_spinup_dmp_filepath", type=str) # continuous-cover (age-distributed) spinup
#    parser.add_argument("cc_spinup_length", type=int)
#    parser.add_argument("cc_spinup_N", type=int)
 
    parser.add_argument("sim_date", type=str)
    parser.add_argument("sim_name", type=str)
    parser.add_argument("coarseness", type=int)

    args = parser.parse_args()

    common_spinup_dmp_filepath = args.common_spinup_dmp_filepath
#    cc_spinup_length = args.cc_spinup_length
#    cc_spinup_N = args.cc_spinup_N
    sim_date = args.sim_date
    sim_name = args.sim_name
    coarseness = args.coarseness

    print("Simulation settings from command line")
except SystemExit:
    print("Standard simulation settings")

    # "common" means used by all simulations
    common_spinup_dmp_filepath = f"common_spinup_pine_160_2000"
#    common_spinup_dmp_filepath = f"common_spinup_pine_20_2000"

#    common_spinup_species = "pine"
#    common_spinup_length = 1 * 20
#    common_spinup_N = 2_000

#    sim_date = "2023-05-22"
    sim_date = "2023-06-09"
#    sim_name = "mixed-aged_pine_long"
#    sim_name = "even-aged_pine_long"
#    sim_name = "even-aged_spruce_long"
#    sim_name = "even-aged_mixed_long"
    sim_name = "BAU"
    
sim_dict = {
    "common_spinup_dmp_filepath": common_spinup_dmp_filepath,
#    "common_spinup_length": common_spinup_length,
#    "common_spinup_N": common_spinup_N,

    "sim_date": sim_date,
    "sim_name": sim_name,

    "sim_length": 8 * 20,
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
sim_cohort_path = SIMULATIONS_PATH.joinpath(sim_cohort_name)
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
    for clear_cut_time in [80]:
        trigger = OnStandAges(stand_ages=[Q_(common_spinup_length+clear_cut_time-1, "yr")])
        action = CutWaitAndReplant(nr_waiting=3)
        ms_item = (trigger, action)
        ms.append(ms_item)
    
    tree_in_stand.management_strategy = ManagementStrategy(ms)
        
print(stand)
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

print(stand)

# +
# add new SBA-dependent thinning with updated blocks
for tree_in_stand in stand.trees:
    trigger = OnSBALimit(
        sba_lim=Q_("25 m^2/ha"),
        blocked_stand_ages=[
            (Q_(common_spinup_length+70, "yr"), Q_(common_spinup_length+80, "yr")),
            (Q_(common_spinup_length+150, "yr"), Q_(common_spinup_length+160, "yr"))
        ]
    )
    action = ThinStand(f=lambda dth: 18.0)
    ms_item = (trigger, action)
    ms = tree_in_stand.management_strategy.trigger_action_list
    ms.append(ms_item)
    tree_in_stand.management_strategy = ManagementStrategy(ms)
    
print(stand)
# -

# ### Add final felling if asked for

# +
final_felling = True

if final_felling:
    stand.add_final_felling(Q_(total_length, "yr"))
    
print(stand)
# -

sim_profile = "BAU" # dummy, currently used for logging only

# +
sim_name = sim_dict["sim_name"]

emergency_action_str, emergency_direction = "Die", "below"
#emergency_action_str, emergency_direction = "Thin", "below"
#emergency_action_str, emergency_direction = "CutWait3AndReplant", "above"

emergency_q = 0.75 # remaining fraction after emergency thinning (in case it is asked for)

logfile_path = sim_cohort_path.joinpath(sim_name + ".log")
print(f"log file: {logfile_path}")
# -

recorded_simulation = RecordedSimulation.from_simulation_run(
    sim_name,
    sim_profile,
    light_model,
#    "Spitters",
    forcing,
#    custom_species_params,
    species_params,
    stand,
#    final_felling,
    emergency_action_str,
    emergency_direction,
    emergency_q, # fraction to keep
    logfile_path,
    recorded_spinup_simulation
)

# ## Save recorded simulation and all the objects

filepath = sim_cohort_path.joinpath(sim_name + ".dmp")
recorded_simulation.save_to_file(filepath)
print(filepath)

ds = recorded_simulation.ds
filepath = sim_cohort_path.joinpath(sim_name + ".nc")
ds.to_netcdf(str(filepath))
print(filepath)

recorded_simulation.ds

# ## Create DMR

dmr = utils.create_dmr_from_stocks_and_fluxes(ds)



filepath = sim_cohort_path.joinpath(sim_name + ".nc")
ds_long = xr.open_dataset(str(filepath))
ds_long

(ds_long.stocks.sum(dim=["entity", "pool"])*1e-03).plot()



# ## Cut out last part of simulation

start_year = 130
nr_years = 81
ds = ds_long.sel(time=ds_long.time[start_year:start_year+nr_years])
ds = ds_long.sel(time=ds_long.time[-nr_years:])
ds = ds.assign({"time": np.arange(len(ds.time))})
ds

# ## Create discrete model run, load initial age data

sim_name = sim_name.replace("_long", "")
print(sim_name)

# create discrete model run from stocks and fluxes
# shorten the data time step artificially to be able to create DMR
#nr_all_pools = stand.nr_trees * stand.nr_tree_pools + stand.nr_soil_pools
dmr = utils.create_dmr_from_stocks_and_fluxes(ds, GPP_total_prepend=ds_long.GPP_total[start_year-1])

dmr.initialize_state_transition_operator_matrix_cache(10_000)

# ## Compute transit time variables, carbon sequestration, add to dataset

len(simulation.GPP_totals)

ds_long.GPP_total.data


