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

# # Smooth patch transition at the end, one tree every 20 yr

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
from BFCPM.simulation_parameters import stand_params_library
from BFCPM.simulation import utils as sim_utils
from BFCPM.soil.dead_wood_classes.C_model import SoilCDeadWoodClasses
from BFCPM.wood_products.simple_wood_product_model.C_model import SimpleWoodProductModel
from BFCPM.management.library import species_setting_from_sim_profile
from BFCPM.management.management_strategy import (
    ManagementStrategy,
    OnStandAges, OnSBALimit, PCT,
    Cut, CutWaitAndReplant, ThinStand, Thin
)

from BFCPM.stand import Stand
from BFCPM.simulation.library import prepare_forcing

from BFCPM.simulation.recorded_simulation import RecordedSimulation
from BFCPM.trees.single_tree_params import species_params
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
    
    parser.add_argument("pre_spinup_date", type=str)
    parser.add_argument("pre_spinup_species", type=str)
    parser.add_argument("common_spinup_dmp_filepath", type=str) # clear-cut spinup

    parser.add_argument("sim_date", type=str)
#    parser.add_argument("sim_name", type=str)
    parser.add_argument("species", type=str)
    parser.add_argument("coarseness", type=int)

    parser.add_argument("delay", type=int)

    args = parser.parse_args()

    pre_spinup_date = args.pre_spinup_date
    pre_spinup_species = args.pre_spinup_species
    common_spinup_dmp_filepath = args.common_spinup_dmp_filepath
#    cc_spinup_length = args.cc_spinup_length
#    cc_spinup_N = args.cc_spinup_N
    sim_date = args.sim_date
#    sim_name = args.sim_name
    species = args.species
    coarseness = args.coarseness

    delay = args.delay
    
    print("Simulation settings from command line")
except SystemExit:
    print("Standard simulation settings")

    pre_spinup_date = "2023-10-18"
    pre_spinup_species = "pine"

    coarseness = 12
    
    # "common" means used by all simulations
#    common_spinup_dmp_filepath = f"common_spinup_pine_160_2000"
    common_spinup_dmp_filepath = f"DWC_common_spinup_clear_cut_{pre_spinup_species}_{coarseness:02d}"
#    common_spinup_dmp_filepath = f"common_spinup_pine_20_2000"

#    sim_date = "2023-05-22"
#    sim_date = "2023-06-09"
#    sim_date = "2024-02-14"
    sim_date = "2024-03-05"
    species = "pine"
    coarseness = 12
    delay = 20
    
sim_dict = {
    "pre_spinup_date": pre_spinup_date,
    "pre_spinup_species": pre_spinup_species,

    "common_spinup_dmp_filepath": common_spinup_dmp_filepath,
#    "common_spinup_length": common_spinup_length,
#    "common_spinup_N": common_spinup_N,

    "sim_date": sim_date,
    "species": species,
    "coarseness": coarseness,
    
    "sim_length": 2 * 8 * 20,
    "N_BAU": 2_000,
    "N_cc": 1_500,

    "delay": delay
}

sim_name = f"patch_transition_at_end_smooth_{sim_dict['species']}_{sim_dict['sim_length']}_{coarseness:02d}_{delay:02d}"
sim_dict["sim_name"] = sim_name


print(sim_dict)
# -

# ## Set up forcing and simulation length

# +
light_model = "Zhao" # Zhao or Spitters

# start `spinup_length` years earlier so as to have the true start again at 2000

# WARNING: change that back! Also in the internal function!

nr_copies = sim_dict["sim_length"] // 20
#nr_copies = sim_dict["sim_length"] // 10
#nr_copies = sim_dict["sim_length"] // 1

## WARNING: TODO: XXX, kaum wird so bei 218 herum ein Baum gepflanzt,
# geht die C-Ausnahme komplett in den Arsch, bei allen Bäumen!!!
# Da passt was nicht, aber so richtig nicht.

forcing = prepare_forcing(
    nr_copies=nr_copies,
    year_offset=0,
    coarseness=sim_dict["coarseness"]
)

sim_cohort_name = ""
sim_cohort_path = SIMULATIONS_PATH.joinpath(sim_cohort_name)
sim_cohort_path = sim_cohort_path.joinpath(f"{sim_dict['sim_date']}")

sim_cohort_path.mkdir(exist_ok=True)
print(sim_cohort_path)
# -

# ## Load spinup simulation and try to continue from here

# +
# why is this here?
#pint.set_application_registry(ureg)
# -

filepath = sim_cohort_path.joinpath(sim_dict["common_spinup_dmp_filepath"] + ".dmp")
recorded_spinup_simulation = RecordedSimulation.from_file(filepath)
spinup_simulation = recorded_spinup_simulation.simulation

stand = deepcopy(spinup_simulation.stand)

common_spinup_length = stand.age.to("year").magnitude
total_length = common_spinup_length + sim_dict["sim_length"]
common_spinup_length, total_length

# ## Add initial clear cut and continue with BAU

delay = sim_dict["delay"]

# +
# add immediate clear cut with replanting

nr_wait = 3

if delay > 0:
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
# then add clear cut every 80 years with replanting

if delay > 0:
    for tree_in_stand in stand.trees:
        ms = list()
        for clear_cut_time in [80, 160, 240]:
            trigger = OnStandAges(stand_ages=[Q_(common_spinup_length+clear_cut_time-1, "yr")])
            action = CutWaitAndReplant(nr_waiting=3)
            ms_item = (trigger, action)
            ms.append(ms_item)
    
        tree_in_stand.management_strategy = ManagementStrategy(ms)
        
    print(stand)
# -

# ## Add PCT and SBA-dependent thinning fom 25 to 18

if delay > 0:
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

if delay > 0:
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
    
    print(stand)
# -

# ## Add timed cuts with 20 years between them
#
# - add ``delay`` in a smart way

# wait until first SBA-dependent thinning (after 48 years)
# new trees do not survive, so begin process earlier
if delay == 0:
    shift = 0
elif delay <= 50:
    shift = 50
else:
    shift = delay

# +
for k, tree_in_stand in enumerate(reversed(stand.trees)):
    # add new (trigger, action pairs), don't remove old ones
    ms = tree_in_stand.management_strategy.trigger_action_list

#    if k == 1:
#        # remove the two tallest trees to make enough space for the small tree
#        clear_cut_time = shift + (k - 1) * 20
#    elif k == 2:
#        clear_cut_time = shift + (k - 2) * 20
#    else:
#        clear_cut_time = shift + k * 20

    clear_cut_time = shift + k * 20
    
    trigger = OnStandAges(stand_ages=[Q_(common_spinup_length+clear_cut_time, "yr")])
    action = Cut()
    ms_item = (trigger, action)
    ms.append(ms_item)
    
    tree_in_stand.management_strategy = ManagementStrategy(ms)
        
print(stand)
# -

# ## Plant new trees every 20 years

# +
management_strategies = [
    [
        (f"StandAge{shift+common_spinup_length+k*20+3}", "Plant"),
        (f"StandAge{shift+common_spinup_length+k*20+80-1}", "CutWait3AndReplant"),   
        (f"StandAge{shift+common_spinup_length+k*20+160-1}", "CutWait3AndReplant"),   
        (f"StandAge{shift+common_spinup_length+k*20+240-1}", "CutWait3AndReplant"),   
    ] for k in range(4)
]

species = sim_dict["species"]
N = sim_dict["N_cc"]
if species in ["pine", "spruce"]:
    sim_profile =  [
        (species, 1.0, N / 10_000 / 4, management_strategies[k], "waiting")
        for k in range(4)
    ]
else:
    raise ValueError(f"Unknown patch species: {species}")
# -


species_setting = species_setting_from_sim_profile(sim_profile)
species_setting

# +
# %%time

stand.add_trees_from_setting(
    species_setting,
    custom_species_params=custom_species_params,
    custom_global_tree_params=custom_global_tree_params
)

for tree in stand.trees[-4:]:
    tree.name = "cc_" + tree.name
    
print(stand)

# +
# update additional variables of new trees

d = recorded_spinup_simulation.additional_vars
stand_age = stand.age.to("yr").magnitude

for tree in stand.trees[-4:]:
    for variable_name in d.keys():
        variable = d[variable_name]
        if isinstance(variable, dict):
            empty_values = [np.nan] * stand_age
            d[variable_name][tree.name] = empty_values

# actually has no effect        
recorded_spinup_simulation.additional_vars = d
# -

# ### Add final felling if asked for

# +
final_felling = True

if final_felling:
    penultimate_stand_age = total_length - 1
    for tree_in_stand in stand.trees:
        trigger = OnStandAges(stand_ages=[Q_(penultimate_stand_age-1, "yr")])
        action = Cut()
        ms_item = (trigger, action)
        ms = tree_in_stand.management_strategy.trigger_action_list
        ms.append(ms_item)
        tree_in_stand.management_strategy = ManagementStrategy(ms)

print(stand)
# -

sim_profile = sim_dict["sim_name"]

# +
sim_name = sim_dict["sim_name"]

#emergency_action_str, emergency_direction, emergency_stand_action_str = "Die", "below", ""
emergency_action_str, emergency_direction, emergency_stand_action_str = "Die", "below", "ThinStandToSBA18"
#emergency_action_str, emergency_direction = "Die", "below"
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
    forcing,
    custom_species_params,
    stand,
    emergency_action_str,
    emergency_direction,
    emergency_q, # fraction to keep
    emergency_stand_action_str, # in case of emergency, also do this
    recorded_spinup_simulation,
)

# ## Save recorded simulation

filepath = sim_cohort_path.joinpath(sim_name + ".dmp")
recorded_simulation.save_to_file(filepath)
print(filepath)

# ## Load recorded simulation

filepath = sim_cohort_path.joinpath(sim_name + ".dmp")
recorded_simulation = RecordedSimulation.from_file(filepath)

ds = recorded_simulation.ds
ds

# ## Save datasets (long and simulation only)

filepath = sim_cohort_path.joinpath(sim_name + "_no_dmr" + ".nc")
ds.to_netcdf(str(filepath))
print(filepath)

# +
start_year = common_spinup_length

ds_sim = ds.sel(time=ds.time[start_year:])
ds_sim = ds_sim.assign({"time": np.arange(len(ds_sim.time))})
ds_sim
# -

filepath = sim_cohort_path.joinpath(sim_name + "_no_dmr_sim.nc")
ds_sim.to_netcdf(str(filepath))
print(filepath)

# stop parallel computation here, I don't need the rest currently and it takes too long
import sys
sys.exit(0)







# ## Compute transit time variables, carbon sequestration, add to dataset

# +
# load fake equilibrium dmr
pre_spinup_species = sim_dict["pre_spinup_species"]

spinups_path = PRE_SPINUPS_PATH.joinpath(sim_dict["pre_spinup_date"])
#pre_spinup_name = f"DWC_{light_model}_{pre_spinup_species}_{coarseness:02d}_2nd_round"
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

# ## Save long dataset, discrete model run, and spinup (fake equilibrium) model run

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

# ## Save simulation discrete model run

filepath = sim_cohort_path.joinpath(sim_name + "_sim.dmr")
dmr_sim.save_to_file(filepath)
print("Simulation discrete model run")
print(filepath)
print()
# ## Save augmented simulation dataset

filepath = sim_cohort_path.joinpath(sim_name + "_sim.nc")
ds_sim.to_netcdf(str(filepath))
print(filepath)






