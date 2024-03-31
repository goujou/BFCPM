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

# # Run single MeanTree simulation for just a very long time and see what happens

# %load_ext autoreload

# +
import numpy as np
import xarray as xr
import argparse

from bgc_md2.notebook_helpers import write_to_logfile
from CompartmentalSystems.discrete_model_run import DiscreteModelRun as DMR
from LAPM.discrete_linear_autonomous_pool_model import DiscreteLinearAutonomousPoolModel as DLAPM

from BFCPM import utils
from BFCPM.__init__ import PRE_SPINUPS_PATH, SIMULATIONS_PATH, Q_
from BFCPM.simulation_parameters import stand_params_library
from BFCPM.simulation import utils as sim_utils
from BFCPM.soil.simple_soil_model.C_model import SimpleSoilCModel
from BFCPM.soil.dead_wood_classes.C_model import SoilCDeadWoodClasses
from BFCPM.wood_products.simple_wood_product_model.C_model import SimpleWoodProductModel
from BFCPM.management.library import species_setting_from_sim_profile

from BFCPM.stand import Stand
from BFCPM.simulation.library import prepare_forcing
from BFCPM.simulation.recorded_simulation import RecordedSimulation
from BFCPM.trees.single_tree_params import species_params
from BFCPM.params import global_tree_params

# %autoreload 2

# +
# #%tb

custom_species_params = species_params.copy()
custom_global_tree_params = global_tree_params.copy()

try:
    parser = argparse.ArgumentParser()
    parser.add_argument("pre_spinup_date", type=str)
    parser.add_argument("pre_spinup_species", type=str)
    
    parser.add_argument("sim_date", type=str)
    parser.add_argument("species", type=str)
    parser.add_argument("sim_length", type=int)
    parser.add_argument("trees_per_ha", type=int)
    parser.add_argument("coarseness", type=int)

    args = parser.parse_args()

    pre_spinup_date = args.pre_spinup_date
    pre_spinup_species = args.pre_spinup_species

    sim_date = args.sim_date
    species = args.species
    sim_length = args.sim_length
    N = args.trees_per_ha
    coarseness = args.coarseness
    
    print("Simulation settings from command line")
except SystemExit:
    print("Standard simulation settings")

    pre_spinup_date = "2023-10-18"
    pre_spinup_species = "pine"

#    sim_date = "2023-10-19"
    sim_date = "2024-02-14"
    species = "pine"
#    species = "spruce"
#    sim_length = 16 * 20
    sim_length = 6 * 20
#    N = 2_000
#    N = 1
    N = 1_000
#    coarseness = 1
    coarseness = 12   
    
sim_dict = {
    "pre_spinup_date": pre_spinup_date,
    "pre_spinup_species": pre_spinup_species,
    
    "sim_date": sim_date,
    "species": species,
    "sim_length": sim_length,
    "N": N,
    "coarseness": coarseness
}

sim_name = f"single_{sim_dict['species']}_{sim_dict['sim_length']}_{int(sim_dict['N'])}_{coarseness:02d}_q75"
sim_dict["sim_name"] = sim_name

print(sim_dict)
# -

# ## Set up forcing and simulation length

# +
# simulation data
# start 0 years earlier so as to have the true start again at 2000
nr_copies = sim_dict["sim_length"] // 20
forcing = prepare_forcing(
    nr_copies=nr_copies,
    year_offset=0,
    coarseness=sim_dict["coarseness"]
)

sim_cohort_name = ""
sim_cohort_path = SIMULATIONS_PATH.joinpath(sim_cohort_name)
sim_cohort_path = sim_cohort_path.joinpath(f"{sim_dict['sim_date']}")

sim_cohort_path.mkdir(exist_ok=True, parents=True)
print(sim_cohort_path)
# -

# ## Load spinup data: soil and wood product stocks and age structure

# +
spinups_path = PRE_SPINUPS_PATH.joinpath(sim_dict["pre_spinup_date"])

light_model = "Zhao" # Zhao or Spitters

#pre_spinup_name = f"DWC_{light_model}_{sim_dict['pre_spinup_species']}_{sim_dict['coarseness']:02d}_2nd_round"
pre_spinup_name = f"DWC_{light_model}_{sim_dict['pre_spinup_species']}_2nd_round"
dmr_path = spinups_path.joinpath(pre_spinup_name + ".dmr_eq")

# load fake equilibrium dmr
dmr_eq = DLAPM.load_from_file(dmr_path)

# initialize soil and wood product models with spinup stocks
soil_model = SoilCDeadWoodClasses(initial_stocks=Q_(dmr_eq.xss[dmr_eq.soil_pool_nrs], "gC/m^2"))
wood_product_model = SimpleWoodProductModel(initial_stocks=Q_(dmr_eq.xss[dmr_eq.wood_product_pool_nrs], "gC/m^2"))
#soil_model = SoilCDeadWoodClasses()
#wood_product_model = SimpleWoodProductModel()

stand_params = stand_params_library["default"]
stand_params["soil_model"] = soil_model
stand_params["wood_product_model"] = wood_product_model
stand_params["wood_product_interface_name"] = "no_harvesting"

dmr_path


# +
species = sim_dict["species"]

management_strategy = [
    ("StandAge3", "Plant"),
#    ("PCT", "T0.75"), # pre-commercial thinning
#    ("DBH35-80", "CutWait3AndReplant"),
#    # needs to be lower priority than any cutting, otherwise cutting might be delayed
#    ("SBA25", "ThinStandToSBA18"), # SBA dependent thinning
]


sim_profile = [
    (species, 1.0, sim_dict["N"] / 10_000, management_strategy, "waiting"),
]
#sim_profile = [
#    (species, 1.0, sim_dict["N"] / 4 / 10_000, management_strategy, "waiting"),
#    (species, 1.2, sim_dict["N"] / 4 / 10_000, management_strategy, "waiting"),
#    (species, 1.4, sim_dict["N"] / 4 / 10_000, management_strategy, "waiting"),
#    (species, 1.6, sim_dict["N"] / 4 / 10_000, management_strategy, "waiting"),
#]

#emergency_action_str, emergency_direction, emergency_stand_action_str = "Die", "below", "ThinStandToSBA18"
emergency_action_str, emergency_direction, emergency_stand_action_str = "Thin", "below", ""
#emergency_action_str, emergency_direction = "CutWait3AndReplant", "above"
emergency_q = 0.75
#emergency_q = 0.95

species_setting = species_setting_from_sim_profile(sim_profile)

logfile_path = sim_cohort_path.joinpath(sim_name + ".log")
print(f"log file: {logfile_path}")

# +
# %%time

#import warnings
#with warnings.catch_warnings():
#    warnings.simplefilter("error")
    
stand = Stand.create_empty(stand_params)
stand.add_trees_from_setting(
    species_setting,
    custom_species_params=custom_species_params,
    custom_global_tree_params=custom_global_tree_params
)

print(stand)

# +
final_felling = False

if final_felling:
    total_length = sim_dict["sim_length"]
    stand.add_final_felling(Q_(total_length, "yr"))
    
print(stand)
# -

# ## Run simulation

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
    emergency_q, # fraction to keep in case of thinning
    emergency_stand_action_str # what to do with remaining trees in case of removal
)

# ### Save common spinup dataset and simulation

filepath = sim_cohort_path.joinpath(sim_name + ".dmp")
recorded_simulation.save_to_file(filepath)
print(filepath)

ds = recorded_simulation.ds
filepath = sim_cohort_path.joinpath(sim_name + ".ds")
ds.to_netcdf(str(filepath))
print(filepath)

#ds = xr.open_dataset(str(filepath))
ds

# ## Create discrete model run, load initial age data

# create discrete model run from stocks and fluxes
# shorten the data time step artificially to be able to create DMR
#nr_all_pools = stand.nr_trees * stand.nr_tree_pools + stand.nr_soil_pools
dmr = utils.create_dmr_from_stocks_and_fluxes(ds) #, GPP_total_prepend=ds_long.GPP_total[-(nr_years+1)])

# ## Compute transit time variables, carbon sequestration, add to dataset

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
# the base for the x-axis of the video, quite arbitray
base_N = sim_dict["N"] / 10_000
print("Creating stand cross section videos")

for relative in [False]:#, True]:
    print(f"relative = {relative}")
    filepath = sim_cohort_path.joinpath(sim_name + f"_cs{'_rel' if relative else ''}.mp4")
    utils.create_stand_cross_section_video(ds, filepath, base_N, relative=relative)
    print(filepath)
# -
filepath = sim_cohort_path.joinpath(sim_name + ".nc")
ds = xr.open_dataset(filepath)
filepath

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
    clearcut_index=None,
    time_index_stop=len(ds.time)-2,
    year_shift=0,
    cache_size=1_000
)
print(filepath)
# -

