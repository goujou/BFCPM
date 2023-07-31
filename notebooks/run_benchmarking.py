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

# # Run single pine simulation for 80 years for allometry benchmarking
#
# ...160 years spinup

# %load_ext autoreload

import argparse

# +
import numpy as np
import xarray as xr
from BFCPM import DATA_PATH, PRE_SPINUPS_PATH, Q_, utils
from BFCPM.management.library import species_setting_from_sim_profile
from BFCPM.simulation import utils as sim_utils
from BFCPM.simulation.library import prepare_forcing
from BFCPM.simulation.recorded_simulation import RecordedSimulation
from BFCPM.simulation_parameters import stand_params_library
from BFCPM.soil.simple_soil_model.C_model import SimpleSoilCModel
from BFCPM.stand import Stand
from BFCPM.trees.single_tree_params import species_params
from BFCPM.wood_products.simple_wood_product_model.C_model import \
    SimpleWoodProductModel
from bgc_md2.notebook_helpers import write_to_logfile
from CompartmentalSystems.discrete_model_run import DiscreteModelRun as DMR
from LAPM.discrete_linear_autonomous_pool_model import \
    DiscreteLinearAutonomousPoolModel as DLAPM

# %autoreload 2
# -

all_sims_path = DATA_PATH.joinpath("simulations")
all_sims_path.mkdir(exist_ok=True)

# +
# #%tb

try:
    parser = argparse.ArgumentParser()
    parser.add_argument("pre_spinup_date", type=str)

    parser.add_argument("sim_date", type=str)
    parser.add_argument("species", type=str)

    args = parser.parse_args()

    pre_spinup_date = args.pre_spinup_date

    sim_date = args.sim_date
    species = args.species
    print("Simulation settings from command line")
except SystemExit:
    print("Standard simulation settings")

    pre_spinup_date = "2023-07-25"

    sim_date = "2023-07-26"
    species = "pine"
#    species = "spruce"

sim_dict = {
    "pre_spinup_date": pre_spinup_date,
    "sim_date": sim_date,
    "species": species,
    "sim_length": 4 * 20,
    "N": 2_000,
}

print(sim_dict)
# -

# ### Custom species parameters?

# +
# tree species parameter changes can be made here
custom_species_params = species_params.copy()

# alpha = 1.00
# custom_species_params["pine"]["alpha"]["value"] = alpha
# custom_species_params["spruce"]["alpha"]["value"] = alpha
# -

# ## Set up forcing and simulation length

# +
# simulation data
# start 0 years earlier so as to have the true start again at 2000
nr_copies = sim_dict["sim_length"] // 20
forcing = prepare_forcing(nr_copies=nr_copies, year_offset=-0)

sim_cohort_name = ""
sim_cohort_path = all_sims_path.joinpath(sim_cohort_name)
sim_cohort_path = sim_cohort_path.joinpath(f"{sim_dict['sim_date']}")
sim_cohort_path = sim_cohort_path.joinpath("benchmarking")

sim_cohort_path.mkdir(exist_ok=True)
print(sim_cohort_path)
# -

# ## Load spinup data: soil and wood product stocks and age structure

# +
spinups_path = PRE_SPINUPS_PATH.joinpath(sim_dict["pre_spinup_date"])

light_model = "Zhao"  # Zhao or Spitters
# light_model = "Spitters" # Zhao or Spitters

spinup_species = "pine"
# spinup_name = "basic"
spinup_name = f"basic_{light_model}_{spinup_species}_2nd_round"
# spinup_name = f"basic_{light_model}_{spinup_species}_pure_SBA_2nd_round"
dmr_path = spinups_path.joinpath(spinup_name + ".dmr_eq")

# load fake equilibrium dmr
dmr_eq = DLAPM.load_from_file(dmr_path)

# initialize soil and wood product models with spinup stocks
soil_model = SimpleSoilCModel(
    initial_stocks=Q_(dmr_eq.xss[dmr_eq.soil_pool_nrs], "gC/m^2")
)
wood_product_model = SimpleWoodProductModel(
    initial_stocks=Q_(dmr_eq.xss[dmr_eq.wood_product_pool_nrs], "gC/m^2")
)
stand_params = stand_params_library["default"]
stand_params["soil_model"] = soil_model
stand_params["wood_product_model"] = wood_product_model

dmr_path


# +
species = sim_dict["species"]

management_strategy = [
    ("StandAge3", "Plant"),
    ("PCT", "T0.75"),  # pre-commercial thinning
    ("DBH35-80", "CutWait3AndReplant"),
    #    # needs to be lower priority than any cutting, otherwise cutting might be delayed
    #    ("SBA25", "ThinStandToSBA18"), # SBA dependent thinning
]

sim_profile, sim_name = [
    (species, 1.0, sim_dict["N"] / 10_000, management_strategy, "waiting"),
], f"single_{species}"

emergency_action_str, emergency_direction, emergency_stand_action_str = (
    "Die",
    "below",
    "",
)
# emergency_action_str, emergency_direction = "Thin", "below"
# emergency_action_str, emergency_direction = "CutWait3AndReplant", "above"
emergency_q = 0.75

print(sim_name)

species_setting = species_setting_from_sim_profile(sim_profile)

logfile_path = sim_cohort_path.joinpath(sim_name + ".log")
print(f"log file: {logfile_path}")

# +
# %%time

# import warnings
# with warnings.catch_warnings():
#    warnings.simplefilter("error")

stand = Stand.create_empty(stand_params)
stand.add_trees_from_setting(
    species_setting, custom_species_params=custom_species_params
)

print(stand)

# +
final_felling = True

if final_felling:
    total_length = 80
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
    #    final_felling,
    emergency_action_str,
    emergency_direction,
    emergency_q,  # fraction to keep
    emergency_stand_action_str,  # in case of emergency, also do this
)

# ### Save recorded simulation

filepath = sim_cohort_path.joinpath(sim_name + ".dmp")
filepath

recorded_simulation.save_to_file(filepath)
print(filepath)

# +
# recorded_simulation = RecordedSimulation.from_file(filepath)
# -

ds = recorded_simulation.ds
ds

filepath = sim_cohort_path.joinpath(sim_name + ".nc")
ds.to_netcdf(str(filepath))
print(filepath)

# ## Compute transit time variables, carbon sequestration, add to dataset

cache_size = 30_000
verbose = True

# +
# %%time

# create discrete model run from stocks and fluxes
dmr = utils.create_dmr_from_stocks_and_fluxes(
    ds
)  # , GPP_total_prepend=ds_long.GPP_total[-(nr_years+1)])

ds = sim_utils.compute_BTT_vars_and_add_to_ds(
    ds, dmr, dmr_eq, up_to_order=2, cache_size=cache_size, verbose=verbose
)
ds = sim_utils.compute_C_balance_and_CS_and_add_to_ds(
    ds, dmr, cache_size=cache_size, verbose=verbose
)
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
    clearcut_index=None,
    time_index_stop=len(ds.time) - 2,
    year_shift=0,
    cache_size=1_000,
)
print(filepath)
