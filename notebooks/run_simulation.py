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

# # Run clear-cut simulation after age-distributed spinup

# ...spinup 160 or 240 years, simulation 80 years

# %load_ext autoreload

import argparse

# +
import numpy as np
import xarray as xr
from BFCPM import DATA_PATH, PRE_SPINUPS_PATH, Q_, utils
from BFCPM.management.library import species_setting_from_sim_profile
from BFCPM.simulation import utils as sim_utils
from BFCPM.simulation.library import (create_mixed_aged_sim_profile,
                                      load_clear_cut_sim_profiles,
                                      prepare_forcing)
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

# ### Custom species parameters?

# +
# #%tb

try:
    parser = argparse.ArgumentParser()
    parser.add_argument("pre_spinup_date", type=str)

    parser.add_argument(
        "cc_spinup_species", type=str
    )  # continuous-cover (age-distributed) spinup
    parser.add_argument("cc_spinup_length", type=int)
    parser.add_argument("cc_spinup_N", type=int)

    parser.add_argument("sim_date", type=str)
    parser.add_argument("sim_name", type=str)
    parser.add_argument("sim_N", type=int)

    parser.add_argument("emergency_action_str", type=str)
    parser.add_argument("emergency_direction", type=str)
    parser.add_argument("emergency_stand_action_str", type=str)

    args = parser.parse_args()

    pre_spinup_date = args.pre_spinup_date
    cc_spinup_species = args.cc_spinup_species
    cc_spinup_length = args.cc_spinup_length
    cc_spinup_N = args.cc_spinup_N
    sim_date = args.sim_date
    sim_name = args.sim_name
    sim_N = args.sim_N

    emergency_action_str = args.emergency_action_str
    emergency_direction = args.emergency_direction
    emergency_stand_action_str = args.emergency_stand_action_str
    print("Simulation settings from command line")
except SystemExit:
    print("Standard simulation settings")

    pre_spinup_date = "2023-07-25"

    # cc stand for continuous-cover
    cc_spinup_species = "pine"
    cc_spinup_length = 8 * 20

    cc_spinup_N = 1_500

    sim_date = "2023-07-25"
    sim_name = "mixed-aged_pine_long"
    #    sim_name = "even-aged_pine_long"
    #    sim_name = "even-aged_spruce_long"
    #    sim_name = "even-aged_mixed_long"

    sim_N = 2_000

    emergency_action_str = "Die"
    emergency_direction = "below"
    emergency_stand_action_str = "ThinStandToSBA18"

sim_dict = {
    "pre_spinup_date": pre_spinup_date,
    "cc_spinup_species": cc_spinup_species,
    "cc_spinup_length": cc_spinup_length,
    "cc_spinup_N": cc_spinup_N,
    "sim_date": sim_date,
    "sim_name": sim_name,
    "sim_length": 4 * 20,
    "N": sim_N,
    "emergency_action_str": emergency_action_str,
    "emergency_direction": emergency_direction,
    "emergency_stand_action_str": emergency_stand_action_str,
}

print(sim_dict)

# +
# for a simulation testing if lower tree density produces more long-lasting wood products
# saved in new folder (15 April, in order not to overwrite)

# sim_dict["N"] = 1_500
print(sim_dict)

# +
# tree species parameter changes can be made here
custom_species_params = species_params.copy()

alpha = 1.00
custom_species_params["pine"]["alpha"]["value"] = alpha
custom_species_params["spruce"]["alpha"]["value"] = alpha
# -

# ## Set up forcing and simulation length

# +
sim_cohort_name = ""
sim_cohort_path = all_sims_path.joinpath(sim_cohort_name)
sim_cohort_path = sim_cohort_path.joinpath(f"{sim_dict['sim_date']}")

sim_cohort_path.mkdir(exist_ok=True)
print(sim_cohort_path)
# -

# ## Load pre-spinup data: soil and wood product stocks and age structure

# +
spinups_path = PRE_SPINUPS_PATH.joinpath(sim_dict["pre_spinup_date"])

light_model = "Zhao"  # Zhao or Spitters
# light_model = "Spitters" # Zhao or Spitters

pre_spinup_species = "pine"
pre_spinup_name = f"basic_{light_model}_{pre_spinup_species}_2nd_round"
dmr_path = spinups_path.joinpath(pre_spinup_name + ".dmr_eq")

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

print(dmr_path)


# +
# simulation data

# start `spinup_length` years earlier so as to have the true start again at 2000
nr_copies = (sim_dict["cc_spinup_length"] + sim_dict["sim_length"]) // 20
forcing = prepare_forcing(
    nr_copies=nr_copies, year_offset=-sim_dict["cc_spinup_length"]
)

# +
if sim_name == "mixed-aged_pine_long":
    sim_profile = create_mixed_aged_sim_profile(
        sim_dict["cc_spinup_species"], sim_dict["cc_spinup_N"], clear_cut_year=None
    )
    sim_name = sim_name.replace("_long", "") + f"_N{sim_dict['cc_spinup_N']}" + "_long"

else:
    spinup_profile = create_mixed_aged_sim_profile(
        sim_dict["cc_spinup_species"],
        sim_dict["cc_spinup_N"],
        clear_cut_year=sim_dict["cc_spinup_length"],
    )

    clear_cut_sim_profiles = load_clear_cut_sim_profiles(
        sim_dict["N"], sim_dict["cc_spinup_length"], sim_dict["sim_length"]
    )
    clear_cut_sim_profile = clear_cut_sim_profiles[sim_dict["sim_name"]]
    sim_profile = spinup_profile + clear_cut_sim_profile

print(sim_name)
[print(ms) for ms in sim_profile]

# +
emergency_action_str = sim_dict["emergency_action_str"]
emergency_direction = sim_dict["emergency_direction"]
emergency_stand_action_str = sim_dict["emergency_stand_action_str"]
# emergency_action_str, emergency_direction = "Die", "below"
# emergency_action_str, emergency_direction = "Thin", "below"
# emergency_action_str, emergency_direction = "CutWait3AndReplant", "above"

emergency_q = (
    0.75  # remaining fraction after emergency thinning (in case it is asked for)
)

species_setting = species_setting_from_sim_profile(sim_profile)

logfile_path = sim_cohort_path.joinpath(sim_name + ".log")
print(f"log file: {logfile_path}")

# +
# %%time

stand = Stand.create_empty(stand_params)
stand.add_trees_from_setting(
    species_setting, custom_species_params=custom_species_params
)

print(stand)

# +
final_felling = True

if final_felling:
    total_length = sim_dict["cc_spinup_length"] + sim_dict["sim_length"]
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

# ## Cut out last part of simulation

nr_years = sim_dict["sim_length"] + 1
ds_long = ds.copy()
ds = ds_long.sel(time=ds_long.time[-nr_years:])
ds = ds.assign({"time": np.arange(len(ds.time))})
ds

# ## Create discrete model run, load initial age data

sim_name = sim_name.replace("_long", "")
print(sim_name)

# ## Compute transit time variables, carbon sequestration, add to dataset

cache_size = 30_000
verbose = True

# +
# %%time

dmr = utils.create_dmr_from_stocks_and_fluxes(
    ds, GPP_total_prepend=ds_long.GPP_total[-(nr_years + 1)]
)

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
    ds_long,
    dmr_eq,
    np.array([dmr.soil_pool_nrs[-1]]),
    filepath,
    resolution=5,
    time_index_start=sim_dict["cc_spinup_length"] - sim_dict["sim_length"],
    clearcut_index=sim_dict["cc_spinup_length"],
    time_index_stop=len(ds_long.time) - 2,
    year_shift=-sim_dict["cc_spinup_length"],
    cache_size=1_000,
)
print(filepath)
# -
