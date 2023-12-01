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
from BFCPM.params import global_tree_params
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

# examples
#sensitivity_str = "S_L,0.9 rho_RL,1.1 Vcmax,0.95"
#sim_cohort_name = "sensitivity_full_sim"

sensitivity_str = list()
sim_cohort_name = ""

try:
    parser = argparse.ArgumentParser()
    parser.add_argument("pre_spinup_date", type=str)

    parser.add_argument("sim_cohort_name", type=str)
    
    parser.add_argument("sim_date", type=str)
    parser.add_argument("sim_name", type=str)
    parser.add_argument("sim_N", type=int)

    parser.add_argument("emergency_action_str", type=str)
    parser.add_argument("emergency_direction", type=str)
    parser.add_argument("emergency_stand_action_str", type=str)

    # actually it's a list of str
    parser.add_argument("sensitivity_str", type=str, nargs=argparse.REMAINDER, default="")

    args = parser.parse_args()

    pre_spinup_date = args.pre_spinup_date

    sim_cohort_name = args.sim_cohort_name

    sim_date = args.sim_date
    sim_name = args.sim_name
    sim_N = args.sim_N

    emergency_action_str = args.emergency_action_str
    emergency_direction = args.emergency_direction
    emergency_stand_action_str = args.emergency_stand_action_str

    sensitivity_str = args.sensitivity_str
    
    print("Simulation settings from command line")
except SystemExit:
    print("Standard simulation settings")

    pre_spinup_date = "2023-07-25" # publication

#    sim_date = "2023-07-26"
    sim_date = "2023-11-24"

#    sim_name = "mixed-aged_pine_long"
    sim_name = "even-aged_pine_long"
    #    sim_name = "even-aged_spruce_long"
    #    sim_name = "even-aged_mixed_long"

    sim_N = 2_000

    emergency_action_str = "Die"
    emergency_direction = "below"
    emergency_stand_action_str = "ThinStandToSBA18"

sim_dict = {
    "pre_spinup_date": pre_spinup_date,
    "sim_date": sim_date,
    "sim_name": sim_name,
    "sim_length": 4 * 20,
    "N": sim_N,
    "emergency_action_str": emergency_action_str,
    "emergency_direction": emergency_direction,
    "emergency_stand_action_str": emergency_stand_action_str,
    "sensitivity_str": sensitivity_str,
    "sim_cohort_name": sim_cohort_name,
}

print(sim_dict)

# +
# for a simulation testing if lower tree density produces more long-lasting wood products
# saved in new folder (15 April, in order not to overwrite)

# sim_dict["N"] = 1_500
print(sim_dict)
# -

# ### Custom species parameters?

# +
# tree species parameter changes can be made here
custom_species_params = species_params.copy()
custom_global_tree_params = global_tree_params.copy()

# here we change the parameters according to the sensitivity_str
# "S_L,0.9 rho_RL,1.1" will multiply S_L by 0.9 and rho_RL by 1.1
sensitivity_str = sim_dict["sensitivity_str"]
if sensitivity_str:
    print()
    print("Changing parameters for sensitivity analysis:")
    for par_str in sensitivity_str:
        par_str = par_str.strip()
        par_name, par_q = par_str.split(",")
        par_name = par_name.strip()
        par_q = float(par_q.strip())

        print(f"{par_name} by factor {par_q}")

        if par_name in custom_species_params["pine"].keys():
            custom_species_params["pine"][par_name]["value"] *= par_q
            custom_species_params["spruce"][par_name]["value"] *= par_q
        elif par_name == "Vcmax":
            custom_global_tree_params["pine"]["photop"][par_name] *= par_q
            custom_global_tree_params["spruce"]["photop"][par_name] *= par_q
        else:
            raise KeyError(f"Unknown parameter name: {par_name}")

    print()             
# -

# ## Set up forcing and simulation length

# +
sim_cohort_name = sim_dict["sim_cohort_name"]
sim_cohort_path = all_sims_path.joinpath(sim_cohort_name)
sim_cohort_path.mkdir(exist_ok=True)

sensitivity_str = sim_dict["sensitivity_str"]
if sensitivity_str:
    sim_cohort_path = sim_cohort_path.joinpath("-".join(sensitivity_str).replace(",", "_").replace(".", ""))
    sim_cohort_path.mkdir(exist_ok=True)

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
nr_copies = sim_dict["sim_length"] // 20
forcing = prepare_forcing(
    nr_copies=nr_copies, year_offset=0
)

# +
if sim_name == "mixed-aged_pine_long":
    sim_profile = create_mixed_aged_sim_profile(
        "pine", sim_dict["N"], clear_cut_year=None
    )
    sim_name = sim_name.replace("_long", "") + f"_N{sim_dict['N']}"

else:
    clear_cut_sim_profiles = load_clear_cut_sim_profiles(
        sim_dict["N"], 0, sim_dict["sim_length"]
    )
    clear_cut_sim_profile = clear_cut_sim_profiles[sim_dict["sim_name"]]
    sim_profile = clear_cut_sim_profile
    sim_name = sim_name.replace("_long", "")

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
    species_setting, 
    custom_species_params=custom_species_params,
    custom_global_tree_params=custom_global_tree_params,
)

print(stand)

# +
final_felling = True

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

# ## Create discrete model run, load initial age data

# ## Compute transit time variables, carbon sequestration, add to dataset

cache_size = 30_000
verbose = True

# +
# %%time

dmr = utils.create_dmr_from_stocks_and_fluxes(ds)

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
    clearcut_index=0,
    time_index_stop=len(ds.time) - 2,
    year_shift=0,
    cache_size=1_000,
)
print(filepath)
