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

# # Common spinup to be used by different subsequent simulations

# %load_ext autoreload

# +
import numpy as np
import xarray as xr
import argparse

from bgc_md2.notebook_helpers import write_to_logfile
from CompartmentalSystems.discrete_model_run import DiscreteModelRun as DMR
from LAPM.discrete_linear_autonomous_pool_model import DiscreteLinearAutonomousPoolModel as DLAPM

from ACGCA import utils
from BFCPM.params import global_tree_params
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

    parser.add_argument("common_spinup_species", type=str) # continuous-cover (age-distributed) spinup
    parser.add_argument("common_spinup_length", type=int)
    parser.add_argument("common_spinup_N", type=int)
 
    parser.add_argument("sim_date", type=str)
    parser.add_argument("sim_name", type=str)
    parser.add_argument("coarseness", type=int)

    args = parser.parse_args()

    pre_spinup_date = args.pre_spinup_date

    common_spinup_species = args.common_spinup_species
    common_spinup_length = args.common_spinup_length
    common_spinup_N = args.common_spinup_N

    sim_date = args.sim_date
    sim_name = args.sim_name
    coarseness = args.coarseness

    print("Simulation settings from command line")
except SystemExit:
    print("Standard simulation settings")

#    pre_spinup_date = "2023-04-09"
    pre_spinup_date = "2023-10-18"

    # "common" means used by all simulations
    common_spinup_species = "pine"
    common_spinup_length = 8 * 20
    common_spinup_N = 2_000

#    coarseness = 1
    coarseness = 12 # every 12th half-hourly entry (6-hourly)

    #    sim_date = "2023-06-09"
    sim_date = "2024-02-14"
#    sim_name = "mixed-aged_pine_long"
#    sim_name = "even-aged_pine_long"
#    sim_name = "even-aged_spruce_long"
#    sim_name = "even-aged_mixed_long"
#    sim_name = f"common_spinup_{common_spinup_species}_{common_spinup_length}_{common_spinup_N}"
    sim_name = f"DWC_common_spinup_clear_cut_{common_spinup_species}_{coarseness:02d}"
    
sim_dict = {
    "pre_spinup_date": pre_spinup_date,
    
    "common_spinup_species": common_spinup_species,
    "common_spinup_length": common_spinup_length,
    "common_spinup_N": common_spinup_N,
    "coarseness": coarseness,

    "sim_date": sim_date,
    "sim_name": sim_name,
}

print(sim_dict)
# -

# ## Set up forcing and simulation length

# +
# simulation data


# start `spinup_length` years earlier so as to have the true start again at 2000
nr_copies = sim_dict["common_spinup_length"] // 20
forcing = prepare_forcing(
    nr_copies=nr_copies,
    year_offset=-sim_dict["common_spinup_length"],
    coarseness=sim_dict["coarseness"]
)

sim_cohort_name = ""
sim_cohort_path = SIMULATIONS_PATH.joinpath(sim_cohort_name)
sim_cohort_path = sim_cohort_path.joinpath(f"{sim_dict['sim_date']}")

sim_cohort_path.mkdir(exist_ok=True)
print(sim_cohort_path)
# -

# ## Load pre-spinup data: soil and wood product stocks and age structure

# +
spinups_path = DATA_PATH.joinpath("pre_spinups").joinpath(sim_dict["pre_spinup_date"])

light_model = "Zhao" # Zhao or Spitters

pre_spinup_species = sim_dict["pre_spinup_species"]
coarseness = sim_dict["coarseness"]

#pre_spinup_name = f"DWC_{light_model}_{pre_spinup_species}_{coarseness:02d}_2nd_round"
pre_spinup_name = f"DWC_{light_model}_{pre_spinup_species}_2nd_round"
dmr_path = spinups_path.joinpath(pre_spinup_name + ".dmr_eq")


# load fake equilibrium dmr
dmr_eq = DLAPM.load_from_file(dmr_path)

# initialize soil and wood product models with spinup stocks
soil_model = SimpleSoilCModel(initial_stocks=Q_(dmr_eq.xss[dmr_eq.soil_pool_nrs], "gC/m^2"))
wood_product_model = SimpleWoodProductModel(initial_stocks=Q_(dmr_eq.xss[dmr_eq.wood_product_pool_nrs], "gC/m^2"))
stand_params["soil_model"] = soil_model
stand_params["wood_product_model"] = wood_product_model

print(dmr_path)


# +
management_strategy = [
    ("StandAge3", "Plant"),
    ("PCT", "T0.75"), # will be reactivated automatically after a clear cut
    ("SBA25-80-160", "ThinStandToSBA18"),
    ("StandAge79", "CutWait3AndReplant"),   
]

species = sim_dict["common_spinup_species"]
N = sim_dict["common_spinup_N"]
if species in ["pine", "spruce"]:
    sim_profile =  [
        (species, 1.0, N / 10_000 / 4, management_strategy, "waiting"),
        (species, 1.2, N / 10_000 / 4, management_strategy, "waiting"),
        (species, 1.4, N / 10_000 / 4, management_strategy, "waiting"),
        (species, 1.6, N / 10_000 / 4, management_strategy, "waiting"),
    ]

elif species == "mixed":
    sim_profile =  [
        ("pine", 1.2, N / 10_000 / 4, management_strategy, "waiting"),
        ("pine", 1.4, N / 10_000 / 4, management_strategy, "waiting"),
        ("spruce", 1.2, N / 10_000 / 4, management_strategy, "waiting"),
        ("spruce", 1.4, N / 10_000 / 4, management_strategy, "waiting"),
    ]
    
else:
    raise ValueError(f"Unknown common spinup species: {species}")


# +
sim_name = sim_dict["sim_name"]

emergency_action_str, emergency_direction = "Die", "below"
#emergency_action_str, emergency_direction = "Thin", "below"
#emergency_action_str, emergency_direction = "CutWait3AndReplant", "above"

emergency_q = 0.75 # remaining fraction after emergency thinning (in case it is asked for)

species_setting = species_setting_from_sim_profile(sim_profile)

logfile_path = sim_cohort_path.joinpath(sim_name + ".log")
print(f"log file: {logfile_path}")

# +
# %%time
  
stand = Stand.create_empty(stand_params)
stand.add_trees_from_setting(
    species_setting,
    custom_species_params=species_params,
    custom_global_tree_params=global_tree_params
)

print(stand)
# -

# ## Run common spinup

recorded_simulation = RecordedSimulation.from_simulation_run(
    sim_name,
    sim_profile,
    light_model,
    forcing,
    custom_species_params,
    stand,
    emergency_action_str,
    emergency_direction,
    emergency_q, # fraction to keep
    logfile_path
)

# ### Save common spinup dataset and simulation

filepath = sim_cohort_path.joinpath(sim_name + ".dmp")
recorded_simulation.save_to_file(filepath)
print(filepath)

recorded_simulation.ds.stocks.sum(dim=["entity", "pool"]).plot()


recorded_simulation.ds.rho_W.isel(tree=3)


