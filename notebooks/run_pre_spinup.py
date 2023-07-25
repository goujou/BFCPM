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

# # Simulation pre-spinup
#
# To be able to compare the outcomes of serveral simulations, these simulations must start with the same initial data. Such initial data contains stock sizes and the age structure of the pools. To this end, we run a **rather long** simulation with some **arbitrary species composition and management strategy**. Then we create a discrete model run (``dmr``), and average all input fluxes and all compertmental matrixes over the last **arbitrary** ``nr_timesteps`` years to obtain ``mean_U`` and ``mean_B``. Then we compute ``mean_x = inv(Id - mean_B) @ mean_U`` and scale ``mean_U`` to better meet ``mean_x`` with the created a discrete model run in equilibrium (``dmr_eq``). This ``dmr_eq`` is saved to disk and will be loaded before each simulation run.
#
# Before the simulation, we then use ``dmr_eq`` to compute initial age moments up to order 2 and initial age distributions. This way all simulations start with the same stocks and age structure of the pools associated to soil and wood products. Trees will be planted at the beginning of each simulation and have an initial age of zero.

# %load_ext autoreload

# +
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import inv
import xarray as xr
import argparse

from bgc_md2.notebook_helpers import write_to_logfile
from CompartmentalSystems.discrete_model_run import DiscreteModelRun as DMR
from LAPM.discrete_linear_autonomous_pool_model import DiscreteLinearAutonomousPoolModel as DLAPM

from BFMM import utils
from BFMM import DATA_PATH, Q_, PRE_SPINUPS_PATH
from BFMM.simulation_parameters import stand_params_library
from BFMM.simulation import utils as sim_utils
from BFMM.management.library import (
    species_setting_from_sim_profile,
)
from BFMM.soil.simple_soil_model.C_model import SimpleSoilCModel
from BFMM.wood_products.simple_wood_product_model.C_model import SimpleWoodProductModel
from BFMM.stand import Stand
from BFMM.simulation.library import prepare_forcing
from BFMM.simulation.recorded_simulation import RecordedSimulation
from BFMM.trees.single_tree_params import species_params

# %autoreload 2

# +
# #%tb

try:
    parser = argparse.ArgumentParser()
    parser.add_argument("pre_spinup_date", type=str)
    args = parser.parse_args()

    pre_spinup_date = args.pre_spinup_date
    print("Pre-spinup settings from command line")
except SystemExit:
    print("Default pre-spinup settings")
    pre_spinup_date = "2023-07-25"
    
print(pre_spinup_date)

# +
pre_spinups_path = PRE_SPINUPS_PATH.joinpath(pre_spinup_date)
pre_spinups_path.mkdir(exist_ok=True, parents=True)

# filename for the current spinup dmr
species = "pine"
#species = "spruce"
#species = "birch"
#light_model = "Zhao" # Zhao or Spitters
light_model = "Spitters" # Zhao or Spitters
pre_spinup_name = f"basic_{light_model}_{species}"

# output files
dmr_eq_path = pre_spinups_path.joinpath(pre_spinup_name + ".dmr_eq")
dmr_path = pre_spinups_path.joinpath(pre_spinup_name + ".dmr")
nc_path = pre_spinups_path.joinpath(pre_spinup_name + ".nc")
dmp_path = pre_spinups_path.joinpath(pre_spinup_name + ".dmp")
logfile_path = pre_spinups_path.joinpath(pre_spinup_name + ".log")

# number of years at the end of the simulation to be used
# to compute the mean values on which the fake equilibrium is based
nr_timesteps = 50

logfile_path
# -

# ### Custom species parameters?

# tree species parameter changes can be made here
custom_species_params = species_params.copy()

forcing = prepare_forcing(nr_copies=8, year_offset=-160)

# +
management_strategy = [
    ("StandAge3", "Plant"),
    ("PCT", "T0.75"), # pre-commercial thinning
    ("SBA25-80-160", "ThinStandToSBA18"), # SBA dependent thinning, not between 70 and 80, 150 and 160 yrs of sim
#    ("SBAvsDTHBrownLower80-160", "ThinningStandGreenLower"), # SBA and DTH dependent thinning
    ("StandAge79", "CutWait3AndReplant"), # clear cut with replanting after 80 yrs (Triggered after 79 years, cut next year)
    ("DBH35-80-160", "CutWait3AndReplant"),
]

sim_profile = [(species, 1.0, 0.20,  management_strategy, "waiting")]
sim_name = pre_spinup_name

emergency_action_str, emergency_direction, emergency_stand_action_str = "Die", "below", ""
#emergency_action_str, emergency_direction = "Thin", "below"
#emergency_action_str, emergency_direction = "CutWait3AndReplant", "above"
emergency_q = 0.75

species_setting = species_setting_from_sim_profile(sim_profile)

print(sim_name)

# +
# %%time

empty_soil_model = SimpleSoilCModel()
empty_wood_product_model = SimpleWoodProductModel()

stand_params = stand_params_library["default"]
stand_params["soil_model"] = empty_soil_model
stand_params["wood_product_model"] = empty_wood_product_model

stand = Stand.create_empty(stand_params)
stand.add_trees_from_setting(species_setting, custom_species_params=custom_species_params)
print(stand)

# +
final_felling = True

if final_felling:
    total_length = 160
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
    emergency_q, # fraction to keep
    emergency_stand_action_str
)

# ### Save recorded simulation

recorded_simulation.save_to_file(dmp_path)
print(dmp_path)

ds = recorded_simulation.ds
ds

ds.to_netcdf(str(nc_path))
nc_path

ds = xr.load_dataset(nc_path)

# ## Show spinup

# +
fig, axes = plt.subplots(figsize=(12, 4*2), nrows=2)
axes = iter(axes)

ax = next(axes)
ds.stocks.sel(entity="wood_product", pool=["WP_S", "WP_L"]).plot.line(ax=ax, x="time")

ax = next(axes)
ds.stocks.sel(entity="soil", pool=["Litter", "CWD", "SOC"]).plot.line(ax=ax, x="time")

fig.tight_layout()


# +
fig, axes = plt.subplots(figsize=(12, 4*3), nrows=3)
axes = iter(axes)

ax = next(axes)
ds.DBH.plot.line(ax=ax, x="time")
ax.set_title("DBH")

ax = next(axes)
ds.height.plot.line(ax=ax, x="time")
ax.set_title("Tree height")

ax = next(axes)
ds.stand_basal_area.plot(ax=ax)
ax.set_title("Stand basal area")

fig.tight_layout()
# -

# # Fake equilibrium initial soil and wood product model stocks
#
# And save fake equilibrium dmr to file.

# +
# create discrete model run from stocks and fluxes
# shorten the data time step artificially to be able to create DMR
#nr_all_pools = stand.nr_trees * stand.nr_tree_pools + stand.nr_soil_pools
dmr = utils.create_dmr_from_stocks_and_fluxes(ds)

dmr.tree_pool_nrs = utils.get_global_pool_nrs_from_entity_nrs(
    ds.tree_entity_nrs.data,
    ds
)

dmr.soil_pool_nrs = utils.get_global_pool_nrs_from_entity_nrs(
    [ds.soil_entity_nr],
    ds
)
dmr.wood_product_pool_nrs = utils.get_global_pool_nrs_from_entity_nrs(
    [ds.wood_product_entity_nr],
    ds
)

dmr.save_to_file(dmr_path)
dmr_path

# +
# create DMR at a fake equilibrium based on the last timesteps of the simulation

xs = dmr.solve()
mean_B = dmr.Bs[-nr_timesteps:].mean(axis=0)
mean_U = dmr.net_Us[-nr_timesteps:].mean(axis=0) # last nr_timesteps elements
mean_x = xs[-nr_timesteps:].mean(axis=0)

# -


Id = np.identity(dmr.nr_pools)
xss = inv(Id-mean_B) @ mean_U

# +
# scale mean_U to better meet mean_x

mean_U = mean_U * mean_x / xss

dmr_eq = DLAPM(mean_U, mean_B, check_col_sums=False)

# add pool number descriptions to dmr_eq
dmr_eq.tree_pool_nrs = utils.get_global_pool_nrs_from_entity_nrs(
    ds.tree_entity_nrs.data,
    ds
)

dmr_eq.soil_pool_nrs = utils.get_global_pool_nrs_from_entity_nrs(
    [ds.soil_entity_nr],
    ds
)
dmr_eq.wood_product_pool_nrs = utils.get_global_pool_nrs_from_entity_nrs(
    [ds.wood_product_entity_nr],
    ds
)

#mean_GPP = ds.GPP_total[:-nr_timesteps:].mean(dim="time")
#dmr_eq.GPP = mean_GPP

dmr_eq.save_to_file(dmr_eq_path)
dmr_eq_path
# -


# # Second round
#
# Run a second spinup starting with the results of the first spinup.

# +
# load fake equilibrium dmr
dmr_eq = DLAPM.load_from_file(dmr_eq_path)

# initialize soil and wood product models with spinup stocks
soil_model = SimpleSoilCModel(initial_stocks=Q_(dmr_eq.xss[dmr_eq.soil_pool_nrs], "gC/m^2"))
wood_product_model = SimpleWoodProductModel(initial_stocks=Q_(dmr_eq.xss[dmr_eq.wood_product_pool_nrs], "gC/m^2"))
stand_params["soil_model"] = soil_model
stand_params["wood_product_model"] = wood_product_model

# output files
dmr_eq_2nd_round_path = pre_spinups_path.joinpath(pre_spinup_name + "_2nd_round" + ".dmr_eq")
dmr_2nd_round_path = pre_spinups_path.joinpath(pre_spinup_name + "_2nd_round" + ".dmr")
nc_path = pre_spinups_path.joinpath(pre_spinup_name + "_2nd_round" + ".nc")
dmp_path = pre_spinups_path.joinpath(pre_spinup_name + "_2nd_round" + ".dmp")
logfile_path = pre_spinups_path.joinpath(pre_spinup_name + "_2nd_round.log")

stand = Stand.create_empty(stand_params)
stand.add_trees_from_setting(species_setting, custom_species_params=custom_species_params)


# +
final_felling = True

if final_felling:
    total_length = 160
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
    emergency_q, # fraction to keep
    emergency_stand_action_str
)

# ### Save recorded simulation

recorded_simulation.save_to_file(dmp_path)
print(dmp_path)

ds = recorded_simulation.ds
ds

ds.to_netcdf(str(nc_path))
nc_path    

# ## Show spinup

# +
fig, axes = plt.subplots(figsize=(12, 4*2), nrows=2)
axes = iter(axes)

ax = next(axes)
ds.stocks.sel(entity="wood_product", pool=["WP_S", "WP_L"]).plot.line(ax=ax, x="time")

ax = next(axes)
ds.stocks.sel(entity="soil", pool=["Litter", "CWD", "SOC"]).plot.line(ax=ax, x="time")

fig.tight_layout()


# +
fig, axes = plt.subplots(figsize=(12, 4*3), nrows=3)
axes = iter(axes)

ax = next(axes)
ds.DBH.plot.line(ax=ax, x="time")
ax.set_title("DBH")

ax = next(axes)
ds.height.plot.line(ax=ax, x="time")
ax.set_title("Tree height")


ax = next(axes)
ds.stand_basal_area.plot(ax=ax)
ax.set_title("Stand basal area")

fig.tight_layout()
# -

# # Fake equilibrium initial soil and wood product model stocks
#
# And save fake equilibrium dmr to file.

# +
# create discrete model run from stocks and fluxes
# shorten the data time step artificially to be able to create DMR
#nr_all_pools = stand.nr_trees * stand.nr_tree_pools + stand.nr_soil_pools
dmr = utils.create_dmr_from_stocks_and_fluxes(ds)

dmr.tree_pool_nrs = utils.get_global_pool_nrs_from_entity_nrs(
    ds.tree_entity_nrs.data,
    ds
)

dmr.soil_pool_nrs = utils.get_global_pool_nrs_from_entity_nrs(
    [ds.soil_entity_nr],
    ds
)
dmr.wood_product_pool_nrs = utils.get_global_pool_nrs_from_entity_nrs(
    [ds.wood_product_entity_nr],
    ds
)

dmr.save_to_file(dmr_2nd_round_path)

# +
# create DMR at a fake equilibrium based on the last timesteps of the simulation

xs = dmr.solve()
mean_B = dmr.Bs[-nr_timesteps:].mean(axis=0)
mean_U = dmr.net_Us[-nr_timesteps:].mean(axis=0) # last nr_timesteps elements

# scale mean_U to better meet mean_x
mean_x = xs[-nr_timesteps:].mean(axis=0)
Id = np.identity(dmr.nr_pools)
xss = inv(Id-mean_B) @ mean_U
mean_U = mean_U * mean_x / xss

dmr_eq = DLAPM(mean_U, mean_B, check_col_sums=False)

dmr_eq.tree_pool_nrs = utils.get_global_pool_nrs_from_entity_nrs(
    ds.tree_entity_nrs.data,
    ds
)

dmr_eq.soil_pool_nrs = utils.get_global_pool_nrs_from_entity_nrs(
    [ds.soil_entity_nr],
    ds
)
dmr_eq.wood_product_pool_nrs = utils.get_global_pool_nrs_from_entity_nrs(
    [ds.wood_product_entity_nr],
    ds
)

dmr_eq.save_to_file(dmr_eq_2nd_round_path)
dmr_eq_2nd_round_path
# -



