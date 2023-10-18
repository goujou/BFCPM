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

# # Untouched forest for 160/320 yr

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
from BFCPM.__init__ import DATA_PATH, Q_
from BFCPM.simulation import utils as sim_utils
from BFCPM.management.library import species_setting_from_sim_profile
from BFCPM.management.management_strategy import (
    ManagementStrategy,
    OnStandAges, OnSBALimit, PCT,
    Cut, CutWaitAndReplant, ThinStand, Thin
)

from BFCPM.stand import Stand
from BFCPM.simulation.library import prepare_forcing
from BFCPM.prepare_stand import load_wood_product_interface
from BFCPM.trees.single_tree_allocation import SingleTree

from BFCPM.simulation.recorded_simulation import RecordedSimulation
from BFCPM.trees.single_tree_params import species_params

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

#    sim_date = "2023-06-29"
    sim_date = "2023-07-05" # tree deatht at C_S <= 0.5 C_S_star
    sim_name = "DWC_untouched_forest_320"
    
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

# ## Load spinup simulation and try to continue from here

# +
sim_cohort_name = ""
sim_cohort_path = all_sims_path.joinpath(sim_cohort_name)
sim_cohort_path = sim_cohort_path.joinpath(f"{sim_dict['sim_date']}")

sim_cohort_path.mkdir(exist_ok=True)
print(sim_cohort_path)
# -

filepath = sim_cohort_path.joinpath(sim_dict["common_spinup_dmp_filepath"] + ".dmp")
recorded_spinup_simulation = RecordedSimulation.from_file(filepath)
spinup_simulation = recorded_spinup_simulation.simulation

stand = deepcopy(spinup_simulation.stand)

common_spinup_length = stand.age.to("year").magnitude
total_length = common_spinup_length + sim_dict["sim_length"]
total_length

sim_name = sim_dict["sim_name"]
sim_name

wood_product_interface = load_wood_product_interface("no_harvesting", ACGCAMarklundTree, stand.soil_model, stand.wood_product_model)
stand.wood_product_interface = wood_product_interface

# +
light_model = "Zhao" # Zhao or Spitters

# start `spinup_length` years earlier so as to have the true start again at 2000
nr_copies = sim_dict["sim_length"] // 20
forcing = prepare_forcing(nr_copies=nr_copies, year_offset=0)
# -

# ## Remove all management strategies

# +
# remove all management actions from the spinup first

for tree_in_stand in stand.trees:
    ms = list()  
    tree_in_stand.management_strategy = ManagementStrategy(ms)
        
#print(stand)
# -

# ### Add final felling if asked for

# +
#final_felling = True
final_felling = False

if final_felling:
    stand.add_final_felling(Q_(total_length, "yr"))
    
print(stand)
# -

sim_profile = "untouched_forest" # dummy, currently used for logging only

# +
emergency_action_str, emergency_direction, emergency_stand_action_str = "Die", "below", ""
#emergency_action_str, emergency_direction = "Thin", "below"
#emergency_action_str, emergency_direction = "CutWait3AndReplant", "above"

emergency_q = 0.95 # remaining fraction after emergency thinning (in case it is asked for)

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

# +
# %%time

dmr_sim = utils.create_dmr_from_stocks_and_fluxes(ds_sim, GPP_total_prepend=ds.GPP_total[start_year-1])
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

ds.C_S_star_tree


import matplotlib.pyplot as plt

start_index = 160

C_S = Q_(ds.stocks.isel(entity=3).sel(pool="C_S")[start_index:].data, "gC/m^2")
B_S = Q_(ds.stocks.isel(entity=3).sel(pool=["B_TS", "B_OS"]).sum(dim="pool")[start_index:].data, "gC/m^2")
N_per_m2 = Q_(ds.N_per_m2.isel(tree=3)[start_index:].data, "1/m^2")
delta_W = Q_(ds.delta_W.isel(tree=3).data[start_index:], "g_gluc/g_dw")
C_S_star = Q_(ds.C_S_star_tree.isel(tree=3).data[start_index:], "g_gluc/m^2")

C_S = C_S / N_per_m2
B_S = B_S / N_per_m2
C_S_star = C_S_star / N_per_m2

x = np.array(ds.thinning_or_cutting_tree.isel(tree=3).data[start_index:], dtype=int)
x[np.isnan(x)] = 0
xs = np.where(x==1)[0]


# +
plt.plot(C_S.to("g_gluc").to("kg_gluc"), label="$C_S$")
for x in xs:
    plt.axvline(x, c="red", lw=0.4)
    
#plt.plot(B_S.to("g_dw") * delta_W, label=r"maximum $C_S$")
#plt.plot(0.5 * B_S.to("g_dw") * delta_W, label="new death limit")
plt.plot(C_S_star.to("kg_gluc"), label=r"$C_S^{\ast}$")
plt.plot(0.5 * C_S_star.to("kg_gluc"), label="new death limit")

plt.xlabel("time after spinup [yr]")
plt.title(r"Labile C storage for stem and coarse roots and branches ($C_S$)")
plt.legend()
# -

yr = 5
plt.plot(Q_(ds.radius_at_trunk_base.isel(tree=3).diff(dim="time")[start_index:].rolling(time=yr).mean().data, "m/yr").to("mm/yr"))
plt.title(f"{yr}-year averaged radial growth per year")

yr = 5
plt.plot(np.pi*Q_(ds.radius_at_trunk_base.isel(tree=3).diff(dim="time")[start_index:].rolling(time=50).mean().data, "m/yr").to("mm/yr"))
plt.title(f"{yr}-year averaged circumference growth per year")

# ## Do we simulate a kind of self-thinning?

# +
ds_short = ds.where(ds.time >= start_index, drop=True)

ha_per_acre = 0.404686
acre_per_ha = 1 / ha_per_acre

fig, ax = plt.subplots(figsize=(8, 6))

tis = ds_short.time - ds_short.time[0]
sim_name = "untouched forest 320"

#ds = dss_sim[sim_name]
tree_names = ds.entity.isel(entity=ds_short.tree_entity_nrs).data

N_per_m2 = ds_short.N_per_m2.sel(tree=tree_names)

 # actually, Reineke wants the average to be by SBA, not by N_per_m2
#    DBH = ds.DBH.sel(tree=tree_names).weighted(N_per_m2).mean(dim="tree")
#    DBH = Q_(DBH.data, ds.DBH.attrs["units"])
SBA = np.pi * (ds_short.DBH/2)**2
D = ds_short.DBH.weighted(SBA.fillna(0)).mean(dim="tree")
D = Q_(D.data, ds_short.DBH.attrs["units"])
           
# times of cutting or thinning
x = np.array([0] + [v for v in ds_short.time[ds_short.thinning_or_cutting_tree.sel(tree=tree_names).sum(dim="tree") >= 1].data] + [79])

ax.step(tis, N_per_m2.sum(dim="tree") * 10_000, where="pre", label="simulation")
    
# plot Reineke's reference curve (species independent)
#    ax.plot(tis, [(lambda x: np.exp(4.605*acre_per_ha-1.605*np.log(DBH.magnitude[x])))(ti) for ti in tis], label="Reineke's rule", c=colors[sim_name], ls="--")
ax.plot(tis, [(lambda x: np.exp(4.605*acre_per_ha-1.605*np.log(D.magnitude[x])))(ti) for ti in tis], label="Reineke's rule")
ax.legend()
    
ax.set_xlim([tis[0]-1, tis[-1]])
#    ax.set_ylim([0, 2500])
ax.set_title(sim_name.replace("_", " "))
    
ax.set_xlabel("time [yr]")

ax.set_ylabel("trees per ha")
#ax.set_ylim([0,500])
fig.tight_layout()

## save the figure for the publication
#filename = str(pub_figs_path.joinpath("self_thinning.png"))
#fig.savefig(filename)
#filename
# -

