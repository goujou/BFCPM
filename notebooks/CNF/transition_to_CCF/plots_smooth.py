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

# # Transitioning from clear-cut strategy to CCF
# We assume a history with clear-cut strategy. This means we have pines only, planted in very slightly different starting sizes in order to reflect natural variablity. This spinup lasts for 160 years.
#
# After the spinup, i.e. now, be start a transition to a continuous-cover forestry. We assume that a clear-cut roration lasts 80 years, hence we think of 80 forest stands, each one in a different year of the clear-cut rotation. Stands which are less than 50 years into the rotation, continue with their clear-cut strategy until they are 50 years into the rotation and then start the transition. Stands which are already 50 or more years into the rotation start the transition immediately.
#
# ## The transition
# At the beginning of the transition the oldest/tallest 25% of the pine trees are cut down and replaced by new pine seedlings. After 20 years, the now oldest/tallest 25% of the pine trees are cut down and replaced by new seedlings. We continue this way such that after 60 years the last pine trees from the clear-cut strategy are removed and replaced by new seedlings. Then, every 20 years, we cut down the oldest/tallest pine trees and replace them with new seedlings. This way we establish a continuous-cover pine forest with four tree cohorts with ages 20 years apart.
#
# **Note**: In an earlier approach, we just waited for all stands to finish their 80-year clear-cut rotation and plant a new continuous-cover forest stand from scratch. The new, more sophisticated approach, requires more independent stand simulations.
#
# ## Future
# We initiate the transition process not, at year 0 (after 160 years of clear-cut forestry spinup), and run simulations 240 years into the future. Stands with age 50 or more will transition immediately to CCF, stands with the age 1, 2, 3, ...,  49 will wait 49, 48, 47, ... 1 years before they start the transition. After 50 years all stands will be in the transition phase, and after 50 + 60 years, all stands will have replaced all original clear-cut based trees.
#
# ## Assessment metrics
#
# - Yield ($Y = Y_S + Y_L$)
#   
#     Wood product yield is the integral over all fluxes entering the wood-product pools. We consider short-lasting ($W_S$) and long-lasting ($W_L$) wood products.
#
# - Integrated Net Carbon Balance (INCB)
#   
#     INCB(T) is the integral over the C inputs minus the C outputs, or, equivalently, the total C stocks at time T minus the total C stocks at time 0. The dimension is mass per area. INCB ignores the time that carbon spends in the forest/soil/wood-product system and hence outside the atmosphere.
#
# - Integrated Carbon Stocks (ICS)
#
#     ICS(T) is the integral over the total carbon stocks from time $t=0$ to time $t=T$. The deminsion is mass per area times time, because ICS takes into account both the mass of carbon in the system and the time for which this carbon resides in the system. Hence, ICS, in contrast to INCB, is a tool to assess *avoided* radiative damage to the atmosphere.
#
# ## Plots
# - We call the depth of how old a current clear-cut stand is ``delay``, which might be badly chosen.
# - We compare two scenarios: "business as usual (BAU)" and "transition to CNF".
#
# # ## ! ATTENTION
# Should we, for the wood products, make a "virtual final felling", to take into account what is in the forest after 240 years? Or should we report this extra?

# +
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from collections import namedtuple
from tqdm import tqdm

from BFCPM.__init__ import Q_, SIMULATIONS_PATH
from BFCPM import utils

# +
# set plotting properties

mpl.rcParams['lines.linewidth'] = 4

SMALL_SIZE = 16
MEDIUM_SIZE = 17
#BIGGER_SIZE = 30

SMALL_SIZE = MEDIUM_SIZE

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# -

# ## Load simulation data

#sim_date = "2023-05-22"
#sim_date = "2023-06-09"
sim_date = "2024-03-06"
sim_cohort_path = SIMULATIONS_PATH.joinpath(sim_date)

dss_sim = dict()
#dmrs_sim = dict()

for p in sim_cohort_path.iterdir():
    if p.suffix == ".nc":
        sim_name = p.stem
        if (sim_name.find("patch_transition") != -1) and (sim_name.find("no_dmr_sim") != -1):
            delay = int(sim_name[43:45])
#            print(sim_name, delay)

            dss_sim[delay] = xr.open_dataset(str(p))
#            p_dmr = sim_cohort_path.joinpath(sim_name + ".dmr")
#            dmrs_sim[delay] = DMR.load_from_file(p_dmr)

# +
# only 73 out of 80 stands were able to make a successful transition
# the other simulations failed (needs to be investigated)
# for consistency reasons we will also consider only 73 BAU stands for now

nr_keys = len(sorted(dss_sim.keys()))
nr_keys

# +
p = sim_cohort_path.joinpath("DWC_BAU_320_pine_12_sim.nc")
ds_sim_BAU = xr.open_dataset(str(p))

#p = sim_cohort_path.joinpath("DWC_BAU_320_pine_12.nc")
#ds_BAU = xr.open_dataset(str(p))

#p_dmr = sim_cohort_path.joinpath("DWC_BAU_320_pine_12_sim.dmr")
#dmr_sim_BAU = DMR.load_from_file(p_dmr)

# +
# we look 240 years into the future
timespan = 240

# a technical constant, depending which dataset we load:
# 0 if we load the dataset starting from now
# 160 if we load the dataset starting with the spinup
shift = 0
# -

# ## Compute the assessment metrics


# +
#variable_names = "Y_S", "Y_L", "Y", "INCB", "IITT", "ICS"
variable_names = "Total_C_stock", "Y_S", "Y_L", "Y", "INCB", "ICS"
#variable_names = "INCB", "ICS"

units = {
    "Total_C_stock": r"kgC$\,$m${}^{-2}$",
    "Y_S": r"kgC$\,$m${}^{-2}$",
    "Y_L": r"kgC$\,$m${}^{-2}$",
    "Y": r"kgC$\,$m${}^{-2}$",
    "INCB": r"kgC$\,$m${}^{-2}$",
    "ICS": r"kgC$\,$m${}^{-2\,}yr$",
}    
# -

#Metrics = namedtuple("Metrics", ["start_year", "year", "Y_S", "Y_L", "Y", "INCB", "IITT", "ICS"])
Metrics = namedtuple("Metrics", ["start_year", "year", *variable_names])


# +
# create discrete model run from stocks and fluxes
# shorten the data time step artificially to be able to create DMR
#nr_all_pools = stand.nr_trees * stand.nr_tree_pools + stand.nr_soil_pools

def compute_metrics(ds_sim, start_year, end_year):
    ds = ds_sim.sel(time=ds_sim.time[start_year:end_year])

    total_C_stock = Q_(ds.stocks.sum(dim=["pool", "entity"]).data * 1e-03, "yr kgC/m^2").magnitude

    Y_S = ds.internal_fluxes.sel(pool_to="WP_S").sum(dim=["entity_to", "entity_from", "pool_from"]).cumsum(dim="time")
    Y_S = Q_(Y_S.data, ds.stocks.attrs["units"]).to("kgC/m^2").magnitude

    Y_L = ds.internal_fluxes.sel(pool_to="WP_L").sum(dim=["entity_to", "entity_from", "pool_from"]).cumsum(dim="time")
    Y_L = Q_(Y_L.data, ds.stocks.attrs["units"]).to("kgC/m^2").magnitude

    Y = Y_S + Y_L
    
    INCB = (ds.input_fluxes.sum(dim=["entity_to", "pool_to"]) - ds.output_fluxes.sum(dim=["entity_from", "pool_from"])).cumsum()
    INCB = Q_(INCB.data * 1e-03, "kgC/m^2").magnitude
      
    ICS = Q_(ds.stocks.sum(dim=["pool", "entity"]).cumsum().data * 1e-03, "yr kgC/m^2").magnitude
    
    return [Metrics(start_year, year, *el) for year, el in enumerate(zip(total_C_stock, Y_S, Y_L, Y, INCB, ICS))]



# -

# ### Business as usual (BAU)

# +
dfs = []
for delay in tqdm(sorted(dss_sim.keys())):
#    print(delay)

    start_year = shift + delay
    end_year = shift + delay + timespan
    
    metrics = compute_metrics(ds_sim_BAU, start_year, end_year)
    
    df = pd.DataFrame(metrics).reset_index(drop=True)
    df["delay"] = delay
    dfs.append(df)
    
df_BAU = pd.concat(dfs).reset_index()
# -

df_BAU

# +
dfs = []
for delay, ds_sim in tqdm(sorted(dss_sim.items())):
#    print(delay)

    start_year = shift + delay
    end_year = shift + delay + timespan
    
    metrics = compute_metrics(ds_sim, start_year, end_year)
    
    df = pd.DataFrame(metrics).reset_index(drop=True)
    df["delay"] = delay
    dfs.append(df)
    
df_CNF = pd.concat(dfs).reset_index()
# -

df_CNF

# ## Plots
#
# In the following plots, "start_year" and "delay" both means how deep we are in the clear-cut rotation. BAU is business as usual, continue with clear-cut even-aged pine and a rotation time of 80 years.
#
# Transition means that at some point, we start to change to establish a CCF.

# We assume that we have spatially distributed stands at all ages. In different locations, we start the transition at different times, depending on the stand age. If the clear-cut stand age is 30, then we start the transition in 20 years.
#
# The next plots show the mean over all locations (= mean over all current stand ages).

# +
variable_names_tmp = ["Y", "Y_S", "Y_L", "INCB", "ICS"]
titles = [
    "Spatially averaged wood production ($Y$)",
    "Spatially averaged short-lasting wood production ($Y_S$)",
    "Spatially averaged long-lasting wood production ($Y_L$)",
    "Spatially averaged net C balance (INCB)",
    "Spatially averaged climate change mitigation potential (ICS)"
]

n = len(variable_names_tmp)
fig, axes = plt.subplots(figsize=(8, 4*n), nrows=n)

for variable_name, ax, title in zip(variable_names_tmp, axes, titles):
    ax.set_title(title)
    df_BAU.set_index("year").groupby("year")[variable_name].mean().plot(ax=ax, label="BAU", legend=True)
    df_CNF.set_index("year").groupby("year")[variable_name].mean().plot(ax=ax, label="transition", legend=True)
    ax.set_xlabel("")
    
    ax.set_ylabel(units[variable_name])
    ax.set_xlim([0, timespan])

ax.set_xlabel("year in the future")
fig.tight_layout()

# +
n = len(variable_names_tmp)
fig, axes = plt.subplots(figsize=(8, 4*n), nrows=n)

for variable_name, ax, title in zip(variable_names_tmp, axes, titles):
    title = variable_name if variable_name.find("Y") == -1 else "$" + variable_name + "$"
    ax.set_title(title + " (transition minus BAU)")
    ax.plot(
        df_CNF.set_index("year").groupby("year")[variable_name].mean() - df_BAU.set_index("year").groupby("year")[variable_name].mean(),
        color="black"
    )

    ax.set_ylabel(units[variable_name])
    ax.set_xlim([0, timespan])
    ax.axhline(0, alpha=0.5, color="black", lw=1)

ax.set_xlabel("year in the future")
fig.tight_layout()
# -

# ### Selected stands, the number means the age/depth of the current clear-cut rotation

# different values of depth into the clear-cut rotation
plot_times = [0, 20, 40, 60, 80]

# +
variable_names_tmp = variable_names[1:]

n = len(variable_names_tmp)
fig, axes = plt.subplots(figsize=(8, 4*n), nrows=n)

for variable_name, ax in zip(variable_names_tmp, axes):
    for y0 in plot_times:
        df_BAU[df_BAU["delay"] == y0].set_index("year")[variable_name].plot(ax=ax, label=str(y0), legend=True)
    
    title = variable_name if variable_name.find("Y") == -1 else "$" + variable_name + "$"
    ax.set_title(title)
    ax.set_ylabel(units[variable_name])
    ax.set_xlim([0, timespan])
    ax.set_xlabel("")

ax.set_xlabel("year in the future")
fig.suptitle("BAU")
fig.tight_layout()

# +
variable_names_tmp = variable_names[1:]

n = len(variable_names_tmp)
fig, axes = plt.subplots(figsize=(8, 4*n), nrows=n)

for variable_name, ax in zip(variable_names_tmp, axes):
    for y0 in plot_times:
        df_CNF[df_CNF["delay"] == y0].set_index("year")[variable_name].plot(ax=ax, label=str(y0), legend=True)

    title = variable_name if variable_name.find("Y") == -1 else "$" + variable_name + "$"
    ax.set_title(title)
    ax.set_ylabel(units[variable_name])
    ax.set_xlim([0, timespan])
    ax.set_xlabel("")

ax.set_xlabel("year in the future")
fig.suptitle("transition")
fig.tight_layout()
# -

# ### Transition minus BAU
#
# The next plots tell us, which current clear-cut stand age is the best suited (under our current transition strategy) for a transition. In terms of $Y_L$ (long-lasting wood products), a 60 year old clear-cut stand starting the transition process right away is the most productive. Howver, in terms of ICS (climate change mitigation potential), it is pretty bad.

# +
variable_names_tmp = variable_names[1:]

n = len(variable_names_tmp)
fig, axes = plt.subplots(figsize=(8, 4*n), nrows=n)

for variable_name, ax in zip(variable_names_tmp, axes):
    for y0 in plot_times:
        (df_CNF[df_CNF["delay"] == y0].set_index("year")[variable_name] - df_BAU[df_BAU["delay"] == y0].set_index("year")[variable_name]).plot(ax=ax, label=str(y0), legend=True)

    title = variable_name if variable_name.find("Y") == -1 else "$" + variable_name + "$"
    ax.set_title(title)
    ax.set_ylabel(units[variable_name])
    ax.set_xlim([0, timespan])
    ax.set_xlabel("")

    ax.axhline(0, alpha=0.5, color="black", lw=1)

ax.set_xlabel("year in the future")
fig.suptitle("Transition - BAU")
fig.tight_layout()
# -
# ## Summary
#
# ### Spatial aggregation over all stands/plots
# - The transition leads to a rapid relative increase in C stocks.
# - This will settle and after 240 years the C stocks are comparable in both scenarios.
# - The initial increase in C stocks lifts the transition ICS to a higher level, where it stabilizes.
# - Wood productivity will also benefit from the transition.
#
# ### Selected stands/plots with clear-cut stand ages 0, 20, 40, 60
# - A 60 year old stand is very productive in terms of long-lasting wood products if it starts to trasition now.
# - The same stand does not contribute positively to the overall averaged positive relative climate change mitigation if the transition.
# - Different goals (wood production vs climate change mitigation) clash here if we want to select some stand ages for transition or exclude them from it.

# ## TODO
#
# The initial total C stocks in BAU (some years into the rotation) and transition (the same number of years into the rotation) do not always coincide, because the transition simulations did not inlcude the pre-commercial thinning (PCT) during their clear-cut period (until the transition starts). Trying to include PCT into the transition simulations implicated all kinds of simulation problems that have to be debugged.
#
# In the following plot, the lines should start off at the exact same point.

# +
variable_name = "Total_C_stock"
fig, ax = plt.subplots(figsize=(8, 4))

title = variable_name.replace("_", " ")
ax.set_title(title)
df_BAU.set_index("year").groupby("year")[variable_name].mean().plot(ax=ax, label="BAU", legend=True)
df_CNF.set_index("year").groupby("year")[variable_name].mean().plot(ax=ax, label="transition", legend=True)
    
ax.set_ylabel(units[variable_name])
ax.set_xlim([0, timespan])

ax.set_xlabel("year in the future")
fig.tight_layout()
# -








