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

# # First ideas about transitioning from clear-cut strategy to CCF
#
# This is a crude first start. We assume a history with clear-cut strategy. This means we have pines only, planted in very slightly different starting sizes in order to reflect natural variablity.
#
# The idea is to take NOW the decision to transit to a continuous cover strategy. This means we have four age cohorts of pine with an age difference of 20 years. In order to get there, we wait until the clear-cut rotation is finished. This means, if the current clear-cut stand is 20 years old, we wait another 60 years (20+60=80), and then start the transition. Starting the transition means cutting down everything and planting new pines, all the same size. After 20 years we remove 1/4 of them, replant them newly. The same after 40 year, 60 year, 80 years.
#
# We are looking here into a future of 160 years. And we are looking at all different current clear-cut stand ages. A just newly replanted stand needs to wait with its transition for 80 years. An old clear-cut stand waits only some years until it reaches the age of 80 and then the transition begins.
#
# This is only one way to do it. I assume, that nobody wants to cut off a recently re-planted clear-cut site, just because. The time window for the future of 160 years is also cosen rather randomly.
#
# All this here is just a first idea. And it seems to tell me, that transition will come at quite some cost, not only in terms of net C balance over the next 160 years (INCB), but also in terms of climate change mitigation, taking retention times into account (ICS).
#
# It also looks like we can identify that for the climate change mitigaion potential, stand with the current-clear-cut age of 40 have the best balance in terms of CBS,. We let them finish their remaining 40 years and then start the transition. For short-term management actions this means: CHANGE NOTHING. And this is what Swedish forestry has been very good at during the last 50 years, I gather here in the course in Vindeln.
#
# All in all, to me it seems, if we have to reach short-term climate goals, a transition from clear-cut strategy to CCF is not hekpful at all. What is not shown here, but what is the result of my previous study, is that an establsihed CCF does really well in terms of climate change mitigation potential. But our way to get there will be very negative.
#
# It's like in life. You know the solution is to not live on the expense of others. If nobody does, people will be great in general. But how can we get there from our current society? There will be a lot to lose on the way for those who already have, and this is why it is not going to happen fast.
#
# Sorry for the philoophical detour. The trees just reminded me of that. It looks like it's the same.

# # ## ! ATTENTION
# Should we, for the wood products, make a "virtual final felling", to take into account what is in the forest after 160 years? Or should we report this extra?

# +
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import pandas as pd
from collections import namedtuple
from tqdm import tqdm

from CompartmentalSystems.discrete_model_run import DiscreteModelRun as DMR, DMRError

from ACGCA.__init__ import DATA_PATH, Q_, ureg
from ACGCA import utils

# +
# set plotting properties

mpl.rcParams['lines.linewidth'] = 4

SMALL_SIZE = 16
MEDIUM_SIZE = 17
#BIGGER_SIZE = 30

SMALL_SIZE = MEDIUM_SIZE

plt.rc('font', size=SM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# -

all_sims_path = DATA_PATH.joinpath("simulations")
all_sims_path.mkdir(exist_ok=True)

#sim_date = "2023-05-22"
sim_date = "2023-06-09"
sim_cohort_path = all_sims_path.joinpath(sim_date)

dss_long = dict()

sim_names = list()
for p in sim_cohort_path.iterdir():
    if p.suffix == ".nc":
        sim_name = p.stem
        print(len(sim_names), sim_name)
        sim_names.append(sim_name)
        dss_long[sim_name] = xr.open_dataset(str(p))

sim_names = ["BAU", "patch_transition_at_end_pine"]

# ## The variables we compute over a window of 160 years from now
#
# If a clear-cut stand is 60 years old, we will continue it 20 years, then cut it down and start the transition.
# We assume that spatially distributed, today we find clear-cut stands of all ages between 0 and 79 equally frequently.
#
# - Y_S: short lasting wood products (pulpo, bioenergy)
# - Y_L: long_lasting wood products (timber)
# - Y = Y_S + Y_L
# - INCB: integrwated net C balance (input from photosynthesis - loss to the atmosphere, summed up over time)
# - IITT: integrated inputs transit time: How much and how long newly uptaken C spends outside the atmosphere.
# - ICS: integrated C stock. It's a measure to not only account for the amount of C stored, but also for the time. This is a true climate change mitigation potential measure, because in contrast to INCB it takes into account also the time of holding C back from the atmosphere, and the effets of legacy C which stems from the clear-cut management as the stands are right now, when we take the decision to transit.

variable_names = "Y_S", "Y_L", "Y", "INCB", "IITT", "ICS"
#variable_names = "Y_S", "Y_L", "Y", "INCB", "ICS"

#Metrics = namedtuple("Metrics", ["start_year", "year", "Y_S", "Y_L", "Y", "INCB", "IITT", "ICS"])
Metrics = namedtuple("Metrics", ["start_year", "year", *variable_names])

start_year = 160
nr_years = 161


# +
# create discrete model run from stocks and fluxes
# shorten the data time step artificially to be able to create DMR
#nr_all_pools = stand.nr_trees * stand.nr_tree_pools + stand.nr_soil_pools

def compute_metrics(ds_long, start_year, nr_years):
    ds = ds_long.sel(time=ds_long.time[start_year:start_year+nr_years])

    Y_S = ds.internal_fluxes.sel(pool_to="WP_S").sum(dim=["entity_to", "entity_from", "pool_from"]).cumsum(dim="time")
    Y_S = Q_(Y_S.data, ds.stocks.attrs["units"]).to("kgC/m^2").magnitude

    Y_L = ds.internal_fluxes.sel(pool_to="WP_L").sum(dim=["entity_to", "entity_from", "pool_from"]).cumsum(dim="time")
    Y_L = Q_(Y_L.data, ds.stocks.attrs["units"]).to("kgC/m^2").magnitude

    Y = Y_S + Y_L
    
    INCB = (ds.input_fluxes.sum(dim=["entity_to", "pool_to"]) - ds.output_fluxes.sum(dim=["entity_from", "pool_from"])).cumsum()
    INCB = Q_(INCB.data * 1e-03, "kgC/m^2").magnitude

    try:
        dmr = utils.create_dmr_from_stocks_and_fluxes(ds, GPP_total_prepend=ds_long.GPP_total[start_year-1])
        dmr.initialize_state_transition_operator_matrix_cache(50_000)

        IITT = Q_(dmr.CS_through_time(0) * 1e-03, "yr kgC/m^2")
        del dmr._state_transition_operator_matrix_cache
    except DMRError as e:
        print("start_year =", start_year)
        print(e)
        IITT = [Q_(np.nan, "yr kgC/m^2").magnitude] * len(INCB)
        
    ICS = Q_(ds.stocks.sum(dim=["pool", "entity"]).cumsum().data * 1e-03, "yr kgC/m^2").magnitude
    
    return [Metrics(start_year, year, *el) for year, el in enumerate(zip(Y_S, Y_L, Y, INCB, IITT, ICS))]
#    return [Metrics(start_year, year, *el) for year, el in enumerate(zip(Y_S, Y_L, Y, INCB, ICS))]



# +
# %%time

window = range(80, 80+81)

dfs = []
for sim_name in sim_names:
    print(sim_name)
    ds_long=dss_long[sim_name]
    metrics = [compute_metrics(ds_long, start_year, nr_years) for start_year in tqdm(list(window))]
    df = pd.concat([pd.DataFrame(metric) for metric in metrics]).reset_index(drop=True)
    df["start_year"] = df["start_year"] - 80
    df.insert(0, "sim_name", sim_name)
    dfs.append(df)
    
df = pd.concat(dfs)
# -

df.to_csv("transition.csv")



df = pd.read_csv("transition.csv")

df

df["IITT"] = df["IITT"].apply(lambda q_str: Q_(q_str).magnitude)

df1 = df[df["sim_name"] == "BAU"] # business as usual
df2 = df[df["sim_name"] != "BAU"] # the transition simulation


# ## Plots
#
# In the following plots, "start_year" means how long we have to wait until the clea-cut rotation is finished. Then we start the transition. BAU is business as usual, continue with clear-cut even-aged pine and a otation time of 80 years.
#
# Transition means that at some point, we start to change to establiosh a CCF.

df1.groupby("start_year")["Y"].max()

# We assume that we have spatially distributed stands at all ages. So in different locations we start thetransition at different times, depending on the stand age. If the clear-cut stand age is 30, then we start the transition in 50 years. We let the clear-cut rotation finish.
#
# The next plots show the mean over all locations (= mean over all current stand ages).

# +
variable_names_tmp = ["Y", "INCB", "ICS"]
titles = ["Mean annual wood production", "Mean annual net C balance", "Mean annual climate change mitigation potential"]
y_labels = ["kgC / m$^2$ / yr", "kgC / m$^2$", "kgC / m$^2\,\cdot$ yr"]

n = len(variable_names_tmp)
fig, axes = plt.subplots(figsize=(12, 6*n), nrows=n)

for variable_name, ax , title, y_label in zip(variable_names_tmp, axes, titles, y_labels):
    ax.set_title(title)
    df1.set_index("year").groupby("start_year")[variable_name].mean().plot(ax=ax, label="BAU", legend=True)
    df2.set_index("year").groupby("start_year")[variable_name].mean().plot(ax=ax, label="transition", legend=True)
    ax.set_xlabel("start year of transition")
    ax.set_ylabel(y_label)
#    ax.set_title("$" + variable_name + "$")
    
fig.tight_layout()
# -

# The x-axis tells the time from now until we start the transition. It depends on the stand age, basically it is 80 - current stand age.
#
#

plot_times = [0, 20, 40, 60, 80]

# +
n = len(variable_names)
fig, axes = plt.subplots(figsize=(12, 6*n), nrows=n)

for variable_name, ax in zip(variable_names, axes):
    for y0 in plot_times:
        df1[df1["start_year"] == y0].set_index("year")[variable_name].plot(ax=ax, label=str(y0), legend=True)
        ax.set_title(variable_name)

fig.suptitle("BAU")
fig.tight_layout()

# +
n = len(variable_names)
fig, axes = plt.subplots(figsize=(12, 6*n), nrows=n)

for variable_name, ax in zip(variable_names, axes):
    for y0 in plot_times:
        df2[df2["start_year"] == y0].set_index("year")[variable_name].plot(ax=ax, label=str(y0), legend=True)
        ax.set_title(variable_name)

fig.suptitle("transition")
fig.tight_layout()
# -

# The next plots tell us, which clear-cut stand age is the best to start a transition! 
# It looks like a completely young is not a good starting moment and an old one neither.
# In the middle at 40 yrs it's a good start.
#
# In terms of ICS it is the least bad. But only on the randomly set time horizon.

# +
n = len(variable_names)
fig, axes = plt.subplots(figsize=(12, 6*n), nrows=n)

for variable_name, ax in zip(variable_names, axes):
    for y0 in plot_times:
        (df2[df2["start_year"] == y0].set_index("year")[variable_name] - df1[df1["start_year"] == y0].set_index("year")[variable_name]).plot(ax=ax, label=str(y0), legend=True)
        ax.set_title(variable_name)

fig.suptitle("Transition - BAU")
fig.tight_layout()
# -



# ## Theme or title of the abstract
#
#
# Transition from clear-cut forestry to CCF clashes wih short-term climate goals


