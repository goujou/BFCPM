# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
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

# # Figures for the manuscript

# %load_ext autoreload

# +
import string
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import xarray as xr
from tqdm import tqdm
from pathlib import Path

from LAPM.discrete_linear_autonomous_pool_model import DiscreteLinearAutonomousPoolModel as DLAPM
from CompartmentalSystems.discrete_model_run import DiscreteModelRun as DMR
import CompartmentalSystems.helpers_reservoir as hr

from BFCPM import utils
from BFCPM import DATA_PATH, FIGS_PATH, Q_, zeta_dw
from BFCPM.trees.single_tree_allocation import allometries
from BFCPM.trees.single_tree_params import species_params

# %autoreload 2


# +
# set plotting properties

mpl.rcParams['lines.linewidth'] = 2

SMALL_SIZE = 16
MEDIUM_SIZE = 17
#BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# -

# ## Load available simulations and sort them

# +
all_sims_path = DATA_PATH.joinpath("simulations")

#sim_date = "2023-05-20" # differenct emergency strategies
#sim_date = "2023-06-08" # corrected wood density
#sim_date = "2023-06-19" # automatic thinning stand to SBA=18 on emergency cutting
#sim_date = "2023-07-26" # publication
sim_date = "2023-11-23" # WP spread

# path to place the figures for the publication
#pub_figs_path = FIGS_PATH.joinpath(f"{sim_date}")
#pub_figs_path.mkdir(exist_ok=True, parents=True)

#print(pub_figs_path)

spinup_length = 160
# -


WP_types = ["WP_short_only", "WP_both", "WP_long_only"]

# reconcile filenames and simulation names
eligible_sim_names = {
    "mixed-aged_pine_N1500": "mixed-aged_pine",
    "even-aged_pine": "even-aged_pine",
    "even-aged_spruce": "even-aged_spruce",
    "even-aged_mixed": "even-aged_mixed",
}

# +
sims_data = dict()
for WP_type in WP_types:
    dss = dict()
    dmrs = dict()
    dmrs_eq = dict()
    dss_long = dict()

    sim_cohort_path = all_sims_path.joinpath(f"{sim_date}_{WP_type}")
    sim_names = list()

    for p in sim_cohort_path.iterdir():
        if (p.suffix == ".nc") and (p.stem.find("_long") == -1):
            file_sim_name = p.stem
            if file_sim_name in eligible_sim_names.keys():
                sim_name = eligible_sim_names[file_sim_name]
                print(WP_type, len(sim_names), sim_name)
                sim_names.append(sim_name)
                dss[sim_name] = xr.open_dataset(str(p))
        
                dmr_path = sim_cohort_path.joinpath(p.stem + ".dmr")
                dmrs[sim_name] = DMR.load_from_file(dmr_path)
    
                dmr_eq_path = sim_cohort_path.joinpath(p.stem + ".dmr_eq")
                dmrs_eq[sim_name] = DLAPM.load_from_file(dmr_eq_path)
        
                ds_long_path = sim_cohort_path.joinpath(p.stem + "_long.nc")
                dss_long[sim_name] = xr.open_dataset(str(ds_long_path))

#                print(dss[sim_name].stocks.sel(entity="wood_product").sum(dim="pool").data[-1])
#                print(dss[sim_name].stocks.sum(dim=["entity", "pool"]).data[0])
#                print(dss[sim_name].stocks.sum(dim=["entity", "pool"]).data[-1])

    sim_data = dict()
    sim_data["dss"] = dss
#    sim_data["dmrs"] = dmrs
#    sim_data["dmrs_eq"] = dmrs_eq
#    sim_data["dss_long"] = dss_long

    sims_data[WP_type] = sim_data


# +
sim_names = [sim_names[k] for k in [3, 1, 2, 0]]

nr_spinup_trees = 4
[print(f"{k}: {sim_name}") for k, sim_name in enumerate(sim_names)];
# -
# ## Carbon sequestration and climate change mitigation

# +
CS_datas = list()
for sim_name in sim_names:
    l = list()
    for WP_type in WP_types:
        ds = sims_data[WP_type]["dss"][sim_name]
        CS = ds.CS_through_time#.isel(time=-1)
        CS = CS.data # gC/m^2 yr, CS.attrs["units"]).to("yr*kgC/m^2")
        l.append(CS)
        
    CS_datas.append(l)
    
CS_datas = Q_(np.array(CS_datas).transpose(), "yr*gC/m^2").to("yr*kgC/m^2")
CS_datas.shape

# +
CB_datas = list()

#ti = -2

for sim_name in sim_names:
    l = list()
    for WP_type in WP_types:
        ds = sims_data[WP_type]["dss"][sim_name]
        x0 = ds.stocks.isel(time=0).sum(dim=["entity", "pool"])
        CB = ds.stocks.sum(dim=["entity", "pool"]) - x0
        CB = CB.data # gC/m^2
        l.append(CB)
      
    CB_datas.append(l)
    
CB_datas = Q_(np.array(CB_datas).transpose(), "gC/m^2").to("kgC/m^2")
CB_datas.shape

# +
cum_stocks_datas = list()
for sim_name in sim_names:
    l = list()
    for WP_type in WP_types:
        ds = sims_data[WP_type]["dss"][sim_name]
        cum_stocks = ds.stocks.sum(dim=["entity", "pool"]).cumsum(dim="time")
        cum_stocks = cum_stocks.data # gC/m^2 yr, ds.stocks.attrs["units"]) * Q_("1 yr")).to("kgC/m^2 yr")
        l.append(cum_stocks)

    cum_stocks_datas.append(l)
    
cum_stocks_datas = Q_(np.array(cum_stocks_datas).transpose(), "yr*gC/m^2").to("yr*kgC/m^2")
cum_stocks_datas.shape

# +
yield_datas = list()
for sim_name in sim_names:
    l = list()
    for WP_type in WP_types:
        ds = sims_data[WP_type]["dss"][sim_name]
        WP_cum = ds.internal_fluxes.sel(entity_to="wood_product").sum(dim=["pool_to", "entity_from", "pool_from"]).cumsum(dim="time")
        l.append(WP_cum)
        
    yield_datas.append(l)

yield_datas = Q_(np.array(yield_datas).transpose(), "gC/m^2").to("kgC/m^2")
yield_datas.shape

# +
fig, axes_ = plt.subplots(figsize=(8, 4*4), nrows=4)
panel_names = iter(string.ascii_uppercase[:len(axes_)])
axes = iter(axes_)

labels = [sim_name.replace("_", " ") for sim_name in sim_names]
hatches = ["/", "x", "\\"]
width = 0.25
x = np.arange(len(sim_names))

ax = next(axes)
panel_name = next(panel_names)
multiplier = 0

for WP_nr, WP_type in enumerate(WP_types):
    offset = width * multiplier
    rects = ax.bar(
        x + offset,
        yield_datas[-1, WP_nr, :], 
        width,
        color=[colors[sim_name] for sim_name in sim_names],
        edgecolor="black", hatch=hatches[WP_nr],
        label=WP_type.replace("_", " ")
    )
    multiplier += 1
                   
ax.set_ylabel(r"kgC$\,$m$^{-2}$")
ax.set_ylim([0, 16])
ax.legend(loc=4)
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')
ax.set_title("Total cumulative wood-product yield C ($Y_S+Y_L$)")
ax.set_xticks([])
ax.set_xticklabels("")

ax = next(axes)
panel_name = next(panel_names)
multiplier = 0

for WP_nr, WP_type in enumerate(WP_types):
    offset = width * multiplier
#    rects = ax.bar(x + offset, CB_datas[-1, WP_nr, :], width, label=WP_type.replace("_", " "))
    rects = ax.bar(
        x + offset,
        CB_datas[-1, WP_nr, :], 
        width,
        color=[colors[sim_name] for sim_name in sim_names],
        edgecolor="black", hatch=hatches[WP_nr],
        label=WP_type.replace("_", " ")
    )
    multiplier += 1

ax.set_ylabel(r"kgC$\,$m$^{-2}$")
ax.set_ylim([-2, 6])
ax.set_title("Integrated Net Carbon Balance (INCB)")
ax.axhline(0, c="black", lw=1)
ax.set_xticks([])
ax.set_xticklabels("")
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')

ax = next(axes)
panel_name = next(panel_names)
multiplier = 0

for WP_nr, WP_type in enumerate(WP_types):
    offset = width * multiplier
#    rects = ax.bar(x + offset, CS_datas[-1, WP_nr, :], width, label=WP_type.replace("_", " "))
    rects = ax.bar(
        x + offset,
        CS_datas[-1, WP_nr, :], 
        width,
        color=[colors[sim_name] for sim_name in sim_names],
        edgecolor="black", hatch=hatches[WP_nr],
        label=WP_type.replace("_", " ")
    )
    multiplier += 1

ax.set_ylabel(r"kgC$\,$m$^{-2}\,$yr")
ax.set_ylim([0, 800])
ax.set_title("Integrated Inputs Transit Time (IITT)")
ax.set_xticks([])
ax.set_xticklabels("")
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')

ax = next(axes)
panel_name = next(panel_names)
multiplier = 0

for WP_nr, WP_type in enumerate(WP_types):
    offset = width * multiplier
#    rects = ax.bar(x + offset, cum_stocks_datas[-1, WP_nr, :], width, label=WP_type.replace("_", " "))
    rects = ax.bar(
        x + offset,
        cum_stocks_datas[-1, WP_nr, :], 
        width,
        color=[colors[sim_name] for sim_name in sim_names],
        edgecolor="black", hatch=hatches[WP_nr],
        label=WP_type.replace("_", " ")
    )
    multiplier += 1

ax.set_ylabel(r"kgC$\,$m$^{-2}\,$yr")
ax.set_ylim([0, 1500])
ax.set_title("Integrated Carbon Stocks (ICS)")
ax.set_xticklabels("")
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')

for tick in ax.get_xticklabels():
    tick.set_rotation(60)

ax.set_xticks(x + width, [sim_name.replace("_", " ") for sim_name in sim_names])
fig.tight_layout()

leg = axes_[0].get_legend()
for lh in leg.legend_handles:
    lh.set_color("white")
    lh.set_edgecolor("black")


# +
fig, axes = plt.subplots(figsize=(8, 4), ncols=2)
panel_names = iter(string.ascii_uppercase[:len(axes)])
axes = iter(axes)

ax = next(axes)
panel_name = next(panel_names)
#ax.set_title("Integrated Inputs Transit Time (IITT)")
ax.set_title("IITT spread")
#for WP_type_nr, WP_type in enumerate(WP_types):
width = 0.25
for nr, sim_name in enumerate(sim_names):   
#    ax.plot(CS_datas[:, WP_type_nr, nr], label=sim_name.replace("_", " "), c=colors[sim_name])
    bottom = CS_datas[-1, 0, nr]
    ax.bar(nr*width, height = CS_datas[-1, 2, nr] - bottom, bottom=bottom, width=width*0.8, label=sim_name.replace("_", " "))
#    print([nr*width-width/2, nr*width+width/2])
#    print([CS_datas[-1, 1, nr].magnitude]*2)
    ax.plot([nr*width-width/2*0.8, nr*width+width/2*0.8], [CS_datas[-1, 1, nr].magnitude]*2, c="black")

ax.set_ylim([450, 650])
ax.set_ylabel(r"kgC$\,$m$^{-2}\,$yr")
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')
ax.set_xticks([])
ax.legend(loc=4, fontsize=12)


ax = next(axes)
panel_name = next(panel_names)
#ax.set_title("Integrated Inputs Transit Time (IITT)")
ax.set_title("ICS spread")
#for WP_type_nr, WP_type in enumerate(WP_types):
width = 0.25
for nr, sim_name in enumerate(sim_names):   
#    ax.plot(cum_stocks_datas[:, WP_type_nr, nr], label=sim_name.replace("_", " "), c=colors[sim_name])
    bottom = cum_stocks_datas[-1, 0, nr]
    ax.bar(nr*width, height = cum_stocks_datas[-1, 2, nr] - bottom, bottom=bottom, width=width*0.8, label=sim_name.replace("_", " "))
#    print([nr*width-width/2, nr*width+width/2])
#    print([cum_stocks_datas[-1, 1, nr].magnitude]*2)
    ax.plot([nr*width-width/2*0.8, nr*width+width/2*0.8], [cum_stocks_datas[-1, 1, nr].magnitude]*2, c="black")

ax.set_ylim([750, 1300])
#ax.set_ylabel(r"kgC$\,$m$^{-2}\,$yr")
ax.set_ylabel("")
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')
ax.set_xticks([])

fig.tight_layout()
# -


