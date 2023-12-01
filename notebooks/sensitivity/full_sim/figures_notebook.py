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

# # Sensitivity data table to copy to latex file
#
# Needs slight adaptation only.

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
import pandas as pd

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


spinup_length = 160

#sim_cohort_path = all_sims_path.joinpath(sim_date)
sim_cohort_path = all_sims_path.joinpath("sensitivity_full_sim")

print(sim_cohort_path)


# -


# ## Check if the simulations have a proper continuous-cover spinup
# This means, all trees survived and where cut only following the schedule.

def check_proper_cc_spinup(ds):
    proper_years = {
        0: [20, 100],
        1: [40, 120],
        2: [60, 140],
        3: [80, 160]
    }

    tree_nrs, yrs = np.where(ds.thinning_or_cutting_tree.data[:, :4]==1)
    for tree_nr, yr in zip(tree_nrs, yrs):
        if yr <= 160:
            if yr not in check_proper_cc_spinup[tree_nr]:
                return False

    return True


def get_par_name_and_q(s):
    par_names = ["rho_RL", "R_mL", "S_R", "Vcmax"]
    for par_name in par_names:
        if s.find(par_name) == -1:
            continue
    
        q_s = s.split(par_name + "_")[1]
        q = float(q_s[0] + "." + q_s[1:])

        return par_name, q


# reconcile filenames and simulation names
eligible_sim_names = {
    "mixed-aged_pine_N1500": "mixed-aged_pine",
    "even-aged_pine": "even-aged_pine",
    "even-aged_spruce": "even-aged_spruce",
    "even-aged_mixed": "even-aged_mixed",
}

# +
col_names = [
    "sim_name", "proper_cc_spinup", "par_name", "q",
    "total_C",
    "WP_cum", "WPS_cum", "WPL_cum",
    "INCB", "IITT", "ICS"
]
d = {col_name: list() for col_name in col_names}

for p in sim_cohort_path.glob("**/*.nc"):
    if p.stem.find("_long") != -1:
#        print(p.stem, p.parent.parent.name)
        ds_long = xr.load_dataset(str(p))

        file_sim_name = p.stem.replace("_long", "")
        d["sim_name"].append(eligible_sim_names[file_sim_name])

        d["proper_cc_spinup"].append(check_proper_cc_spinup(ds_long))
        par_name, q = get_par_name_and_q(p.parent.parent.name)
        d["par_name"].append(par_name)
        d["q"].append(q)

        ds = xr.open_dataset(str(p.parent.joinpath(file_sim_name + ".nc")))

        d["total_C"].append(ds.stocks.sum(dim=["entity", "pool"]).data[-1] * 1e-03)
        
        d["WPS_cum"].append(ds.internal_fluxes.sel(pool_to="WP_S").sum(dim=["entity_to", "entity_from", "pool_from"]).cumsum(dim="time").data[-1] * 1e-03)
        d["WPL_cum"].append(ds.internal_fluxes.sel(pool_to="WP_L").sum(dim=["entity_to", "entity_from", "pool_from"]).cumsum(dim="time").data[-1] * 1e-03)
        d["WP_cum"].append(d["WPS_cum"][-1] + d["WPL_cum"][-1])

        x0 = ds.stocks.isel(time=0).sum(dim=["entity", "pool"])
        d["INCB"].append((ds.stocks.sum(dim=["entity", "pool"]) - x0).data[-1] * 1e-03)
        d["IITT"].append(ds.CS_through_time.data[-1] * 1e-03)
        d["ICS"].append(ds.stocks.sum(dim=["entity", "pool"]).cumsum(dim="time").data[-1] * 1e-03)

df = pd.DataFrame(d)
df
# -

sim_names = ["mixed-aged_pine", "even-aged_pine", "even-aged_spruce", "even-aged_mixed"]
df.sort_values(by="sim_name", key=lambda xs: [sim_names.index(x) for x in xs]).reset_index(drop=True)


# +
def SEM(x):
    return x.sem()

def RS(x):
    return np.ptp(x) / np.mean(x) * 100


# -

pd.options.display.float_format = '{:,.2f}'.format
df_sens = df.groupby(by=["par_name", "sim_name"])[["total_C", "WP_cum", "IITT", "ICS"]].agg(RS)#.reset_index()
df_sens = df_sens.reindex(sim_names, level="sim_name")
#df_sens.columns = ["Parameter", "Scenario", "total carbon", "cum. WP", "IITT", "ICS"]
df_sens#.columns

# +
s = df_sens.to_latex(float_format=lambda x: '%10.2f' % x)
s = s.replace("par_name", r"\thead{Parameter}").replace("sim_name", r"\thead{Scenario}")
s = s.replace("\\toprule", "").replace("\\midrule", "").replace("\\bottomrule", "")
s = s.replace("total_C", r"\thead{total stock}").replace("WP_cum", r"\thead{cum. WP}")
s = s.replace("IITT", r"\thead{IITT}").replace("ICS", r"\thead{ICS}")
s = s.replace("R_mL", r"$R_{\text{mL}}$").replace("S_R", r"$S_R$").replace("Vcmax", r"$V_{cmax,25}$").replace("rho_RL", r"$\rho_{\text{RL}}$")
s = s.replace("aged_", "aged ")

s = s.replace("mixed-aged pine", r"mixed-aged pine${}^1$", 1)
s = s.replace("even-aged pine", r"mixed-aged pine${}^2$", 1)
s = s.replace("even-aged spruce", r"mixed-aged pine${}^3$", 1)
s = s.replace("even-aged mixed", r"mixed-aged pine${}^4$", 1)

s = s.replace("19.95", r"19.95${}^1$")
s = s.replace("17.27", r"17.27${}^2$")
s = s.replace("32.40", r"32.40${}^3$")
s = s.replace("12.69", r"12.69${}^4$")

s = s.replace("11.43", r"\textbf{11.43}").replace("30.36", r"\textbf{30.36}").replace("19.95", r"\textbf{19.95}").replace("32.40", r"\textit{\textbf{32.40}}")
s = s.replace("18.15", r"\textit{18.15}").replace("18.19", r"\textit{18.19}").replace("19.93", r"\textit{19.93}")

print(s)
# -


