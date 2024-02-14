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

# # Analyse a simulation

# %load_ext autoreload

# +
#import string
import matplotlib as mpl
import matplotlib.pyplot as plt
#from matplotlib.lines import Line2D
import numpy as np
import xarray as xr
#from tqdm import tqdm
from pathlib import Path

#from LAPM.discrete_linear_autonomous_pool_model import DiscreteLinearAutonomousPoolModel as DLAPM
#from CompartmentalSystems.discrete_model_run import DiscreteModelRun as DMR
#import CompartmentalSystems.helpers_reservoir as hr

#from ACGCA import utils
from BFCPM.__init__ import PRE_SPINUPS_PATH as all_sims_path
#from ACGCA.alloc.ACGCA_marklund_tree import allometries
#from ACGCA.alloc.ACGCA_marklund_tree_params import species_params

# %autoreload 2


# +
# set plotting properties

mpl.rcParams['lines.linewidth'] = 2

SMALL_SIZE = 16
MEDIUM_SIZE = 17
#BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
# -

# ## Load available simulations

# +
sim_date = "2023-10-18" # earlier tree death
sim_cohort_path = all_sims_path.joinpath(sim_date)
sim_names = ["DWC_Zhao_spruce"]#, "DWC_Zhao_pine_12"]

sim_cohort_path
# -


dss = dict()
#dmrs = dict()
#dmrs_eq = dict()
#dss_sim = dict()
#dmrs_sim = dict()

for sim_name in sim_names:
    nc_path = sim_cohort_path.joinpath(sim_name + ".nc")
    ds = xr.open_dataset(str(nc_path))
    dss[sim_name] = ds

ds = dss[sim_names[0]]
ds

# +
var_names = ["GPP_year", "GPP_total", "NPP", "stand_basal_area", "mean_tree_height", "dominant_tree_height", "total_C_stock", "tree_biomass"]
fig, axes = plt.subplots(figsize=(12, 4*len(var_names)), nrows=len(var_names))

for var_name, ax in zip(var_names, axes):
    for sim_name in sim_names:
        ds = dss[sim_name]
        ds[var_name].plot(ax=ax, label=sim_name)

axes[0].legend()
fig.tight_layout();
# -

n = 80
B_TS = ds.stocks.isel(entity=0).sel(pool="B_TS")[:n]
B_TH = ds.stocks.isel(entity=0).sel(pool="B_TH")[:n]
B_T = B_TS + B_TH

B_TS.plot(label="B_TS")
B_TH.plot(label="B_TH")
B_T.plot(label="B_T")
plt.legend()

B_TS.diff(dim="time").rolling(time=10).mean().plot(label="B_TS")
B_TH.diff(dim="time").rolling(time=10).mean().plot(label="B_TH")
B_T.diff(dim="time").rolling(time=10).mean().plot(label="B_T")
plt.legend()

# +
us = ds.internal_fluxes.isel(entity_to=0, entity_from=0).sel(pool_to="B_TS").sum(dim="pool_from")[:n]
phis = us / B_TS
phis[np.isnan(phis)] = 0
phis[np.isinf(phis)] = 0

B_TS_to_B_TH = ds.internal_fluxes.isel(entity_to=0, entity_from=0).sel(pool_to="B_TH", pool_from="B_TS")[:n]
rs = B_TS_to_B_TH / B_TS
rs[np.isnan(rs)] = 0
rs[np.isinf(rs)] = 0
# -

phis.plot(label="u")
rs.plot(label="r")
plt.legend()

(phis-rs).plot()

plt.plot(B_TS, phis-rs)

plt.plot(ds.time[:80], phis-rs, label=r"$\varphi-z$")
plt.legend()


x_min, x_max = np.min(B_TS.data), np.max(B_TS.data)
x_min, x_max

from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

ts = ds.time.data[:n]
phi_x = interp1d(B_TS.data, phis.data, bounds_error=False, fill_value=(phis[0], phis[-1]))#, kind="previous")
r_x = interp1d(B_TS.data, rs.data, bounds_error=False, fill_value=(rs[0], rs[-1]))#, kind="previous")
f_x_phi_minus_r = interp1d(B_TS.data, (phis - rs).data, bounds_error=False, fill_value=((phis-rs)[0], (phis-rs)[-1]))#, kind="previous")

plt.plot(B_TS, phis-rs, label=r"{\varphi(x)-r(x)$")
plt.plot(B_TS, f_x_phi_minus_r(B_TS.data), ls="--")
plt.plot(B_TS, phi_x(B_TS) - r_x(B_TS), ls="-.")


def interp(f_x_template, f_source, distance, tup0, bounds, constraints=None):
    def deviation(*args):
        f_ = f_x_template(*args)
        return np.sum(distance(f_(B_TS.data), f_source(B_TS.data)))

    def g(tup):
        return deviation(*tup)
        
    res = minimize(g, x0=tup0, bounds=bounds, constraints=constraints)

    def func_maker(x):
        return f_x_template(*x)
    
    f = func_maker(res.x)
    return f


# +
f_x_template = lambda a, k, x0: lambda x: a * np.exp(-k*(x-x0))
#tup0 = np.array([0.5, 0.0001, 0.0]) # pine
tup0 = np.array([0.3, 0.001, 0.0]) # spruce
bounds = [(0, 1), (0, 1), (-x_max, x_max)]
f_x_exp = interp(f_x_template, f_x_phi_minus_r, lambda x, y: np.abs(x-y), tup0, bounds)

f_x_template = lambda a, b: lambda x: a - b*x
tup0 = np.array([0.3, 0.3*1/5_000]) # pine
bounds = [(0, 1), (0, 1)]
constraints = [{"type": "ineq", "fun": lambda x: x[0] - x_max * x[1]}]
f_x_lin = interp(f_x_template, f_x_phi_minus_r, lambda x, y: np.abs(x-y)**(1/1), tup0, bounds=bounds, constraints=constraints)
# -

plt.plot(B_TS, phis - rs, label=r"$\varphi(x)-r(x)$")
plt.plot(B_TS, f_x_exp(B_TS), ls="--", label="exponential approximation")
plt.plot(B_TS, f_x_lin(B_TS), ls="-.", label="linear approximation")
#plt.plot(B_TS, 0.3 * np.exp(-0.001 * B_TS))
plt.legend()

plt.plot(B_TS, phis - rs, label=r"$\varphi(x)-z(x)$")
plt.plot(B_TS, f_x_exp(B_TS), ls="--", label="exponential approximation")
#plt.plot(B_TS, f_x_lin(B_TS), ls="-.", label="linear approximation")
#plt.plot(B_TS, 0.3 * np.exp(-0.001 * B_TS))
plt.xlabel("Stem carbon")
plt.legend()

plt.plot(B_TS, phis, label=r"$\varphi(x)")
plt.xlabel("Stem carbon")
plt.legend()


# +
def make_f_t(f_x):
    def g(t, x):
        return f_x(x) * x   

    res = solve_ivp(g, t_span=(ts[0], ts[-1]), y0=np.array(B_TS[5]).reshape(-1), dense_output=True)
    return lambda t: res.sol(t-5).reshape(-1)

f_t = make_f_t(f_x_phi_minus_r)
f_t_exp = make_f_t(f_x_exp)
f_t_lin = make_f_t(f_x_lin)


# +
#B_TS.plot(label="B_TS")
plt.plot(ts, B_TS, label="alive stem carbon")
#plt.plot(ts, f_t(ts), label="numerical original")
#if x0 >= 0:
#    label = r"$\varphi(x)-r(x) \approx " + f"{round(a, 1)}" + r"\cdot\mathrm{exp}^{-" + f"{round(k, 4)}" + r"\,(x-" + f"{round(x0, 4)})" + r"}$"
#else:
#    label = r"$\varphi(x)-r(x) \approx " + f"{round(a, 1)}" + r"\cdot\mathrm{exp}^{-" + f"{round(k, 4)}" + r"\,(x+" + f"{-round(x0, 4)})" + r"}$"
label = "exponential approximation"
plt.plot(ts, f_t_exp(ts), ls="--", label=label)

#label = r"$\varphi(x)-r(x) \approx" + f"{round(n,2)}" + r"-" + f"{round(m,5):2f}" + "\cdot x$"
#label = "linear approximation"
#plt.plot(ts, f_t_lin(ts), ls="-.", label=label)
plt.legend(loc=4)

# +
phi_x_template = lambda a, k, x0: lambda x: a * np.exp(-k*(x-x0))
tup0 = np.array([0.5, 0.0001, 0.0])
bounds = [(0, 1), (0, 1), (-x_max, x_max)]
phi_x_exp = interp(phi_x_template, phi_x, lambda x, y: np.abs(x-y), tup0, bounds)

phi_x_template = lambda a, b: lambda x: a - b*x
tup0 = np.array([0.3, 0.3*1/5_000])
bounds = [(0, 1), (0, 1)]
constraints = [{"type": "ineq", "fun": lambda x: x[0] - x_max * x[1]}]
phi_x_lin = interp(phi_x_template, phi_x, lambda x, y: np.abs(x-y)**2, tup0, bounds=bounds, constraints=constraints)
# -

plt.plot(B_TS, phi_x(B_TS), label=r"$\varphi(x)$")
plt.plot(B_TS, phi_x_exp(B_TS), ls="--", label="exponential approximation")
plt.plot(B_TS, phi_x_lin(B_TS), ls="-.", label="linear approximation")
plt.legend()

# +
r_x_template = lambda a, k, x0: lambda x: a * np.exp(-k*(x-x0))
tup0 = np.array([0.2, 0.001, 0.0])
bounds = [(0, 1), (0, 1), (-x_max, x_max)]
r_x_exp = interp(r_x_template, r_x, lambda x, y: np.abs(x-y), tup0, bounds)

r_x_template = lambda a, b: lambda x: a - b*x
tup0 = np.array([0.15, 0.15*1/5_000])
bounds = [(0, 1), (0, 1)]
constraints = [{"type": "ineq", "fun": lambda x: x[0] - x_max * x[1]}]
r_x_lin = interp(r_x_template, r_x, lambda x, y: np.abs(x-y), tup0, bounds=bounds, constraints=constraints)
# -

plt.plot(B_TS, rs, label=r"$r(x)$")
plt.plot(B_TS, r_x_exp(B_TS), ls="--", label="exponential approximation")
plt.plot(B_TS, r_x_lin(B_TS), ls="-.", label="linear approximation")
plt.axhline(0, c="black", lw=1)
plt.legend()


# +
def make_f_t(phi_x, r_x):
    def g(t, x):
        return (phi_x(x) - r_x(x)) * x   

    res = solve_ivp(g, t_span=(ts[0], ts[-1]), y0=np.array(B_TS[5]).reshape(-1), dense_output=True)
    return lambda t: res.sol(t-5).reshape(-1)

f_t = make_f_t(phi_x, r_x)
f_t_exp = make_f_t(phi_x_exp, r_x_exp)
f_t_lin = make_f_t(phi_x_lin, r_x_lin)


# +
#B_TS.plot(label="B_TS")
plt.plot(ts, B_TS, label="B_TS")
plt.plot(ts, f_t(ts), label="numerical original")
#if x0 >= 0:
#    label = r"$\varphi(x)-r(x) \approx " + f"{round(a, 1)}" + r"\cdot\mathrm{exp}^{-" + f"{round(k, 4)}" + r"\,(x-" + f"{round(x0, 4)})" + r"}$"
#else:
#    label = r"$\varphi(x)-r(x) \approx " + f"{round(a, 1)}" + r"\cdot\mathrm{exp}^{-" + f"{round(k, 4)}" + r"\,(x+" + f"{-round(x0, 4)})" + r"}$"
label = "exponential approximation"
plt.plot(ts, f_t_exp(ts), ls="--", label=label)

#label = r"$\varphi(x)-r(x) \approx" + f"{round(n,2)}" + r"-" + f"{round(m,5):2f}" + "\cdot x$"
label = "linear approximation"
plt.plot(ts, f_t_lin(ts), ls="-.", label=label)
plt.legend()
# -

plt.plot(B_TS, phi_x_lin(B_TS) - r_x_lin(B_TS))
plt.plot(B_TS, f_x_lin(B_TS))


def make_f_a_t_and_g_a_t(f_t, phi_x):
    def g_a(t, a):
        x = f_t(t)
        return 1 - a * phi_x(x) 

    a0 = 0.
    a0 = np.array(a0).reshape((1,))
    res = solve_ivp(g_a, t_span=(0, ts[-1]), y0=a0, dense_output=True)

    f_a_t = lambda t: res.sol(t).reshape(-1)
    g_a_t = lambda t: g_a(t, f_a_t(t))

    return f_a_t, g_a_t


f_a_t, g_a_t = make_f_a_t_and_g_a_t(f_t, phi_x)
f_a_t_exp, g_a_t_exp = make_f_a_t_and_g_a_t(f_t_exp, phi_x_exp)
f_a_t_lin, g_a_t_lin = make_f_a_t_and_g_a_t(f_t_lin, phi_x_lin)

plt.plot(ts, f_a_t(ts), label=r"$a(t)$")
plt.plot(ts, f_a_t_exp(ts), label="exponential approximation")
#plt.plot(ts, f_a_t_lin(ts), label="linear approximation")
plt.plot(ts, 1/phi_x_exp(f_t_exp(ts)), label="1/phi_x_exp")
plt.legend()

# a * phi < 1 means aging
plt.plot(ts, f_a_t_exp(ts) * phi_x_exp(f_t_exp(ts)), label=r"$a\,\varphi$, exponential approx")
plt.legend()

# a * phi < 1 means aging
plt.plot(ts, f_a_t_exp(ts), label="age of living stem carbon")
plt.plot(ts, ts, label="maximum possible age")
plt.legend()

plt.plot(ts, g_a_t_exp(ts), label="age increase of living stem carbon")
plt.axhline(0, color="black")
plt.legend()

ts_ = ts[1:]

plt.plot(ts_, np.diff(f_a_t(ts)), label=r"$\dot{a}(t)$")
plt.plot(ts_, np.diff(f_a_t_exp(ts)), label="exponential approximation")
#plt.plot(ts_, np.diff(f_a_t_lin(ts)), label="linear approximation")
plt.legend()





def round_arr(arr: np.ndarray, decimals: int) -> np.ndarray:
    """Round array to `decimals` decimals."""
    try:
        return round(float(arr), decimals)
    except TypeError:
        return np.array([round(x, decimals) for x in arr])


# +
key = sim_names[0]

dbhs = dss_sim[key].DBH.isel(time=0)
print("dbh:", round_arr(dbhs.data, 1), "cm")

Hs = dss_sim[key].height.isel(time=0)
print("H:", round_arr(Hs.data, 1), "m")

tree_biomasses = dss_sim[key].stocks.isel(entity=range(nr_spinup_trees), time=0).sum(dim="pool") * 1e-03
print("tree biomass:", round_arr(tree_biomasses.data, 3), "total:", round_arr(sum(tree_biomasses.data), 1), "kgC/m^2")

soil_biomass = dss_sim[key].stocks.sel(entity="soil").isel(time=0).sum(dim="pool") * 1e-03
print("soil biomass:", round_arr(soil_biomass.data, 1),"kgC/m^2")

forest_stand_biomass = tree_biomasses.sum() + soil_biomass
print("forest stand biomass:", round_arr(forest_stand_biomass.data, 1), "kgC/m^2")

WP_biomass = dss_sim[key].stocks.sel(entity="wood_product").isel(time=0).sum(dim="pool") * 1e-03
print("WP biomass:", round_arr(WP_biomass.data, 1), "kgC/m^2")

total_biomass = dss_sim[key].stocks.isel(time=0).sum(dim=["entity", "pool"]) * 1e-03
print("total biomass:", round_arr(total_biomass.data, 1), "kgC/m^2")
# -

# ## Show the mean diameter at breast height, the stand basal area, the tree C stock, and the stand's total C stock

# +
fig, axes = plt.subplots(figsize=(12, 4*7), nrows=7)
axes_list = axes

ax0 = axes[0]
axes = iter(axes)
panel_names = iter(string.ascii_uppercase[:len(axes_list)])
colors = dict()

ax = next(axes)
panel_name = next(panel_names)
ax.set_title("Mean diameter at breast height")
for sim_name in sim_names:
    ds = dss_sim[sim_name]
    tree_names = ds.entity.isel(entity=ds.tree_entity_nrs).data
    N_per_m2 = ds.N_per_m2.sel(tree=tree_names)
    
    DBH = ds.DBH.sel(tree=tree_names).weighted(N_per_m2).mean(dim="tree")
    DBH = Q_(DBH.data, ds.DBH.attrs["units"])
    l, = ax.plot(ds.time, DBH, label=sim_name.replace("_", " "))
    colors[sim_name] = (l.get_c())  

ax.set_ylabel("cm")
ax.legend()
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')

   
#ax = next(axes)
#panel_name = next(panel_names)
#ax.set_title("Mean tree height")
#for sim_name in sim_names:
#    ds = dss_sim[sim_name]
#    tree_names = ds.entity.isel(entity=ds.tree_entity_nrs).data
#    N_per_m2 = ds.N_per_m2.sel(tree=tree_names)
#    
#    H = ds.height.sel(tree=tree_names).weighted(N_per_m2).mean(dim="tree")
#    H = Q_(H.data, ds.DBH.attrs["units"])
#    l, = ax.plot(ds.time, H, label=sim_name.replace("_", " "))
#    colors[sim_name] = (l.get_c())  
#
#ax.set_ylabel("m")
#ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')

   
ax = next(axes)
panel_name = next(panel_names)
ax.set_title("Standing volume")
for sim_name in sim_names:
    ds = dss_sim[sim_name]
    tree_names = ds.entity.isel(entity=ds.tree_entity_nrs).data
    N_per_m2 = ds.N_per_m2.sel(tree=tree_names)
    
    standing_volume = ds.V_T_tree.sel(tree=tree_names).sum(dim="tree")
    standing_volume = Q_(standing_volume.data, ds.V_T_tree.attrs["units"]).to("m^3/ha")
    l, = ax.plot(ds.time, standing_volume, label=sim_name.replace("_", " "))
    colors[sim_name] = (l.get_c())  

ax.set_ylabel(r"m$^3\,$ha$^{-1}$")
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')

   
axes = iter(axes)
panel_names = iter(string.ascii_uppercase[:len(axes_list)])
colors = dict()

ax = next(axes)
panel_name = next(panel_names)
ax.set_title("Number of trees per hectare")
for sim_name in sim_names:
    ds = dss_sim[sim_name]
    tree_names = ds.entity.isel(entity=ds.tree_entity_nrs).data
    N_per_m2 = ds.N_per_m2.sel(tree=tree_names).sum(dim="tree")
    N_per_m2 = Q_(N_per_m2.data, "1/m^2").to("1/ha")
    l, = ax.plot(ds.time, N_per_m2.magnitude, label=sim_name.replace("_", " "))
    colors[sim_name] = (l.get_c())  

ax.set_ylabel("ha$^{-1}$")
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')

   
ax = next(axes)
panel_name = next(panel_names)
ax.set_title("Stand basal area")
for sim_name in sim_names:
    ds = dss_sim[sim_name]
    tree_names = ds.entity.isel(entity=ds.tree_entity_nrs).data
    
    sba = ds.stand_basal_area
    sba = Q_(sba.data, sba.attrs["units"])
    l, = ax.plot(ds.time, sba, label=sim_name.replace("_", " "))
    
       
ax.set_ylabel(r"m$^2\,$ha$^{-1}$")
#ax.set_ylim([0, 35])
#ax.set_xlim([0, 80])
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')
ax.axhline(18, color="black", alpha=0.2)
ax.axhline(25, color="black", alpha=0.2)


ax = next(axes)
panel_name = next(panel_names)
for sim_name in sim_names:
    ds = dss_sim[sim_name]
    var = ds.total_C_stock / 1000
    var.plot(ax=ax, label=sim_name.replace("_", " "))
    print(sim_name, "total C stock", var[-1].data, "gC/m^2")

ax.set_title("Total C stock")
ax.set_ylabel(r"kgC$\,$m$^{-2}$")
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')
#ax.set_ylim([7, 15])
#ax.set_ylim([0, ax.get_ylim()[1]])
ax.set_xlabel("")


ax = next(axes)
panel_name = next(panel_names)
ax.set_title("Total tree C")
for sim_name in sim_names:
    ds = dss_sim[sim_name]
    tree_names = ds.entity.isel(entity=ds.tree_entity_nrs).data
    
    tree_biomass = ds.tree_biomass_tree.sel(tree=tree_names).sum(dim="tree")
    tree_biomass = Q_(tree_biomass.data, ds.tree_biomass_tree.attrs["units"]).to("kgC/m^2")
    l, = ax.plot(ds.time, tree_biomass, label=sim_name.replace("_", " "))
    print("Total tree C (40 yr)", sim_name, tree_biomass[40])
    
ax.set_ylabel(r"kgC$\,$m$^{-2}$")
#ax.set_xlim([0, 80])
#ax.set_ylim([0, 10])
ax.set_ylim([0, ax.get_ylim()[-1]])
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')


ax = next(axes)
panel_name = next(panel_names)
for sim_name in sim_names:
    ds = dss_sim[sim_name]
    var = ds.stocks.sel(entity="soil").sum(dim="pool") / 1000
    var.plot(ax=ax, label=sim_name)

ax.set_title("Total soil C (Litter + CWD + SOM)")
ax.set_ylabel(r"kgC$\,$m$^{-2}$")
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')


ax0.legend()
#for ax in axes_list:
#    ax.set_xlim([0, sim_length])
    
ax.set_xlabel("time [yr]")
    
fig.tight_layout()

# save the figure for the publication
#filename = str(pub_figs_path.joinpath("dbh_sba_and_C_stocks.png"))
#fig.savefig(filename)
#filename
# -

# ## Box plots data

# +
stocks_datas = list()
for sim_name in sim_names:
    ds = dss_sim[sim_name]
    stocks = ds.total_C_stock.data # gC/m^2
    stocks_trees_and_soil = ds.total_C_stock - ds.stocks.sel(entity="wood_product").sum(dim="pool")
    data = np.array([stocks, stocks_trees_and_soil])
    stocks_datas.append(data)
    
stocks_datas = Q_(np.array(stocks_datas).transpose(), "gC/m^2").to("kgC/m^2")
stocks_datas.shape


# +
# %%time

# using the CS from the data set does not work,because we just cut CS out
# of a longer ds, and then we get legacy effects

CS_datas = list()
for sim_name in sim_names:
#    dmr_sim = dmrs_sim[sim_name]
#    CS = dmr_sim.CS_through_time(0, verbose=True)
    ds_sim = dss_sim[sim_name]
    
    CS = ds_sim.CS_through_time.data
    
#    mask = np.zeros(dmr_sim.nr_pools).astype(bool)
#    mask[dmr_sim.wood_product_pool_nrs] = True
#    CS_trees_and_soil = dmr_sim.CS_through_time(0, mask=mask, verbose=True)

    CS_trees_and_soil = ds_sim.CS_through_time_trees_and_soil.data

    data = np.array([CS, CS_trees_and_soil])

    CS_datas.append(data)
    
CS_datas = Q_(np.array(CS_datas).transpose(), "yr*gC/m^2").to("yr*kgC/m^2")
CS_datas.shape

# +
CB_datas = list()

#ti = -2

for sim_name in sim_names:
    ds = dss_sim[sim_name]
#    CB = ds.C_balance_through_time.isel(time=ti)
    x0 = ds.stocks.isel(time=0).sum(dim=["entity", "pool"])
    CB = ds.stocks.sum(dim=["entity", "pool"]) - x0
    CB = CB.data # gC/m^2
    
#    CB_trees_and_soil = ds.C_balance_through_time_trees_and_soil.isel(time=ti)
    x0_trees_and_soil = x0 - ds.stocks.isel(time=0).sel(entity="wood_product").sum(dim="pool")
    CB_trees_and_soil = (ds.stocks.sum(dim=["entity", "pool"]) - ds.stocks.sel(entity="wood_product").sum(dim="pool")) - x0_trees_and_soil
    CB_trees_and_soil = CB_trees_and_soil.data # gC/m^2
    data = np.array([CB, CB_trees_and_soil])
    
    CB_datas.append(data)
    
CB_datas = Q_(np.array(CB_datas).transpose(), "gC/m^2").to("kgC/m^2")
CB_datas.shape

# +
cum_stocks_datas = list()
for sim_name in sim_names:
    ds = dss_sim[sim_name]
    cum_stocks = ds.stocks.sum(dim=["entity", "pool"]).cumsum(dim="time")
    cum_stocks = cum_stocks.data # gC/m^2 yr, ds.stocks.attrs["units"]) * Q_("1 yr")).to("kgC/m^2 yr")
    
    trees_and_soil_entity_nrs = ds.tree_entity_nrs.data.tolist() + [ds.soil_entity_nr]
    cum_stocks_trees_and_soil = ds.stocks.isel(entity=trees_and_soil_entity_nrs).sum(dim=["entity", "pool"]).cumsum(dim="time")
    cum_stocks_trees_and_soil = cum_stocks_trees_and_soil.data # gC/m^2 yr, ds.stocks.attrs["units"]) * Q_("1 yr")).to("kgC/m^2 yr")
    
    data = np.array([cum_stocks, cum_stocks_trees_and_soil])
    cum_stocks_datas.append(data)
    
cum_stocks_datas = Q_(np.array(cum_stocks_datas).transpose(), "yr*gC/m^2").to("yr*kgC/m^2")
cum_stocks_datas.shape

# +
yield_datas = list()
for sim_name in sim_names:
    ds = dss_sim[sim_name]
    WPS_cum = ds.internal_fluxes.sel(pool_to="WP_S").sum(dim=["entity_to", "entity_from", "pool_from"]).cumsum(dim="time")
    WPS_cum = WPS_cum.data # gC/m^2, ds.stocks.attrs["units"])
    WPL_cum = ds.internal_fluxes.sel(pool_to="WP_L").sum(dim=["entity_to", "entity_from", "pool_from"]).cumsum(dim="time")
    WPL_cum = WPL_cum.data # gC/m^2, ds.stocks.attrs["units"])
#    print(sim_name, WPS_cum[-1])
    data = np.array([WPS_cum, WPL_cum])

    yield_datas.append(data)

yield_datas = Q_(np.array(yield_datas).transpose(), "gC/m^2").to("kgC/m^2")
yield_datas.shape

# +
fig, axes = plt.subplots(figsize=(12, 4*4), nrows=4)
panel_names = iter(string.ascii_uppercase[:len(axes)])
axes = iter(axes)

make_space = 1.5

labels = [sim_name.replace("_", " ") for sim_name in sim_names]

ax = next(axes)
panel_name = next(panel_names)
x = np.arange(len(sim_names))
width = 2 / (len(sim_names)+1) / make_space
rects1 = ax.bar(x - width/2, yield_datas[-1, 0, :], width, label=r'short-lasting WPs ($Y_S$)')
rects2 = ax.bar(x + width/2, yield_datas[-1, 1, :], width, label=r'long-lasting WPs ($Y_L$)')
#ax.set_ylabel(f"{yield_datas[-1, 0].units:~P}")
ax.set_ylabel(r"kgC$\,$m$^{-2}$")
ax.legend(loc=4)
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')
ax.set_title("Cumulative wood-product yield")
ax.set_xticklabels("")
#ax.set_ylim([0, 9])

ax = next(axes)
panel_name = next(panel_names)
x = np.arange(len(sim_names)) 
width = 2 / (len(sim_names)+1) / make_space
rects1 = ax.bar(x - width/2, CB_datas[-1, 0, :], width, label='entire system')
rects2 = ax.bar(x + width/2, CB_datas[-1, 1, :], width, label='forest stand')
ax.set_ylabel(r"kgC$\,$m$^{-2}$")
ax.set_title("Integrated Net Carbon Balance (INCB)")
ax.axhline(0, c="black", lw=1)
ax.legend(loc=4)
ax.set_xticklabels("")
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')
#ax.set_ylim([-5, 4])


ax = next(axes)
panel_name = next(panel_names)
x = np.arange(len(sim_names)) 
width = 2 / (len(sim_names)+1) / make_space
rects1 = ax.bar(x - width/2, CS_datas[-1, 0, :], width, label='entire system')
rects2 = ax.bar(x + width/2, CS_datas[-1, 1, :], width, label='forest stand')
ax.set_ylabel(r"kgC$\,$m$^{-2}\,$yr")
ax.set_title("Integrated Inputs Transit Time (IITT)")
ax.legend(loc=4)
ax.set_xticklabels("")
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')
#ax.set_ylim([0, 700])


ax = next(axes)
panel_name = next(panel_names)
x = np.arange(len(sim_names)) 
width = 2 / (len(sim_names)+1) / make_space
rects1 = ax.bar(x - width/2, cum_stocks_datas[-1, 0, :], width, label='entire system')
rects2 = ax.bar(x + width/2, cum_stocks_datas[-1, 1, :], width, label='forest stand')
ax.set_ylabel(r"kgC$\,$m$^{-2}\,$yr")
ax.set_title("Integrated Carbon Stocks (ICS)")
ax.legend(loc=4)
ax.set_xticklabels("")
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')
#ax.set_ylim([0, 1200])

ax.set_xticks(x)
ax.set_xticklabels(labels)

for tick in ax.get_xticklabels():
    tick.set_rotation(60)
    
fig.tight_layout()

# save the figure for the publication
#filename = str(pub_figs_path.joinpath("bars.png"))
#fig.savefig(filename)
#filename


# +
print("\nCWPY")
for nr, sim_name in enumerate(sim_names):
    x = yield_datas[-1, :, nr]
    print(sim_name, round_arr(x.magnitude, 1), x.units)

print("\nINCB")
for nr, sim_name in enumerate(sim_names):
    x = CB_datas[-1, :, nr]
    print(sim_name, round_arr(x.magnitude, 1), x.units)

print("\nIITT")
for nr, sim_name in enumerate(sim_names):
    x = CS_datas[-1, :, nr]
    print(sim_name, round_arr(x.magnitude, 1), x.units)

print("\nICS")
for nr, sim_name in enumerate(sim_names):
    x = cum_stocks_datas[-1, :, nr]
    print(sim_name, round_arr(x.magnitude, 1), x.units)

# -

def compute_SOC_age_variables(sim_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dmr_eq = dmrs_eq[sim_name]
    ds = dss[sim_name]
    dmr = dmrs[sim_name]

    start_age_data = utils.load_start_age_data_from_eq_for_soil_and_wood_products(dmr, dmr_eq, up_to_order=1)

    mean_age_vector = dmr.age_moment_vector(1, start_age_data["start_age_moments_1"])
    SOC_pool_nr = dmr.soil_pool_nrs[-1]
    sim_SOC_age_mean = mean_age_vector[spinup_length:, SOC_pool_nr]

    q = 0.5
    SOC_age_median = np.array([dmr.age_quantile_bin_at_time_bin(q, it, SOC_pool_nr, start_age_data["p0"]) for it in range(len(dmr.times))]);
    sim_SOC_age_median = SOC_age_median[spinup_length:]

    q = 0.95
    SOC_age_quantile_95 = np.array([dmr.age_quantile_bin_at_time_bin(q, it, SOC_pool_nr, start_age_data["p0"]) for it in range(len(dmr.times))]);
    sim_SOC_age_quantile_95 = SOC_age_quantile_95[spinup_length:]

    return sim_SOC_age_mean, sim_SOC_age_median, sim_SOC_age_quantile_95


# +
fig, axes_list = plt.subplots(figsize=(12, 4*5), nrows=5)
panel_names = iter(string.ascii_uppercase[:len(axes_list)])
axes = iter(axes_list)

base_nr = -1

ax = next(axes)
panel_name = next(panel_names)
#ax.set_title(r"Total cumulative wood-product yield C ($Y_S+Y_L$) relative to base")
ax.set_title(r"Total cumulative wood-product yield C ($Y_S+Y_L$)")
combined_yield_datas = yield_datas[:, 0, :] + yield_datas[:, 1, :]
for nr, sim_name in enumerate(sim_names):
    if nr == base_nr:
        continue
   
    l, = ax.plot(combined_yield_datas[:, nr] - combined_yield_datas[:, base_nr], label=sim_name.replace("_", " "), c=colors[sim_name])
    
#ax.axhline(0, c="black", ls="--")
ax.legend()
#ax.set_xlim([0, 80])
ax.set_ylabel(r"kgC$\,$m$^{-2}$")
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')


ax = next(axes)
panel_name = next(panel_names)
#ax.set_title("Integrated Net Carbon Balance (INCB) relative to base")
ax.set_title("Integrated Net Carbon Balance (INCB)")
for nr, sim_name in enumerate(sim_names):
    if nr == base_nr:
        continue
    
#    l, = ax.plot(CB_datas[:, 0, nr] - CB_datas[:, 0, base_nr], label=sim_name.replace("_", " "), c=colors[sim_name])
    l, = ax.plot(CB_datas[:, 0, nr], label=sim_name.replace("_", " "), c=colors[sim_name])

#ax.axhline(0, c="black", ls="--")
#ax.set_xlim([0, 80])
ax.set_ylabel(r"kgC$\,$m$^{-2}$")
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')

    
ax = next(axes)
panel_name = next(panel_names)
#ax.set_title("Integrated Inputs Transit Time (IITT) relative to base")
ax.set_title("Integrated Inputs Transit Time (IITT)")
for nr, sim_name in enumerate(sim_names):
    if nr == base_nr:
        continue
    
#    ax.plot(CS_datas[:, 0, nr] - CS_datas[:, 0, base_nr], label=sim_name.replace("_", " "), c=colors[sim_name])
    ax.plot(CS_datas[:, 0, nr], label=sim_name.replace("_", " "), c=colors[sim_name])
    
#ax.axhline(0, c="black", ls="--")
#ax.set_ylim([ax.get_ylim()[0], 60])
#ax.set_xlim([0, 80])
ax.set_ylabel(r"kgC$\,$m$^{-2}\,$yr")
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')

ax = next(axes)
panel_name = next(panel_names)
#ax.set_title("Integrated Carbon Stocks (ICS) relative to base")
ax.set_title("Integrated Carbon Stocks (ICS)")
for nr, sim_name in enumerate(sim_names):
    if nr == base_nr:
        continue
    
#    ax.plot(cum_stocks_datas[:, 0, nr] - cum_stocks_datas[:, 0, base_nr], label=sim_name, c=colors[sim_name])
    ax.plot(cum_stocks_datas[:, 0, nr], label=sim_name, c=colors[sim_name])

#ax.axhline(0, c="black", ls="--")
#ax.set_xlim([0, 80])
#ax.set_ylim([ax.get_ylim()[0], 60])
ax.set_ylabel(r"kgC$\,$m$^{-2}\,$yr")
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')


ax = next(axes)
panel_name = next(panel_names)
#ax.set_title("Integrated Carbon Stocks (ICS) relative to base")
ax.set_title("SOC age")
for nr, sim_name in enumerate(sim_names):
    if nr == base_nr:
        continue
    
    SOC_age_mean, SOC_age_median, SOC_age_quantile_95 = compute_SOC_age_variables(sim_name)
    
    l1, = ax.plot(SOC_age_mean, c=colors[sim_name], ls="-")
    l2, = ax.plot(SOC_age_median, c=colors[sim_name], ls="--")
    l3, = ax.plot(SOC_age_quantile_95, c=colors[sim_name], ls=":")

#ax.axhline(0, c="black", ls="--")
#ax.set_xlim([0, 80])
#ax.set_ylim([ax.get_ylim()[0], 60])
ax.set_ylabel("yr")
ax.set_xlabel("time [yr]")
ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')
labels = ["mean", "median", "95-percentile"]

l1.set_c("black")
l2.set_c("black")
l3.set_c("black")
ax.legend([l1, l2, l3], labels)

for ax in axes_list:
    ax.set_xlim([0, sim_length])

fig.tight_layout()

# save the figure for the publication
#filename = str(pub_figs_path.joinpath("yield_seq_and_ccmp.png"))
#fig.savefig(filename)
#filename
# -



# ## Ordering according to different CS metrics (entire system = 0 in the middle, stand only = 1)

# ### INCB

[sim_names[k] for k in np.argsort(-CB_datas[-1, 0, :])], -np.sort(-CB_datas[-1, 0, :]).round(1)

[sim_names[k] for k in np.argsort(-CB_datas[-1, 1, :])], -np.sort(-CB_datas[-1, 1, :]).round(1)

# ### IITT

[sim_names[k] for k in np.argsort(-CS_datas[-1, 0, :])], -np.sort(-CS_datas[-1, 0, :]).round(1)

[sim_names[k] for k in np.argsort(-CS_datas[-1, 1, :])], -np.sort(-CS_datas[-1, 1, :]).round(1)

# # ICS

[sim_names[k] for k in np.argsort(-cum_stocks_datas[-1, 0, :])], -np.sort(-cum_stocks_datas[-1, 0, :]).round(1)

[sim_names[k] for k in np.argsort(-cum_stocks_datas[-1, 1, :])], -np.sort(-cum_stocks_datas[-1, 1, :]).round(1)

# ## Ordering according to different WP metrics (short-lasting = 0 in the middle, long-lasting = 1)

[sim_names[k] for k in np.argsort(-yield_datas[-1, 0, :])], -np.sort(-yield_datas[-1, 0, :]).round(1)

[sim_names[k] for k in np.argsort(-yield_datas[-1, 1, :])], -np.sort(-yield_datas[-1, 1, :]).round(1)

[sim_names[k] for k in np.argsort(-yield_datas[-1, :, :].sum(axis=0))], -np.sort(-yield_datas[-1, :, :].sum(axis=0)).round(1)

# ## Do we simulate a kind of self-thinning?

# +
ha_per_acre = 0.404686
acre_per_ha = 1 / ha_per_acre

fig, ax = plt.subplots(figsize=(8, 6))

tis = ds.time - ds.time[0]
sim_name = "untouched forest 320"

ds = dss_sim[sim_name]
tree_names = ds.entity.isel(entity=ds.tree_entity_nrs).data

N_per_m2 = ds.N_per_m2.sel(tree=tree_names)

 # actually, Reineke wants the average to be by SBA, not by N_per_m2
#    DBH = ds.DBH.sel(tree=tree_names).weighted(N_per_m2).mean(dim="tree")
#    DBH = Q_(DBH.data, ds.DBH.attrs["units"])
SBA = np.pi * (ds.DBH/2)**2
D = ds.DBH.weighted(SBA.fillna(0)).mean(dim="tree")
D = Q_(D.data, ds.DBH.attrs["units"])
           
# times of cutting or thinning
x = np.array([0] + [v for v in ds.time[ds.thinning_or_cutting_tree.sel(tree=tree_names).sum(dim="tree") >= 1].data] + [79])

ax.step(tis, N_per_m2.sum(dim="tree") * 10_000, where="pre", label="simulation", c=colors[sim_name])
    
# plot Reineke's reference curve (species independent)
#    ax.plot(tis, [(lambda x: np.exp(4.605*acre_per_ha-1.605*np.log(DBH.magnitude[x])))(ti) for ti in tis], label="Reineke's rule", c=colors[sim_name], ls="--")
ax.plot(tis, [(lambda x: np.exp(4.605*acre_per_ha-1.605*np.log(D.magnitude[x])))(ti) for ti in tis], label="Reineke's rule", c=colors[sim_name], ls="--")
ax.legend()
    
ax.set_xlim([tis[0]-1, tis[-1]])
#    ax.set_ylim([0, 2500])
ax.set_title(sim_name.replace("_", " "))
    
ax.set_xlabel("time [yr]")
ax.text(-0.05, 1.1, panel_name + ")", transform=ax.transAxes, size=20, weight='bold')

ax.set_ylabel("trees per ha")

fig.tight_layout()

## save the figure for the publication
#filename = str(pub_figs_path.joinpath("self_thinning.png"))
#fig.savefig(filename)
#filename
# -

# ## C in different MeanTrees

# +
fig, axes = plt.subplots(figsize=(12, 4*len(sim_names)), nrows=len(sim_names))
axes_list = axes

ax0 = axes[0]
axes = iter(axes)
panel_names = iter(string.ascii_uppercase[:len(axes_list)])
markers = ["o", "s", "P", "D"]

for panel_name, sim_name, color in zip(panel_names, sim_names, colors):
    ax = next(axes)
    ax.set_title(sim_name.replace("_", " "))
    
    ds = dss[sim_name]
    color = colors[sim_name]
    tree_names = ds.entity.isel(entity=ds.tree_entity_nrs).data
    if sim_name != "mixed-aged_pine":
        tree_names = tree_names[nr_spinup_trees:]
        tree_ids = np.arange(4, 8)
    else:
        tree_ids = np.arange(0, 4)
    
    for tree_id, tree_name, marker in zip(tree_ids, tree_names, markers):
        ax.plot(
            ds.time, ds.stocks.isel(entity=tree_id).sum(dim="pool") * 1e-03,
            label=tree_name, c=color, marker=marker, markevery=8 + 2 * tree_id
        )

    ax.legend()
    ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')
    ax.set_ylabel(r"kgC$\,$m$^{-2}$")
    ax.set_ylim([0, ax.get_ylim()[-1]])

axes_list[-1].set_xlabel("time [yr]")
fig.tight_layout()

# save the figure for the publication
filename = str(pub_figs_path.joinpath("tree_C.png"))
fig.savefig(filename)
filename
# -
# ## Benchmarking figures

# ## 5-yr radial growth
# We check the annual radial growth at breast height over 5 years, because Repola (2008: birch, 2009: pine and spruce) provides values for it. Naturally, we use `r_BH = tree_in_stand.dbh / 2` for that.
#
# - Repola (2009):
#     - pine: mean=0.54 cm, std=0.33, range=0.04-2.03
#     - spruce: mean=0.76 cm, std=0.41, range=0.07-2.48
#     
# In the plots below, the dashed line represents the Repola mean, the semi-transparent area is the Repola standard deviation around the mean and the even more transparent area is the range of values found in Repola.


# +
def compute_r_BH_growth_5_yrs_for_single_ti(DBH_tree, ti):
    if ti < 4:
        return np.nan
    
    dbhs = DBH_tree[(ti-4):(ti+1)]
    Delta_r_5_yrs = (dbhs[-1]-dbhs[0]) / 2
    if Delta_r_5_yrs < 0:
        return np.nan
        
    return float(Delta_r_5_yrs.data)
        
def compute_r_BH_growth_5_yrs(ds, tree_names):
    DBH = ds.DBH.sel(tree=tree_names)
    N_per_m2 = ds.N_per_m2.sel(tree=tree_names)

    num: float = 0
    denom: float = 0
    for tree_name in tree_names:
        DBH_tree = DBH.sel(tree=tree_name)
        Delta_r_5_yrs_tree = np.array([compute_r_BH_growth_5_yrs_for_single_ti(DBH_tree, ti) for ti in range(len(DBH_tree))])
        Delta_r_5_yrs_tree[np.isnan(Delta_r_5_yrs_tree)] = 0
        
        N_per_m2_tree = N_per_m2.sel(tree=tree_name).data
        N_per_m2_tree[np.isnan(N_per_m2_tree)] = 0
        num += Delta_r_5_yrs_tree * N_per_m2_tree
        denom += N_per_m2_tree
    
    res = num / denom
    res[res == 0] = np.nan
    return res


# +
sim_names_tmp = ["even-aged_pine", "even-aged_spruce"]

fig, axes = plt.subplots(figsize=(12, 4*len(sim_names_tmp)), nrows=len(sim_names_tmp))

species_data = {
    "pine": {"mean": 0.54, "std": 0.33, "range": (0.04, 2.03)},
    "spruce": {"mean": 0.76, "std": 0.41, "range": (0.07, 2.48)},
    "birch": {"mean": 0.75, "std": 0.58, "range": (0.05, 3.47)},
}

fig.suptitle("Radial growth (dbh$/2$) over 5 years averaged over all trees")
titles = [s.replace("_", " ").capitalize() for s in sim_names_tmp]
for sim_name, ax, title, panel_name in zip(sim_names_tmp, axes, titles, string.ascii_uppercase[:len(axes)]):
    ds = dss[sim_name]
    ax.set_title(title)
    ds = dss[sim_name]
    tree_names = ds.entity.isel(entity=ds.tree_entity_nrs).data[nr_spinup_trees:]
    species = str(ds.species.sel(tree=tree_names[0]).data)
    
    r_BH_growth_5_yrs = compute_r_BH_growth_5_yrs(ds, tree_names)
    l, = ax.plot(ds.time, r_BH_growth_5_yrs, c=colors[sim_name])
   
    mean = species_data[species]["mean"]
    std = species_data[species]["std"]
    range_ = species_data[species]["range"]
    n = len(ds.time)
    ax.fill_between(ds.time, [mean-std]*n, [mean+std]*n, alpha=0.2, color="black")
    ax.fill_between(ds.time, [range_[0]]*n, [range_[1]]*n, alpha=0.1, color="black")
    ax.axhline(y=mean, c="black", ls="--")
    
    ax.set_xlim([ds.time[0], ds.time[-1]])
    ax.set_ylim([0, 2.6])
    ax.set_xlabel("time [yr]")
    ax.set_ylabel(r"$\Delta$ [cm / 5yr]")
    ax.text(-0.05, 1.1, panel_name+")", transform=ax.transAxes, size=20, weight='bold')

    # vertical thinning lines
    for x in ds.time[ds.thinning_or_cutting_tree.sum(dim="tree") >= 1][:-1]:
        ax.axvline(x.data, alpha=0.2, color="black")
    
fig.tight_layout()

# save the figure for the publication
filename = str(pub_figs_path.joinpath("benchmarking_radial_growth.png"))
fig.savefig(filename)
filename
# -


# # Compute fluxes caused by cuttings in mixed-aged pine scenario (kgC/m^2)

# +
ds = dss["mixed-aged_pine"]

# when which tree was cut
data =[
    (2000, ["pine3"]), (2020, ["pine0"]), (2040, ["pine1"]), (2060, ["pine2"]),
    (2079, ["pine0", "pine1", "pine2", "pine3"])
]

for (year, tree_names) in data:
    print(year)
    print("Soil:", round(ds.internal_fluxes.sel(entity_from=tree_names, entity_to="soil", time=year-2000).sum(dim=["entity_from", "pool_from", "pool_to"]).data / 1000, 1))
    print("WP:", round(ds.internal_fluxes.sel(entity_from=tree_names, entity_to="wood_product", time=year-2000).sum(dim=["entity_from", "pool_from", "pool_to"]).data / 1000, 1))
    print("    WP_S:", round(ds.internal_fluxes.sel(entity_from=tree_names, entity_to="wood_product", pool_to="WP_S", time=year-2000).sum(dim=["entity_from", "pool_from"]).data / 1000, 1))
    print("    WP_L:", round(ds.internal_fluxes.sel(entity_from=tree_names, entity_to="wood_product", pool_to="WP_L", time=year-2000).sum(dim=["entity_from", "pool_from"]).data / 1000, 1))


# -

# ## Create stand cross-section videos

# +
# %%time

# the base for the x-axis of the video, quite arbitray, could be initial total tree number
base_N = 2_000 / 10_000
print("Creating stand cross section videos")

for sim_name in sim_names:
    ds = dss[sim_name]
    print(sim_name)
    filepath = sim_cohort_path.joinpath(sim_name + "_stand_cross_sect.mp4")
    utils.create_stand_cross_section_video(ds, filepath, base_N)
    print(filepath)
# -

# ## Create simulation video: last 80 years of spinup and then the 80 years of simulation

# +
# %%time

for sim_name in sim_names:
    print(sim_name)
    animation_filepath = sim_cohort_path.joinpath(sim_name + "_sim.mp4")
    ds = dss[sim_name]
    ds_long = xr.open_dataset(str(sim_cohort_path.joinpath(sim_name + "_long.nc")))
    dmr = dmrs[sim_name]
    dmr_eq = dmrs_eq[sim_name]

    utils.create_simulation_video(
        ds_long,
        dmr_eq,
        np.array([dmr.soil_pool_nrs[-1]]),
        animation_filepath, 
        resolution=10,
        time_index_start=spinup_length-80,
        clearcut_index=spinup_length,
        year_shift=-spinup_length,
        time_index_stop=len(ds_long.time)-2,
        cache_size=1_000
    )
    print(animation_filepath)

# +
fig, axes = plt.subplots(figsize=(12, 4*2), nrows=2)

ax = axes[0]
ax.set_title("Stand leaf area (dashed lines are means through time)")
for sim_name in sim_names:
    ds = dss[sim_name]
    var = ds.LA_tree.sum(dim="tree")
    l, = ax.plot(ds.time, var, label=sim_name.replace("_", " "))
    ax.axhline(var.mean(dim="time"), c=l.get_c(), ls = "--")
    
ax.legend()
ax.set_ylabel(r"m$^2\,$m$^{-2}$")

ax = axes[1]
ax.set_title("Stand total tree number (N)")
for sim_name in sim_names:
    ds = dss[sim_name]
    var = ds.N_per_m2.sum(dim="tree")
    l, = ax.plot(ds.time, var, label=sim_name.replace("_", " "))
    ax.axhline(var.mean(dim="time"), c=l.get_c(), ls = "--")
    
ax.set_ylabel(r"m$^{-2}$")

fig.tight_layout()
# -

# ## Overyielding (WP_S + WP_L and INCB + IITT + ICS)

sim_name = 'mixed-aged_pine'
ds = dss[sim_name]

dmr_path = sim_cohort_path.joinpath(sim_name + ".dmr")
dmr = DMR.load_from_file(dmr_path)
dmr.initialize_state_transition_operator_matrix_cache(10_000)
dmr.nr_pools

nr_tree_pools, nr_pines, nr_spruces = 10, 2, 2
base_pool_nr = 0# nr_spinup_trees * nr_tree_pools
pine_pools = np.arange(base_pool_nr, base_pool_nr + nr_pines * nr_tree_pools)
print(pine_pools)
base_pool_nr = base_pool_nr + nr_pines * nr_tree_pools
spruce_pools = np.arange(base_pool_nr, base_pool_nr + nr_spruces * nr_tree_pools)
print(spruce_pools)

# +
mask_pine = np.ones(dmr.start_values.shape, dtype=bool)
mask_pine[pine_pools] = False

#start_values_pine = dmr.start_values.copy()
#start_values_pine[mask_pine] = 0

net_Us_pine = dmr.net_Us.copy()
net_Us_pine[:, mask_pine] = 0

dmr_pine = DMR.from_Bs_and_net_Us(dmr.start_values, dmr.times, dmr.Bs, net_Us_pine)
dmr_pine.initialize_state_transition_operator_matrix_cache(10_000)

# +
mask_spruce = np.ones(dmr.start_values.shape, dtype=bool)
mask_spruce[spruce_pools] = False

#start_values_spruce = dmr.start_values.copy()
#start_values_spruce[mask_spruce] = 0

net_Us_spruce = dmr.net_Us.copy()
net_Us_spruce[:, mask_spruce] = 0

dmr_spruce = DMR.from_Bs_and_net_Us(dmr.start_values, dmr.times, dmr.Bs, net_Us_spruce)
dmr_spruce.initialize_state_transition_operator_matrix_cache(10_000)
# -

WP_cum = dict()
for pool_to in ["WP_S", "WP_L"]:
    WP_cum[pool_to] = dict()
    for sim_name in sim_names:
        WP_cum[pool_to][sim_name] = dict()
        ds = dss[sim_name]
        for species in ["pine", "spruce"]:
            species_entity_nrs = ds.tree_entity_nrs[ds.species.data==species]
            if sim_name != "mixed-aged_pine":
                species_entity_nrs = species_entity_nrs[species_entity_nrs >= nr_spinup_trees]
                
            if len(species_entity_nrs) > 0:
                y = ds.internal_fluxes.isel(entity_from=species_entity_nrs).sel(entity_to="wood_product", pool_to=pool_to).sum(dim=["tree_entity_nrs", "pool_from"]).cumsum(dim="time")
                y = Q_(y.data, ds.stocks.attrs["units"]).to("kgC/m^2")
                
                y = y - y[0]
                WP_cum[pool_to][sim_name][species] = y
        
        d = WP_cum[pool_to][sim_name]
        d["total"] = Q_(np.sum([data.data for data in d.values()], axis=0), "kgC/m^2")
    
    d = WP_cum[pool_to]
    d["theoretical mix"] = 0.5 * d["even-aged_pine"]["total"] + 0.5 * d["even-aged_spruce"]["total"]

INCB = dict()
for sim_nr, sim_name in enumerate(sim_names):
    INCB[sim_name] = CB_datas[:, 0, sim_nr]
INCB["theoretical mix"] = 0.5 * INCB["even-aged_pine"] + 0.5 * INCB["even-aged_spruce"]

# +
IITT = dict()
for sim_nr, sim_name in enumerate(sim_names):
    IITT[sim_name] = CS_datas[:, 0, sim_nr]

IITT["pine in the mix"] = Q_(dmr_pine.CS_through_time(0) / 1000, "kgC / m^2 * yr")
IITT["spruce in the mix"] = Q_(dmr_spruce.CS_through_time(0) / 1000, "kgC / m^2 * yr")
IITT["theoretical mix"] = 0.5 * IITT["even-aged_pine"] + 0.5 * IITT["even-aged_spruce"]

# +
ICS = dict()
#ICS["pine"] = Q_(np.cumsum(dmr_pine.solve().sum(axis=1)), "gC / m^2 * yr")
#ICS["spruce"] = Q_(np.cumsum(dmr_spruce.solve().sum(axis=1)), "gC / m^2 * yr")
for sim_nr, sim_name in enumerate(sim_names):
    ICS[sim_name] = cum_stocks_datas[:, 0, sim_nr]

ICS["theoretical mix"] = 0.5 * ICS["even-aged_pine"] + 0.5 * ICS["even-aged_spruce"]


# +
fig, axes = plt.subplots(figsize=(12, 4*5), nrows=5)

ax = axes[0]
ax.set_title("Short-lasting wood products excluding initial clear cut")
d = WP_cum["WP_S"]
for sim_name in sim_names:
    color = colors[sim_name]
    ax.plot(ds.time, d[sim_name]["total"], label=sim_name.replace("_", " "), color=color)

sim_name = "even-aged_mixed"
color = colors[sim_name]
ax.plot(ds.time, d[sim_name]["pine"], label="pine in the mix", ls="--", color=colors["even-aged_mixed"])
ax.plot(ds.time, d[sim_name]["spruce"], label="spruce in the mix", ls=":", color=colors["even-aged_mixed"])
ax.plot(ds.time, d["theoretical mix"], label="theoretical mix", color="black")

ax.legend()
ax.set_ylabel(r"kgC$\,$m$^{-2}$")
ax.set_ylim([0, ax.get_ylim()[-1]])


ax = axes[1]
ax.set_title("Long-lasting wood products excluding initial clear cut")
d = WP_cum["WP_L"]
for sim_name in sim_names:
    color = colors[sim_name]
    ax.plot(ds.time, d[sim_name]["total"], label=sim_name.replace("_", " "), color=color)

sim_name = "even-aged_mixed"
color = colors[sim_name]
ax.plot(ds.time, d[sim_name]["pine"], label="pine in the mix", ls="--", color=colors["even-aged_mixed"])
ax.plot(ds.time, d[sim_name]["spruce"], label="spruce in the mix", ls=":", color=colors["even-aged_mixed"])
ax.plot(ds.time, d["theoretical mix"], label="theoretical mix", color="black")
ax.legend()

ax.legend()
ax.set_ylabel(r"kgC$\,$m$^{-2}$")
ax.set_ylim([0, ax.get_ylim()[-1]])


ax = axes[2]
ax.set_title("INCB")
for sim_name in sim_names:
    ax.plot(ds.time, INCB[sim_name], label=sim_name.replace("_", " "))

ax.plot(ds.time, INCB["theoretical mix"], color="black", label="theoretical mix")
ax.axhline(0, c="black", alpha=0.2, lw=2)
ax.set_ylabel(r"kgC$\,$m$^{-2}$")
ax.legend()

ax = axes[3]
ax.set_title("IITT")
for sim_name in sim_names:
    ax.plot(ds.time, IITT[sim_name], label=sim_name.replace("_", " "))
ax.plot(ds.time, IITT["pine in the mix"], ls="--", color=colors["even-aged_mixed"], label="pine in the mix")
ax.plot(ds.time, IITT["spruce in the mix"], ls=":", color=colors["even-aged_mixed"], label="spruce in the mix")
ax.plot(ds.time, IITT["theoretical mix"], color="black", label="theoretical mix")
ax.legend()

ax.set_ylim([0, ax.get_ylim()[-1]])
ax.set_ylabel(r"kgC$\,$m$^{-2}\,$yr")

ax = axes[4]
ax.set_title("ICS")
for sim_name in sim_names:
    ax.plot(ds.time, ICS[sim_name], label=sim_name.replace("_", " "))
ax.plot(ds.time, ICS["theoretical mix"], color="black", label="theoretical mix")
ax.legend()
ax.set_ylim([0, ax.get_ylim()[-1]])

for ax in axes:
    ax.set_xlim([0, 80])
ax.set_ylabel(r"kgC$\,$m$^{-2}\,$yr")

fig.tight_layout()
# -

base = IITT["pine in the mix"][-1] + IITT["spruce in the mix"][-1]
print("pine share  :", IITT["pine in the mix"][-1] / base * 100)
print("spruce share:", IITT["spruce in the mix"][-1] / base * 100)

# ## Total C drop during first 20 years


for sim_name in sim_names[1:]:
    print(sim_name)
    ds = dss[sim_name]
    min_time = ds.stocks.sum(dim=["entity", "pool"])[:40].argmin()
    print("min time", min_time.data)
    x0 = ds.stocks.isel(time=0).sum().data * 1e-03
    x = ds.stocks.isel(time=min_time).sum().data * 1e-03
    print(x0, x, x0-x, x/x0 * 100, 100 - x/x0 * 100)
    print()

# +
ds = dss["even-aged_pine"]
x0 = ds.stocks.isel(time=0).sum().data * 1e-03
x = ds.stocks.isel(time=20).sum().data * 1e-03

print(x0, x, x0-x, x/x0 * 100)
# -



# ## Compute fluxes caused by cuttings in mixed-aged pine scenario (gC/m^2)

ds = dss["mixed-aged_pine"]

# +
data =[
    (2000, ["pine3"]), (2020, ["pine0"]), (2040, ["pine1"]), (2060, ["pine2"]),
    (2079, ["pine0", "pine1", "pine2", "pine3"])
]

for (year, tree_names) in data:
    print(year)
    print("Soil:", round(ds.internal_fluxes.sel(entity_from=tree_names, entity_to="soil", time=year-2000).sum(dim=["entity_from", "pool_from", "pool_to"]).data / 1000, 1))
    print("WP:", round(ds.internal_fluxes.sel(entity_from=tree_names, entity_to="wood_product", time=year-2000).sum(dim=["entity_from", "pool_from", "pool_to"]).data / 1000, 1))
# -

# ## Even-aged, single-species, total stand carbon use efficiency (CUE)

# +
#sim_names = ["pine", "spruce"]

fig, ax = plt.subplots(figsize=(12, 4))

ax.set_title("Total stand carbon use efficiency")
for sim_name in ["even-aged_pine", "even-aged_spruce"]:
    ds = dss[sim_name]
    tree_names = ds.entity.isel(entity=ds.tree_entity_nrs).data[nr_spinup_trees:]  
    species = str(ds.species.sel(tree=tree_names[0]).data)
    
    GPP = ds.LabileC_assimilated_tree.sel(tree=tree_names).sum(dim="tree")
    Ra = ds.R_A_tree.sel(tree=tree_names).sum(dim="tree")
    Rd = ds.LabileC_respired_tree.sel(tree=tree_names).sum(dim="tree")
    CUE_with_Rd = (GPP-Ra) / (GPP+Rd)
    CUE_with_Rd[CUE_with_Rd==-np.inf] = np.nan
    
#    ax.plot(ds.time, ds.CUE, label=species)
    ax.plot(ds.time, CUE_with_Rd, label=sim_name.replace("_", " "))

#    print(sim_name, ds.CUE.mean(dim="time").data, ds.CUE.min(dim="time").data, ds.CUE.max(dim="time").data)
    print(sim_name, CUE_with_Rd.mean(dim="time").data, CUE_with_Rd.min(dim="time").data, CUE_with_Rd.max(dim="time").data)

ax.legend()
ax.set_xlim([0, 80])
ax.set_ylim([0, 1])
ax.set_xlabel("time [yr]")

fig.tight_layout()
# -

# ## Trunk wood density

# +
fig, ax = plt.subplots(figsize=(12, 4))

sim_names_tmp = ["even-aged_pine", "even-aged_spruce"]

ax.set_title("Wood density: stem biomass / stem volume versus time")

for sim_name in sim_names_tmp:
#for sim_name in sim_names:
    ds = dss[sim_name]
    tree_names = ds.entity.isel(entity=ds.tree_entity_nrs).data[nr_spinup_trees:]
#    tree_names = ds.entity.isel(entity=ds.tree_entity_nrs).data[-1:]
#    species = str(ds.species.sel(tree=tree_name).data)
    
    V_TS_ACGCA_tree = ds.V_TS_ACGCA_tree.sel(tree=tree_names)
    V_TS_ACGCA_tree = Q_(V_TS_ACGCA_tree.data, V_TS_ACGCA_tree.attrs["units"])

    V_TH_ACGCA_tree = ds.V_TH_ACGCA_tree.sel(tree=tree_names)
    V_TH_ACGCA_tree = Q_(V_TH_ACGCA_tree.data, V_TH_ACGCA_tree.attrs["units"])
    
    V_T_ACGCA_tree = V_TS_ACGCA_tree + V_TH_ACGCA_tree
    
    B_TH = ds.stocks.sel(entity=tree_names, pool="B_TH")
    B_TH = Q_(B_TH.data, B_TH.attrs["units"])
    B_TS = ds.stocks.sel(entity=tree_names, pool="B_TS")
    B_TS = Q_(B_TS.data, B_TS.attrs["units"])

    B_OS = ds.stocks.sel(entity=tree_names, pool="B_OS")
    B_OS = Q_(B_OS.data, B_OS.attrs["units"])
    C_S = ds.stocks.sel(entity=tree_names, pool="C_S")
    C_S = Q_(C_S.data, C_S.attrs["units"])
    C_TS = B_TS / (B_OS + B_TS) * C_S
    
    N_per_m2 = ds.N_per_m2.sel(tree=tree_names)
    N_per_m2 = Q_(N_per_m2.data, N_per_m2.attrs["units"])
    
    # to go to the single tree
    B_TH = B_TH / N_per_m2
    B_TS = B_TS / N_per_m2
    C_TS = C_TS / N_per_m2
    
    # multiply by 2 because the density is in g_dw / m^2 not in gC / m^2
    B_TS = B_TS.to("kg_dw")
    B_TH = B_TH.to("kg_dw")
    C_TS = C_TS.to("kg_dw")
    
    times = ds.time
#    y = (B_TS + B_TH) / V_T_ACGCA_tree
    y = (B_TS + B_TH + C_TS) / V_T_ACGCA_tree
    y = np.nansum(y * N_per_m2, axis=0) / np.nansum(N_per_m2, axis=0)
#    print(y)
    print(sim_name, np.nanmean(y[:-2]), np.nanmin(y[:-2]), np.nanmax(y))
    l, = ax.plot(times, y, label=f"{sim_name.replace('_', ' ')}, stem wood")
    
#    y = (B_TS) / V_TS_ACGCA_tree
#    print("SW")
#    print(y)
#    axes[0].plot(times, y, label=f"{species}, stem sapwood", c=l.get_c(), marker="o")

#    Hs = Q_(ds.height.sel(tree=tree_name).data, ds.height.attrs["units"])
#    axes[1].plot(Hs, y, label=f"{species}, stem sapwood", c=l.get_c(), marker="o")
    
#    y = (B_TH) / V_TH_ACGCA_tree
#    print("HW")
#    print(y)
#    axes[0].plot(times, y, label=f"{species}, stem heartwood", c=l.get_c(), ls="-.")
#    axes[1].plot(Hs, y, label=f"{species}, stem heartwood", c=l.get_c(), ls="-.")
    
ax.legend()
ax.set_xlim([times[0], times[-1]])
#ax.set_xlim([ds.time[0], ds.time[-1]])
#ax.set_ylim([0, ax.get_ylim()[-1]])
#ax.set_ylabel(y.units)
ax.set_ylabel("kg$_{\mathrm{dw}}$ / m$^3$")
ax.set_xlabel(f"time [yr]")

fig.tight_layout()
# -



