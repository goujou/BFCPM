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

# # Untouched forest for 160/320 yr analysis

# %load_ext autoreload

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from BFCPM.__init__ import Q_, SIMULATIONS_PATH


# +
sim_date = "2023-10-19"
sim_name = "DWC_untouched_forest_320_pine_12"
sim_path = SIMULATIONS_PATH.joinpath(sim_date).joinpath(sim_name + ".nc")

ds = xr.open_dataset(str(sim_path))

# -

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

