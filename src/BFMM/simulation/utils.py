"""Some handy functions regarding a simulation."""

from __future__ import annotations

import warnings
from typing import Any, Dict, List

import numpy as np
import xarray as xr
from CompartmentalSystems.discrete_model_run import DiscreteModelRun as DMR
from LAPM.discrete_linear_autonomous_pool_model import \
    DiscreteLinearAutonomousPoolModel as DLAPM
from tqdm import tqdm

from .. import Q_, utils
from .simulation import Simulation


def add_simulation_tree_dict_to_ds(
    ds: xr.Dataset, variables: List[Dict[str, Any]]
) -> xr.Dataset:
    """Add tree data recorded during a simulation to the simulation dataset.

    This function is called right after the simulation run with the
    recorded data belonging to trees.
    """
    nr_tree_entities = len(ds.tree_entity_nrs)
    tree_names = ds.entity.isel(entity=ds.tree_entity_nrs).data

    nr_times = len(ds.coords["time"])
    data_vars = dict()
    for variable in variables:
        data = np.nan * np.ones((nr_tree_entities, nr_times))
        for tree_nr, tree_name in enumerate(ds.coords["tree"].data):
            q = Q_.from_list(variable["data"][tree_name])
            data[tree_nr, : len(q)] = q.magnitude

        data_vars[variable["name"]] = xr.DataArray(
            data=data, dims=["tree", "time"], attrs={"units": q.get_netcdf_unit()}
        )

    return ds.assign(data_vars).assign_coords({"tree": tree_names})


def add_simulation_tree_data_to_ds(
    ds: xr.Dataset, variables: List[Dict[str, Any]]
) -> xr.Dataset:
    """Add tree data recorded during a simulation to the simulation dataset.

    This function is called when additional quantities have been computed who
    themselves rely on the simulation dataset with ecorded data.
    """
    nr_tree_entities = len(ds.tree_entity_nrs)
    tree_names = ds.entity.isel(entity=ds.tree_entity_nrs).data

    nr_times = len(ds.coords["time"])
    data_vars = dict()
    for variable in variables:
        data = np.nan * np.ones((nr_tree_entities, nr_times))
        q = variable["data"]
        data[:, : q.shape[1]] = q.magnitude

        data_vars[variable["name"]] = xr.DataArray(
            data=data, dims=["tree", "time"], attrs={"units": q.get_netcdf_unit()}
        )

    return ds.assign(data_vars).assign_coords({"tree": tree_names})


def add_simulation_data_to_ds(
    ds: xr.Dataset, variables: List[Dict[str, Any]]
) -> xr.Dataset:
    """Add simulation stand data recorded during a simulation to the simulation dataset."""
    nr_times = len(ds.coords["time"])

    data_vars = dict()
    for variable in variables:
        data = np.nan * np.ones(nr_times)
        q = Q_.from_list(variable["data"])
        data[: len(q)] = q.magnitude

        data_vars[variable["name"]] = xr.DataArray(
            data=data, dims=["time"], attrs={"units": q.get_netcdf_unit()}
        )

    return ds.assign(data_vars)


def add_additional_simulation_quantities_to_ds(ds: xr.Dataset) -> xr.Dataset:
    """Based on a simulation dataset, compute more data and add it."""
    # divisions of zero by zero make warnings, but the result is nan
    # nan is okaym but the warnings look pretty disgusting
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        new_ds = ds.copy()

        total_C_stock = Q_(ds.stocks.sum(dim=["entity", "pool"]).data, "gC/m^2")
        cum_WP_S_input = Q_(
            ds.internal_fluxes.sel(entity_to="wood_product", pool_to="WP_S")
            .sum(dim=["entity_from", "pool_from"])
            .cumsum(dim="time")
            .data,
            "gC/m^2",
        )
        cum_WP_L_input = Q_(
            ds.internal_fluxes.sel(entity_to="wood_product", pool_to="WP_L")
            .sum(dim=["entity_from", "pool_from"])
            .cumsum(dim="time")
            .data,
            "gC/m^2",
        )
        cum_R = Q_(
            ds.output_fluxes.sum(dim=["entity_from", "pool_from"])
            .cumsum(dim="time")
            .data,
            "gC/m^2",
        )

        tree_biomass = Q_(
            ds.stocks.sel(entity=ds.tree).sum(dim=["tree", "pool"]).data, "gC/m^2"
        )
        max_dbh = Q_(ds.DBH.max(dim="tree").data, "cm")
        soil_biomass = Q_(ds.stocks.sel(entity="soil").sum(dim=["pool"]).data, "gC/m^2")
        wood_product_biomass = Q_(
            ds.stocks.sel(entity="wood_product").sum(dim=["pool"]).data, "gC/m^2"
        )

        R_A = Q_(
            ds.output_fluxes.sel(entity_from=ds.tree)
            .sum(dim=["tree", "pool_from"])
            .data,
            "gC/m^2/yr",
        )
        R_M = Q_(ds.R_M_tree.sum(dim="tree").data, ds.R_M_tree.attrs["units"])
        R_G = R_A - R_M
        R_H = Q_(
            ds.output_fluxes.sel(entity_from="soil").sum(dim="pool_from").data,
            "gC/m^2/yr",
        )

        GPP = Q_(ds.GPP_year.data, "gC/m^2/yr")
        NPP = GPP - R_A
        CUE = NPP / GPP
        mleaf = Q_(
            ds.stocks.sel(entity=ds.tree)
            .sel(pool=["B_L", "C_L"])
            .sum(dim=["tree", "pool"])
            .data,
            "gC/m^2",
        )
        GPP_per_leaf_biomass = GPP / mleaf
        NPP_per_leaf_biomass = NPP / mleaf

        # tree data
        GPP_tree = Q_(
            ds.input_fluxes.sel(entity_to=ds.tree).sum(dim="pool_to").data, "gC/m^2/yr"
        )
        R_A_tree = Q_(
            ds.output_fluxes.sel(entity_from=ds.tree).sum(dim="pool_from").data,
            "gC/m^2/yr",
        )
        NPP_tree = GPP_tree - R_A_tree
        CUE_tree = NPP_tree / GPP_tree

        R_M_tree = Q_(ds.R_M_tree.data, ds.R_M_tree.attrs["units"])
        R_G_tree = R_A_tree - R_M_tree

        RA_L_tree = Q_((ds.GL_tree + ds.ML_tree).data, ds.GL_tree.attrs["units"])
        RA_R_tree = Q_((ds.GR_tree + ds.MR_tree).data, ds.GR_tree.attrs["units"])
        RA_S_tree = Q_((ds.GS_tree + ds.MS_tree).data, ds.GS_tree.attrs["units"])

        mleaf_tree = Q_(
            ds.stocks.sel(entity=ds.tree).sel(pool=["B_L", "C_L"]).sum(dim="pool").data,
            "gC/m^2",
        )
        GPP_per_leaf_biomass_tree = GPP_tree / mleaf_tree
        NPP_per_leaf_biomass_tree = NPP_tree / mleaf_tree

        r_BH = Q_(ds.DBH.data / 2, ds.DBH.attrs["units"])
        r_BH_growth_5_yrs = np.nan * r_BH
        r_BH_growth_5_yrs[:, 5:] = Q_(
            np.array(
                [
                    r_BH.magnitude[:, k] - r_BH.magnitude[:, k - 5]
                    for k in np.arange(5, r_BH.shape[1])
                ]
            ).transpose(),
            ds.DBH.attrs["units"],
        )

        tree_biomass_tree = Q_(
            ds.stocks.sel(entity=ds.tree).sum(dim=["pool"]).data, "gC/m^2"
        )
        thinning_or_cutting_tree = Q_((ds.N_per_m2.diff(dim="time") < 0).data, "")

        # single tree data, i.e. per plant
        N_per_m2_tree = Q_(ds.N_per_m2.data, ds.N_per_m2.attrs["units"])
        GPP_single_tree = GPP_tree / N_per_m2_tree
        R_A_single_tree = R_A_tree / N_per_m2_tree
        NPP_single_tree = GPP_single_tree - R_A_single_tree
        CUE_single_tree = NPP_single_tree / GPP_single_tree
        R_M_single_tree = R_M_tree / N_per_m2_tree
        R_G_single_tree = R_G_tree / N_per_m2_tree

        mleaf_single_tree = mleaf_tree / N_per_m2_tree
        GPP_per_leaf_biomass_single_tree = GPP_per_leaf_biomass_tree / N_per_m2_tree
        NPP_per_leaf_biomass_single_tree = NPP_per_leaf_biomass_tree / N_per_m2_tree

        tree_biomass_single_tree = tree_biomass / N_per_m2_tree

        LA_single_tree = Q_(ds.LA_tree.data, ds.LA_tree.attrs["units"]) / N_per_m2_tree
        V_T_single_tree = (
            Q_(ds.V_T_tree.data, ds.V_T_tree.attrs["units"]) / N_per_m2_tree
        )
        V_TH_single_tree = (
            Q_(ds.V_TH_tree.data, ds.V_TH_tree.attrs["units"]) / N_per_m2_tree
        )
        V_TS_single_tree = (
            Q_(ds.V_TS_tree.data, ds.V_TS_tree.attrs["units"]) / N_per_m2_tree
        )

        LabileC_assimilated_single_tree = (
            Q_(
                ds.LabileC_assimilated_tree.data,
                ds.LabileC_assimilated_tree.attrs["units"],
            )
            / N_per_m2_tree
        )
        LabileC_respired_single_tree = (
            Q_(ds.LabileC_respired_tree.data, ds.LabileC_respired_tree.attrs["units"])
            / N_per_m2_tree
        )

        GL_single_tree = Q_(ds.GL_tree.data, ds.GL_tree.attrs["units"]) / N_per_m2_tree
        GR_single_tree = Q_(ds.GR_tree.data, ds.GR_tree.attrs["units"]) / N_per_m2_tree
        GS_single_tree = Q_(ds.GS_tree.data, ds.GS_tree.attrs["units"]) / N_per_m2_tree
        ML_single_tree = Q_(ds.ML_tree.data, ds.ML_tree.attrs["units"]) / N_per_m2_tree
        MR_single_tree = Q_(ds.MR_tree.data, ds.MR_tree.attrs["units"]) / N_per_m2_tree
        MS_single_tree = Q_(ds.MS_tree.data, ds.MS_tree.attrs["units"]) / N_per_m2_tree

        RA_L_single_tree = RA_L_tree / N_per_m2_tree
        RA_R_single_tree = RA_R_tree / N_per_m2_tree
        RA_S_single_tree = RA_S_tree / N_per_m2_tree

        f_L_times_E_single_tree = (
            Q_(ds.f_L_times_E_tree.data, ds.f_L_times_E_tree.attrs["units"])
            / N_per_m2_tree
        )
        f_R_times_E_single_tree = (
            Q_(ds.f_R_times_E_tree.data, ds.f_R_times_E_tree.attrs["units"])
            / N_per_m2_tree
        )
        f_O_times_E_single_tree = (
            Q_(ds.f_O_times_E_tree.data, ds.f_O_times_E_tree.attrs["units"])
            / N_per_m2_tree
        )
        f_T_times_E_single_tree = (
            Q_(ds.f_T_times_E_tree.data, ds.f_T_times_E_tree.attrs["units"])
            / N_per_m2_tree
        )
        #        f_CS_times_CS_single_tree = (
        #            Q_(ds.f_CS_times_CS_tree.data, ds.f_CS_times_CS_tree.attrs["units"])
        #            / N_per_m2_tree
        #        )

        R_M_correction_single_tree = (
            Q_(ds.R_M_correction_tree.data, ds.R_M_correction_tree.attrs["units"])
            / N_per_m2_tree
        )

        B_S_star_single_tree = (
            Q_(ds.B_S_star_tree.data, ds.B_S_star_tree.attrs["units"]) / N_per_m2_tree
        )

        data_vars = [
            {"name": "total_C_stock", "data": total_C_stock},
            {"name": "cumulative_WP_S_input", "data": cum_WP_S_input},
            {"name": "cumulative_WP_L_input", "data": cum_WP_L_input},
            {"name": "cumulative_release", "data": cum_R},
            {"name": "tree_biomass", "data": tree_biomass},
            {"name": "soil_biomass", "data": soil_biomass},
            {"name": "wood_product_biomass", "data": wood_product_biomass},
            {"name": "max_dbh", "data": max_dbh},
            {"name": "NPP", "data": NPP},
            {"name": "R_A", "data": R_A},
            {"name": "R_H", "data": R_H},
            {"name": "R_M", "data": R_M},
            {"name": "R_G", "data": R_G},
            {"name": "leaf_biomass", "data": mleaf},
            {"name": "GPP_per_leaf_biomass", "data": GPP_per_leaf_biomass},
            {"name": "NPP_per_leaf_biomass", "data": NPP_per_leaf_biomass},
            {"name": "CUE", "data": CUE},
        ]
        new_ds = add_simulation_data_to_ds(new_ds, data_vars)

        tree_data_vars = [
            {"name": "GPP_tree", "data": GPP_tree},
            {"name": "NPP_tree", "data": NPP_tree},
            {"name": "R_A_tree", "data": R_A_tree},
            {"name": "R_G_tree", "data": R_G_tree},
            {"name": "CUE_tree", "data": CUE_tree},
            {"name": "mleaf_tree", "data": mleaf_tree},
            {"name": "GPP_per_leaf_biomass_tree", "data": GPP_per_leaf_biomass_tree},
            {"name": "NPP_per_leaf_biomass_tree", "data": NPP_per_leaf_biomass_tree},
            {"name": "r_BH_growth_5_yrs", "data": r_BH_growth_5_yrs},
            {"name": "tree_biomass_tree", "data": tree_biomass_tree},
            {"name": "RA_L_tree", "data": RA_L_tree},
            {"name": "RA_R_tree", "data": RA_R_tree},
            {"name": "RA_S_tree", "data": RA_S_tree},
            {"name": "thinning_or_cutting_tree", "data": thinning_or_cutting_tree},
            # single tree data vars
            {"name": "GPP_single_tree", "data": GPP_single_tree},
            {"name": "NPP_single_tree", "data": NPP_single_tree},
            {"name": "R_A_single_tree", "data": R_A_single_tree},
            {"name": "R_M_single_tree", "data": R_M_single_tree},
            {"name": "R_G_single_tree", "data": R_G_single_tree},
            {"name": "CUE_single_tree", "data": CUE_single_tree},
            {"name": "mleaf_single_tree", "data": mleaf_single_tree},
            {
                "name": "GPP_per_leaf_biomass_single_tree",
                "data": GPP_per_leaf_biomass_single_tree,
            },
            {
                "name": "NPP_per_leaf_biomass_single_tree",
                "data": NPP_per_leaf_biomass_single_tree,
            },
            {"name": "tree_biomass_single_tree", "data": tree_biomass_single_tree},
            {"name": "LA_single_tree", "data": LA_single_tree},
            {"name": "V_T_single_tree", "data": V_T_single_tree},
            {"name": "V_TH_single_tree", "data": V_TH_single_tree},
            {"name": "V_TS_single_tree", "data": V_TS_single_tree},
            {
                "name": "LabileC_assimilated_single_tree",
                "data": LabileC_assimilated_single_tree,
            },
            {
                "name": "LabileC_respired_single_tree",
                "data": LabileC_respired_single_tree,
            },
            {"name": "GL_single_tree", "data": GL_single_tree},
            {"name": "GR_single_tree", "data": GR_single_tree},
            {"name": "GS_single_tree", "data": GS_single_tree},
            {"name": "ML_single_tree", "data": ML_single_tree},
            {"name": "MR_single_tree", "data": MR_single_tree},
            {"name": "MS_single_tree", "data": MS_single_tree},
            {"name": "RA_L_single_tree", "data": RA_L_single_tree},
            {"name": "RA_R_single_tree", "data": RA_R_single_tree},
            {"name": "RA_S_single_tree", "data": RA_S_single_tree},
            {"name": "f_L_times_E_single_tree", "data": f_L_times_E_single_tree},
            {"name": "f_R_times_E_single_tree", "data": f_R_times_E_single_tree},
            {"name": "f_O_times_E_single_tree", "data": f_O_times_E_single_tree},
            {"name": "f_T_times_E_single_tree", "data": f_T_times_E_single_tree},
            #            {"name": "f_CS_times_CS_single_tree", "data": f_CS_times_CS_single_tree},
            {"name": "R_M_correction_single_tree", "data": R_M_correction_single_tree},
            {"name": "B_S_star_single_tree", "data": B_S_star_single_tree},
        ]

        new_ds = add_simulation_tree_data_to_ds(new_ds, tree_data_vars)

        # make time coordinate a simulation length in years
        new_ds = new_ds.assign_coords({"time": np.arange(len(ds.time))})
        attrs = new_ds.time.attrs
        attrs["units"] = "yr"
        new_ds.time.attrs.update(attrs)

        return new_ds


def compute_BTT_vars_and_add_to_ds(
    ds: xr.Dataset,
    dmr: DMR,
    dmr_eq: DLAPM,
    up_to_order: int,
    cache_size: int = False,
    verbose: bool = False,
) -> xr.Dataset:
    """Compute DMR quantities and add them to the simulation dataset.

    Quantities to compute:
        - backward transit time mean and second moment
    """
    if up_to_order != 2:
        raise ValueError("Current implementation only for first and second moment.")

    new_ds = ds.copy()

    # create dmr with inputs onlt based on dmr
    dmr_inputs_only = utils.create_dmr_inputs_only(dmr)

    # create dmrs (also inputs only) for trees only and for trees and soil (no WP)
    #    dmr_trees = dmr.restrict_to_pools(dmr.tree_pool_nrs)

    tree_and_soil_pool_nrs = np.append(dmr.tree_pool_nrs, dmr.soil_pool_nrs)
    dmr_trees_and_soil = dmr.restrict_to_pools(tree_and_soil_pool_nrs)

    #    dmr_trees_inputs_only = dmr_inputs_only.restrict_to_pools(dmr.tree_pool_nrs)
    dmr_trees_and_soil_inputs_only = dmr_inputs_only.restrict_to_pools(
        tree_and_soil_pool_nrs
    )

    # with cache the BTT computations are much faster
    if cache_size:
        dmr.initialize_state_transition_operator_matrix_cache(cache_size)
        dmr_inputs_only.initialize_state_transition_operator_matrix_cache(cache_size)
        dmr_trees_and_soil.initialize_state_transition_operator_matrix_cache(cache_size)
        dmr_trees_and_soil_inputs_only.initialize_state_transition_operator_matrix_cache(
            cache_size
        )

    # load start age data
    start_age_data = utils.load_start_age_data_from_eq_for_soil_and_wood_products(
        dmr, dmr_eq, up_to_order=up_to_order
    )

    # compute different mean transit times
    start_mean_age = start_age_data["start_age_moments_1"]
    start_mean_age_inputs_only = start_age_data["start_age_moments_1_inputs_only"]

    btt_mean = Q_(dmr.backward_transit_time_moment(1, start_mean_age), "yr")

    btt_mean_trees_and_soil = Q_(
        dmr_trees_and_soil.backward_transit_time_moment(
            1, start_mean_age[:, tree_and_soil_pool_nrs]  # type: ignore
        ),
        "yr",
    )

    btt_mean_inputs_only = Q_(
        dmr_inputs_only.backward_transit_time_moment(1, start_mean_age_inputs_only),
        "yr",
    )

    btt_mean_trees_and_soil_inputs_only = Q_(
        dmr_trees_and_soil_inputs_only.backward_transit_time_moment(
            1, start_mean_age_inputs_only[:, tree_and_soil_pool_nrs]  # type: ignore
        ),
        "yr",
    )

    # add BTT to new_ds
    stand_data_variables = [
        {"name": "btt_mean", "data": btt_mean},
        {"name": "btt_mean_trees_and_soil", "data": btt_mean_trees_and_soil},
        {"name": "btt_mean_inputs_only", "data": btt_mean_inputs_only},
        {
            "name": "btt_mean_trees_and_soil_inputs_only",
            "data": btt_mean_trees_and_soil_inputs_only,
        },
    ]
    new_ds = add_simulation_data_to_ds(new_ds, stand_data_variables)

    # compute different transit times standard deviations
    start_age_moments_2 = start_age_data["start_age_moments_2"]
    start_age_moments_2_inputs_only = start_age_data["start_age_moments_2_inputs_only"]

    btt_moment_2 = Q_(dmr.backward_transit_time_moment(2, start_age_moments_2), "yr^2")
    btt_sd = np.sqrt(btt_moment_2 - btt_mean**2)

    btt_moment_2_trees_and_soil = Q_(
        dmr_trees_and_soil.backward_transit_time_moment(
            2, start_age_moments_2[:, tree_and_soil_pool_nrs]  # type: ignore
        ),
        "yr^2",
    )
    btt_sd_trees_and_soil = np.sqrt(
        btt_moment_2_trees_and_soil - btt_mean_trees_and_soil**2
    )

    btt_moment_2_inputs_only = Q_(
        dmr_inputs_only.backward_transit_time_moment(
            2, start_age_moments_2_inputs_only
        ),
        "yr^2",
    )
    btt_sd_inputs_only = np.sqrt(btt_moment_2_inputs_only - btt_mean_inputs_only**2)

    btt_moment_2_trees_and_soil_inputs_only = Q_(
        dmr_trees_and_soil_inputs_only.backward_transit_time_moment(
            2, start_age_moments_2_inputs_only[:, tree_and_soil_pool_nrs]  # type: ignore
        ),
        "yr^2",
    )
    btt_sd_trees_and_soil_inputs_only = np.sqrt(
        btt_moment_2_trees_and_soil_inputs_only
        - btt_mean_trees_and_soil_inputs_only**2
    )

    stand_data_variables = [
        {"name": "btt_sd", "data": btt_sd},
        {"name": "btt_sd_trees_and_soil", "data": btt_sd_trees_and_soil},
        {"name": "btt_sd_inputs_only", "data": btt_sd_inputs_only},
        {
            "name": "btt_sd_trees_and_soil_inputs_only",
            "data": btt_sd_trees_and_soil_inputs_only,
        },
    ]
    new_ds = add_simulation_data_to_ds(new_ds, stand_data_variables)

    # BTT quantiles
    P0 = start_age_data["P0"]
    P0_inputs_only = start_age_data["P0_inputs_only"]

    P0_trees_and_soil = lambda ai: P0(ai)[tree_and_soil_pool_nrs]
    P0_trees_and_soil_inputs_only = lambda ai: P0_inputs_only(ai)[
        tree_and_soil_pool_nrs
    ]

    q = 0.5
    btt_median = Q_(dmr.backward_transit_time_quantiles(q, P0, verbose=verbose), "yr")
    btt_median_trees_and_soil = Q_(
        dmr_trees_and_soil.backward_transit_time_quantiles(
            q, P0_trees_and_soil, verbose=verbose
        ),
        "yr",
    )
    btt_median_inputs_only = Q_(
        dmr_inputs_only.backward_transit_time_quantiles(
            q, P0_inputs_only, verbose=verbose
        ),
        "yr",
    )
    btt_median_trees_and_soil_inputs_only = Q_(
        dmr_trees_and_soil_inputs_only.backward_transit_time_quantiles(
            q, P0_trees_and_soil_inputs_only, verbose=verbose
        ),
        "yr",
    )

    q = 0.05
    btt_quantile_05 = Q_(
        dmr.backward_transit_time_quantiles(q, P0, verbose=verbose), "yr"
    )
    btt_quantile_05_trees_and_soil = Q_(
        dmr_trees_and_soil.backward_transit_time_quantiles(
            q, P0_trees_and_soil, verbose=verbose
        ),
        "yr",
    )
    btt_quantile_05_inputs_only = Q_(
        dmr_inputs_only.backward_transit_time_quantiles(
            q, P0_inputs_only, verbose=verbose
        ),
        "yr",
    )
    btt_quantile_05_trees_and_soil_inputs_only = Q_(
        dmr_trees_and_soil_inputs_only.backward_transit_time_quantiles(
            q, P0_trees_and_soil_inputs_only, verbose=verbose
        ),
        "yr",
    )

    q = 0.95
    btt_quantile_95 = Q_(
        dmr.backward_transit_time_quantiles(q, P0, verbose=verbose), "yr"
    )
    btt_quantile_95_trees_and_soil = Q_(
        dmr_trees_and_soil.backward_transit_time_quantiles(
            q, P0_trees_and_soil, verbose=verbose
        ),
        "yr",
    )
    btt_quantile_95_inputs_only = Q_(
        dmr_inputs_only.backward_transit_time_quantiles(
            q, P0_inputs_only, verbose=verbose
        ),
        "yr",
    )
    btt_quantile_95_trees_and_soil_inputs_only = Q_(
        dmr_trees_and_soil_inputs_only.backward_transit_time_quantiles(
            q, P0_trees_and_soil_inputs_only, verbose=verbose
        ),
        "yr",
    )

    # legacy respiration/release
    P_sv = dmr.cumulative_pool_age_masses_single_value(P0)
    rho = 1 - dmr.Bs.sum(1)
    P_btt_sv = lambda ai, ti: (rho[ti] * P_sv(ai, ti)).sum()
    R = Q_(dmr.acc_net_external_output_vector().sum(-1), "gC/yr")

    x_ = dmr.times[:-1]
    go_through = tqdm(x_) if verbose else x_
    #    R_legacy = R - Q_(np.array([P_btt_sv(ti, ti) for ti in dmr.times[:-1]]), "gC/yr")
    R_legacy = R - Q_(np.array([P_btt_sv(ti, ti) for ti in go_through]), "gC/yr")

    stand_data_variables = [
        {"name": "btt_median", "data": btt_median},
        {"name": "btt_median_trees_and_soil", "data": btt_median_trees_and_soil},
        {"name": "btt_median_inputs_only", "data": btt_median_inputs_only},
        {
            "name": "btt_median_trees_and_soil_inputs_only",
            "data": btt_median_trees_and_soil_inputs_only,
        },
        {"name": "btt_quantile_05", "data": btt_quantile_05},
        {
            "name": "btt_quantile_05_trees_and_soil",
            "data": btt_quantile_05_trees_and_soil,
        },
        {"name": "btt_quantile_05_inputs_only", "data": btt_quantile_05_inputs_only},
        {
            "name": "btt_quantile_05_trees_and_soil_inputs_only",
            "data": btt_quantile_05_trees_and_soil_inputs_only,
        },
        {"name": "btt_quantile_95", "data": btt_quantile_95},
        {
            "name": "btt_quantile_95_trees_and_soil",
            "data": btt_quantile_95_trees_and_soil,
        },
        {"name": "btt_quantile_95_inputs_only", "data": btt_quantile_95_inputs_only},
        {
            "name": "btt_quantile_95_trees_and_soil_inputs_only",
            "data": btt_quantile_95_trees_and_soil_inputs_only,
        },
        {"name": "R", "data": R},
        {"name": "R_legacy", "data": R_legacy},
    ]
    new_ds = add_simulation_data_to_ds(new_ds, stand_data_variables)

    return new_ds


def compute_C_balance_and_CS_and_add_to_ds(
    ds: xr.Dataset, dmr: DMR, cache_size: int = False, verbose: bool = False
) -> xr.Dataset:
    """Compute DMR quantities and add them to the simulation dataset.

    Quantities to compute:
        - C balance (INCB) and CS (IITT)
    """
    new_ds = ds.copy()

    tree_and_soil_pool_nrs = np.append(dmr.tree_pool_nrs, dmr.soil_pool_nrs)
    dmr_trees_and_soil = dmr.restrict_to_pools(tree_and_soil_pool_nrs)

    # with cache the CS computation is much faster
    if cache_size:
        dmr.initialize_state_transition_operator_matrix_cache(cache_size)
        dmr_trees_and_soil.initialize_state_transition_operator_matrix_cache(cache_size)

    CS_through_time = Q_(dmr.CS_through_time(0, verbose=verbose), "gC / m^2 * yr")
    CS_through_time_trees_and_soil = Q_(
        dmr_trees_and_soil.CS_through_time(0, verbose=verbose), "gC / m^2 * yr"
    )

    C_balance_through_time = Q_(
        np.cumsum((dmr.net_Us - dmr.acc_external_output_vector()).sum(axis=1)),
        "gC / m^2",
    )

    C_balance_through_time_trees_and_soil = Q_(
        np.cumsum(
            (
                dmr_trees_and_soil.net_Us
                - dmr_trees_and_soil.acc_external_output_vector()
            ).sum(axis=1)
        ),
        "gC / m^2",
    )

    # add C_balance and CS to new_ds
    stand_data_variables = [
        {"name": "CS_through_time", "data": CS_through_time},
        {
            "name": "CS_through_time_trees_and_soil",
            "data": CS_through_time_trees_and_soil,
        },
        {"name": "C_balance_through_time", "data": C_balance_through_time},
        {
            "name": "C_balance_through_time_trees_and_soil",
            "data": C_balance_through_time_trees_and_soil,
        },
    ]
    new_ds = add_simulation_data_to_ds(new_ds, stand_data_variables)

    return new_ds


def get_simulation_record_ds(
    simulation: Simulation, additional_vars: Dict[str, Any]
) -> xr.Dataset:
    """Collect the basic variables recorded during a simulation.

    Args:
        simulation: the simulation considered
        additional_vars: add the additional variables returned by
            :meth:`~.library.run_recorded_simulation`
    """
    # collect stocks and fluxes in dataset
    ds = simulation.get_stocks_and_fluxes_dataset()

    tree_data_variables = [
        {"name": "radius_at_trunk_base", "data": additional_vars["rs"]},
        {"name": "height", "data": additional_vars["Hs"]},
        {"name": "DBH", "data": additional_vars["dbhs"]},
        {"name": "R_M_tree", "data": additional_vars["R_Ms"]},
        {"name": "LA_tree", "data": additional_vars["LAs"]},
        {"name": "V_T_tree", "data": additional_vars["V_Ts"]},
        {"name": "V_TH_tree", "data": additional_vars["V_THs"]},
        {"name": "V_TS_tree", "data": additional_vars["V_TSs"]},
        {
            "name": "LabileC_assimilated_tree",
            "data": additional_vars["LabileCs_assimilated"],
        },
        {"name": "LabileC_respired_tree", "data": additional_vars["LabileCs_respired"]},
        {"name": "GL_tree", "data": additional_vars["GLs"]},
        {"name": "GR_tree", "data": additional_vars["GRs"]},
        {"name": "GS_tree", "data": additional_vars["GSs"]},
        {"name": "ML_tree", "data": additional_vars["MLs"]},
        {"name": "MR_tree", "data": additional_vars["MRs"]},
        {"name": "MS_tree", "data": additional_vars["MSs"]},
        {"name": "f_L_times_E_tree", "data": additional_vars["f_L_times_Es"]},
        {"name": "f_R_times_E_tree", "data": additional_vars["f_R_times_Es"]},
        {"name": "f_O_times_E_tree", "data": additional_vars["f_O_times_Es"]},
        {"name": "f_T_times_E_tree", "data": additional_vars["f_T_times_Es"]},
        {"name": "f_CS_times_CS_tree", "data": additional_vars["f_CS_times_CSs"]},
        {"name": "coefficient_sum", "data": additional_vars["coefficient_sums"]},
        {"name": "R_M_correction_tree", "data": additional_vars["R_M_corrections"]},
        {"name": "B_S_star_tree", "data": additional_vars["B_S_stars"]},
        {"name": "C_S_star_tree", "data": additional_vars["C_S_stars"]},
        {"name": "rho_W", "data": additional_vars["rho_Ws"]},
        {"name": "delta_W", "data": additional_vars["delta_Ws"]},
        {"name": "SW", "data": additional_vars["SWs"]},
    ]
    ds = add_simulation_tree_dict_to_ds(ds, tree_data_variables)

    stand_data_variables = [
        {"name": "stand_basal_area", "data": additional_vars["stand_basal_area"]},
        {
            "name": "dominant_tree_height",
            "data": additional_vars["dominant_tree_height"],
        },
        {"name": "mean_tree_height", "data": additional_vars["mean_tree_height"]},
    ]
    ds = add_simulation_data_to_ds(ds, stand_data_variables)

    # add even more sophisticated variables
    ds = add_additional_simulation_quantities_to_ds(ds)

    return ds
