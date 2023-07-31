# -*- coding: utf-8 -*-
"""
This module contains the :class:`~.simulation.Simulation` class.

With instances if this class we can run simulations on a stand.
"""
from __future__ import annotations

import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
import pandas as pd
import xarray as xr
from bgc_md2.notebook_helpers import write_to_logfile
from tqdm import tqdm

from .. import Q_
from ..management.library import ma_data_to_ma
from ..stand import Stand
from ..trees.single_tree_allocation import GlucoseBudgetError, TreeShrinkError
from ..utils import array_to_slice


class Simulation:
    """Main class to run a simulation and store the associated data.

    Args:
        name: name of the simulation
        stand: stand to be used for simulation
        emergency_action_str: element of ["Cut", "CutWait3AndReplant", Thin", "Die"]

            - Cut: Cuts a number of trees from above or below
            - CutWait3AndReplant: Like "Cut" but with delayed replanting.
            - Thin: Thin the entire stand equally.
            - Die: Remove the unsustaibale tree.

        emergency_direction: `above` or `below`, applies only for cutting
        emergency_q: Fraction of trees to keep on "Thin"
    """

    # model main class
    def __init__(
        self,
        name: str,
        stand: Stand,
        emergency_action_str: str,
        emergency_direction: str,
        emergency_q: float,
        emergency_stand_action_str: str = "",
        times: List[Any] = None,
        GPP_totals: List[Q_[float]] = None,
        GPP_years: List[Q_[float]] = None,
    ):
        self.name = name
        self.stand = deepcopy(stand)
        self.emergency_action_str = emergency_action_str
        self.emergency_direction = emergency_direction
        self.emergency_q = emergency_q
        self.emergency_stand_action_str = emergency_stand_action_str

        if times is None:
            times = []
        #        self.history: dict[Any, Stand] = {}

        if GPP_totals is None:
            GPP_totals = []

        if GPP_years is None:
            GPP_years = []

        self.times = times
        self.GPP_totals = GPP_totals
        self.GPP_years = GPP_years

    def run(
        self,
        forcing: pd.DataFrame,
        sw_model_name: str,
        #        final_felling: bool,
        callback: Callable[[Stand], bool] = lambda stand: False,
        logfile_path: Path = None,
        show_pbar: bool = True,
    ):
        """Run the prepared simulation.

        Args:
            forcing: external forcing ``DataFrame``
            sw_model_name: light model, element of ["Spitters", "Zhao"]
            final_felling: clear stand at the end?
            callback: function with stand as argument, called after each simulation
                year; if retruns ``True``, then simulation stops
            logfile_path: path to the logfile
            show_pbar: to show or not to show a progress bar during the simulation
        """
        write_to_logfile(logfile_path, f"Starting: {self.name}")
        stand = self.stand

        self.times.append(forcing.index[0])

        if len(self.GPP_totals) == 0:
            GPP_total = Q_(0, "gC/m^2")
        else:
            GPP_total = self.GPP_totals[-1]

        last_year = forcing.index.year[-1]
        #        if final_felling:
        #            penultimate_stand_age = len(forcing.index.year.unique()) - 1
        #            for tree_in_stand in stand.trees:
        #                trigger = OnStandAges(stand_ages=[Q_(penultimate_stand_age - 1, "yr")])
        #                action = Cut()
        #                ms_item = (trigger, action)
        #
        #                tree_in_stand.management_strategy.add(ms_item)

        first_year = forcing.index.year[0]
        write_to_logfile(
            logfile_path,
            f"simulation from {first_year} to {last_year}",
            #            f"final felling: {final_felling}",
        )

        # The main idea is the following:
        # If a tree recognizes that it cannot sustain itself, a decision
        # must be made which tree to cut or to thin, the smallest one in the stand
        # or the tallest one. Once the decision is made, the year must be run
        # again with photosynhtesis under the new conditions with one
        # tree removed. Since C must be in the transient pool for at least
        # one year before it can be used for respiration or other allocation,
        # repeating one year is not enough, we must repeat even the year before.
        # Hence, the solution is the other way round:
        # We simulate a year and then tentatively simulate the next year.
        # If in this tentative simulation a tree cannot sustain itself, we rerun
        # the current year with the new tree setup
        # and throw the tentative simulation out.
        # If the tentative solution works out fine, we take it as the simulation
        # for the next year in order not to simulate the same year under
        # the same conditions twice.

        forcing_of_year = None
        year = None
        next_year_simulated = False

        # track which stand_tmp raised woh often, just for debugging reasons
        emergency_action_tracker: Dict[int, int] = dict()

        if show_pbar:
            iterator = tqdm(forcing.groupby(forcing.index.year))
        else:
            iterator = forcing.groupby(forcing.index.year)

        for next_year, forcing_of_next_year in iterator:

            if year is not None:  # jump over it in the first step
                log = str(year)
                write_to_logfile(logfile_path, log)

                callback(stand)

                success = False
                while not success:
                    if next_year_simulated:
                        # NOT RELEVANT - does this ever happen?
                        # --> it happens basically always, except for the cases
                        # when we have to jump back one year, because just in
                        # case we are prepared already to go one step forward
                        stand_tmp: Stand = next_stand_tmp
                        failing_tree_names = []
                        GPP_of_year = GPP_of_next_year
                        time = next_time
                    else:
                        # annual 1/2h computations called here
                        # GPP (umol m-2) of whole stand is returned.
                        stand_tmp = deepcopy(stand)
                        try:
                            (
                                failing_tree_names,
                                GPP_of_year,
                                time,
                            ) = stand_tmp.simulate_year(
                                year, forcing_of_year, sw_model_name, logfile_path
                            )
                        except GlucoseBudgetError:
                            tb = traceback.TracebackException.from_exception(error)
                            write_to_logfile(logfile_path, "".join(tb.stack.format()))
                            print("GBE in TIS")
                        except TreeShrinkError as error:
                            tb = traceback.TracebackException.from_exception(error)
                            write_to_logfile(logfile_path, "".join(tb.stack.format()))
                            raise error.__class__(self.name)
                        except AssertionError as error:
                            tb = traceback.TracebackException.from_exception(error)
                            write_to_logfile(logfile_path, "".join(tb.stack.format()))
                            msg = f"Failed to update tree '{self.name}'"
                            raise AssertionError(msg) from error

                    # simulation: the second loop step
                    if failing_tree_names:
                        print(failing_tree_names, "want to shrink")

                    # tempatively run next year
                    GPP_of_next_year: Q_[float] = 0.0
                    next_time: float = 0.0
                    if not failing_tree_names:
                        next_stand_tmp: Stand = deepcopy(stand_tmp)
                        (
                            failing_tree_names,
                            GPP_of_next_year,
                            next_time,
                        ) = next_stand_tmp.simulate_year(
                            next_year, forcing_of_next_year, sw_model_name, logfile_path
                        )

                    if not failing_tree_names:
                        success = True
                        next_year_simulated = True
                    else:
                        next_year_simulated = False
                        # check only first failing tree
                        # the others should profit from cutting

                        # consider only the first failing tree,
                        # the rest will follow later automatically by nr_of_escalation
                        n = 1
                        for tree_name in failing_tree_names[:n]:
                            if id(stand) in emergency_action_tracker.keys():
                                emergency_action_tracker[id(stand)] += 1
                            else:
                                emergency_action_tracker[id(stand)] = 1

                            nr_of_escalation = emergency_action_tracker[id(stand)]
                            stand.assign_emergency_action(
                                tree_name,
                                self.emergency_action_str,
                                self.emergency_direction,
                                self.emergency_q,
                                nr_of_escalation,
                                logfile_path,
                            )

                        # assign management actions to remaining trees if asked for
                        log_thinning = []
                        if self.emergency_stand_action_str:
                            # fake stand without removed emergency trees
                            fake_stand = deepcopy(stand)
                            remaining_tree_list = [
                                tree
                                for tree in stand.trees
                                if tree.name not in failing_tree_names[:n]
                            ]
                            fake_stand.trees = remaining_tree_list
                            for tree_in_stand in stand.trees:
                                if not tree_in_stand.is_alive:
                                    continue

                                if tree_in_stand.name not in failing_tree_names[:n]:
                                    management_action = ma_data_to_ma(
                                        self.emergency_stand_action_str
                                    )
                                    # use the fake stand for an SBA without the
                                    # unsustainable trees
                                    actions = management_action.do(
                                        fake_stand, tree_in_stand
                                    )
                                    # without this check, if there's no action to be taken, I ruin the status of the living trees nevertheless
                                    if len(actions):
                                        log_thinning += [
                                            f"{tree_in_stand.name}, actions: {actions}"
                                        ]
                                        tree_in_stand.status_list[
                                            -1
                                        ] = f"assigned to: {actions}"

                        log = [
                            f"Repeating year {year} with managed stand"
                        ] + log_thinning
                        log = "\n".join(log)
                        print(log)
                        write_to_logfile(logfile_path, log)

                stand = stand_tmp

                # accumulate stand total GPP, append tables of annual GPP
                GPP_total = GPP_total + GPP_of_year * Q_("1 yr")
                self.GPP_totals.append(GPP_total)
                self.GPP_years.append(GPP_of_year)
                self.times.append(time)
            #                self.history[time] = deepcopy(stand)

            year = next_year
            forcing_of_year = forcing_of_next_year
            # end of annual loop

        # don't forget to really apply the already computed last year
        stand = next_stand_tmp
        # accumulate stand total GPP, append tables of annual GPP
        GPP_total = GPP_total + GPP_of_next_year * Q_("1 yr")
        self.GPP_totals.append(GPP_total)
        self.GPP_years.append(GPP_of_next_year)
        self.times.append(next_time)
        #        self.history[next_time] = deepcopy(stand)

        callback(stand)

        log = f"Done: {self.name}"
        print(log)
        write_to_logfile(logfile_path, log)

        self.stand = stand

    def get_stocks_and_fluxes_dataset(
        self, times: List[pd.Timestamp] = None
    ) -> xr.Dataset:
        """Combine stocks and fluxes from trees, soil, and wood products to one common dataset.

        Since trees in the class :class:`~..trees.mea_tree.MeanTree` represent
        single (mean) trees, we have to weigh their stocks and fluxes in the
        stand model by their ``N_per_m2``.
        Consequently, the stand C model is in units of gC/m2.

        Returns:
            ``xarray.Dataset`` given by

            .. code-block:: python

                coords = {
                    "entity": [tree names] + ["soil", "wood_product"],
                    "entity_to": [tree names] + ["soil", "wood_product"],
                    "entity_from": [tree names] + ["soil", "wood_product"],
                    "time": dates at which structure was updated
                    "pool": pool names of tree and soil and wood product
                    "pool_to": pool names of tree and soil and wood product
                    "pool_from": pool names of tree and soil and wood product
                    "tree": tree_names
                }

                data_vars = {
                    "nr_trees": self.nr_trees,
                    "nr_tree_pools": self.nr_tree_pools,
                    "nr_soil_pools": self.nr_soil_pools,
                    "nr_wood_product_pools": nr_wood_product_pools,
                    "tree_pool_nrs": self.tree_pool_nrs,
                    "soil_pool_nrs": self.soil_pool_nrs,
                    "wood_product_pool_nrs": self.wood_product_pool_nrs,
                    "tree_entity_nrs": self.tree_entity_nrs,
                    "soil_entity_nr": self.soil_entity_nr,
                    "wood_product_entity_nr": self.wood_product_entity_nr,

                    "GPP_total": xr.DataArray(
                        data_vars=self.GPP_totals,
                        dims=["entity", "time"],
                        attrs={
                            "units": stock unit [gC m-2],
                            "cell_methods": "time: total"
                        }
                    ),

                    "GPP_year": xr.DataArray(
                        data_vars=self.GPP_years,
                        dims=["entity", "time"],
                        attrs={
                            "units": stock unit [gC m-2 yr-1],
                            "cell_methods": "time: total"
                        }
                    ),

                    "N_per_m2": xr.DataArray(
                        data_vars=Ns_per_m2,
                        dims=["tree", "time"],
                        attrs={
                            "units": [m-2],
                            "cell_methods": "time: instantaneous"
                        }
                    ),

                    "newly_planted_biomass": xr.DataArray(
                        data_vars=newly_panted biomass,
                        dims=["time"],
                        attrs={
                            "units": stock unit [gC m-2],
                            "cell_methods": "time: total"
                        }
                    ),

                    "stocks": xr.DataArray(
                        data_vars=stocks data,
                        dims=["entity", "time", "pool"],
                        attrs={
                            "units": stock unit [gC m-2],
                            "cell_methods": "time: instantaneous"
                        }
                    ),

                    "input_fluxes": xr.DataArray(
                        data_vars=input fluxes data,
                            mean over time step,
                        dims=["entity_to", "time", "pool_to"],
                        attrs={
                            "units": flux unit [gC m-2 yr-1],
                            "cell_methods": "time: total"
                        }
                    ),

                    "output_fluxes": xr.DataArray(
                        data_vars=output fluxes data,
                            mean over time step,
                        dims=["entity_from", "time", "pool_from"],
                        attrs={
                            "units": flux unit [gC m-2 yr-1],
                            "cell_methods": "time: total"
                        }
                    ),

                    "internal_fluxes": xr.DataArray(
                        data_vars=internal fluxes data,
                            mean over time step,
                        dims=["entity_to", "entity_from", "time", "pool_to", "pool_from"],
                        attrs={
                            "units": flux unit [gC m-2 yr-1],
                            "cell_methods": "time: total"
                        }
                    )
                }
        """
        stand = self.stand

        # get the stocks and fluxes from the tree and the soil and the wood product
        # insert an entity dimension at the beginning to
        # identify the trees and the soil and the wood product
        ds_trees = xr.merge(
            [
                tree.get_stocks_and_fluxes_dataset().expand_dims(
                    dim={"entity": [tree.name]}, axis=0  # type: ignore
                )
                for tree in stand.trees
            ]
        )

        # xarray.merge sorts the trees by their name, this makes
        # the ordering of the trees inconsistent with the ordering
        # in stand.trees
        # -> we reindex the entity dimension such that the order in
        # ds_trees is made consistent to stand.trees
        tree_names = [tree.name for tree in stand.trees]
        ds_trees = ds_trees.reindex({"entity": tree_names})

        # soil model
        ds_soil = stand.soil_model.get_stocks_and_fluxes_dataset()
        ds_soil = ds_soil.expand_dims(dim={"entity": ["soil"]}, axis=0)  # type: ignore

        # inferface between tree and soil
        tree_soil_interface = stand.tree_soil_interface

        # wood product model
        ds_wood_product = stand.wood_product_model.get_stocks_and_fluxes_dataset()
        ds_wood_product = ds_wood_product.expand_dims(
            dim={"entity": ["wood_product"]}, axis=0  # type: ignore
        )

        # set up the coordinates of the combined/new dataset
        new_entities = ds_trees.coords["entity"].data.tolist() + [
            "soil",
            "wood_product",
        ]
        tree_pools = ds_trees.coords["pool"].data.tolist()
        soil_pools = ds_soil.coords["pool"].data.tolist()
        wood_product_pools = ds_wood_product.coords["pool"].data.tolist()
        new_pools = tree_pools + soil_pools + wood_product_pools

        if times:
            times = np.array([t.to_datetime64() for t in times])
        else:
            times = np.array([t.to_datetime64() for t in self.times])

        tree_names = [tree.name for tree in stand.trees]

        new_coords = {
            "entity": new_entities,
            "entity_to": new_entities,
            "entity_from": new_entities,
            "time": times,
            "pool": new_pools,
            "pool_to": new_pools,
            "pool_from": new_pools,
            "tree": tree_names,
        }

        nr_trees = len(ds_trees.coords["entity"])
        nr_wood_product_pools = len(wood_product_pools)

        nr_new_times = len(new_coords["time"])
        nr_new_entities = len(new_coords["entity"])
        nr_new_pools = len(new_pools)

        tree_entity_slice = array_to_slice(stand.tree_entity_nrs)
        soil_entity_nr = stand.soil_entity_nr
        wood_product_entity_nr = stand.wood_product_entity_nr

        tree_pool_slice = array_to_slice(stand.tree_pool_nrs)
        soil_pool_slice = array_to_slice(stand.soil_pool_nrs)
        wood_product_pool_slice = array_to_slice(stand.wood_product_pool_nrs)

        # create newly_planted_biomass
        data = np.nan * np.ones((nr_new_entities, nr_new_times, nr_new_pools))
        for tree_nr, tree_in_stand in enumerate(stand.trees):
            entity_nr = stand.tree_entity_nrs[tree_nr]
            tree_newly_planted_biomass = np.array(
                [x.magnitude for x in tree_in_stand.newly_planted_biomass_list]
            )
            data[entity_nr, :-1, tree_pool_slice] = tree_newly_planted_biomass

        newly_planted_biomass = xr.DataArray(
            data=data, dims=["entity", "time", "pool"], attrs=ds_trees.stocks.attrs
        )

        # create tree species
        data = np.array([tree.species for tree in stand.trees])
        speciess = xr.DataArray(data=data, dims=["tree"])

        # create N_per_m2
        data = np.nan * np.ones((nr_trees, nr_new_times))
        for tree_nr, tree_in_stand in enumerate(stand.trees):
            data[tree_nr, :] = ds_trees.N_per_m2.isel(entity=tree_nr)

        Ns_per_m2 = xr.DataArray(
            data=data, dims=["tree", "time"], attrs=ds_trees.N_per_m2.attrs
        )

        # create GPP_total
        data = np.nan * np.ones(nr_new_times)
        q = Q_.from_list(self.GPP_totals)
        data[: len(q)] = q.magnitude
        GPP_total = xr.DataArray(
            data=data,
            dims=["time"],
            attrs={"cell_methods": "time: total", "units": q.get_netcdf_unit()},
        )

        # create GPP_year
        data = np.nan * np.ones(nr_new_times)
        q = Q_.from_list(self.GPP_years)
        data[: len(q)] = q.magnitude
        GPP_year = xr.DataArray(
            data=data,
            dims=["time"],
            attrs={"cell_methods": "time: mean", "units": q.get_netcdf_unit()},
        )

        # create combined stocks
        data = np.nan * np.ones((nr_new_entities, nr_new_times, nr_new_pools))

        # copy tree stocks and soil stocks and wood product stocks
        data[tree_entity_slice, :, tree_pool_slice] = ds_trees.stocks
        data[soil_entity_nr, :, soil_pool_slice] = ds_soil.stocks
        data[
            wood_product_entity_nr, :, wood_product_pool_slice
        ] = ds_wood_product.stocks

        new_stocks = xr.DataArray(
            data=data, dims=["entity", "time", "pool"], attrs=ds_trees.stocks.attrs
        )

        # create combined external input fluxes
        data = np.nan * np.ones((nr_new_entities, nr_new_times, nr_new_pools))

        # external input fluxes to trees do not change
        data[tree_entity_slice, :, tree_pool_slice] = ds_trees.input_fluxes

        # there are no external input fluxes to the soil and the wood products
        data[soil_entity_nr, :-1, soil_pool_slice] = 0
        data[wood_product_entity_nr, :-1, wood_product_pool_slice] = 0

        new_input_fluxes = xr.DataArray(
            data=data,
            dims=["entity_to", "time", "pool_to"],
            attrs=ds_trees.input_fluxes.attrs,
        )

        # create combined external output fluxes
        data = np.nan * np.ones((nr_new_entities, nr_new_times, nr_new_pools))

        # external output fluxes from soil and wood products do not change
        data[soil_entity_nr, :, soil_pool_slice] = ds_soil.output_fluxes
        data[
            wood_product_entity_nr, :, wood_product_pool_slice
        ] = ds_wood_product.output_fluxes

        # first copy external output fluxes, then remove those going to the soil
        data[tree_entity_slice, :, tree_pool_slice] = ds_trees.output_fluxes
        for pool_from in tree_soil_interface.keys():
            data[tree_entity_slice, :-1, new_pools.index(pool_from)] = 0

        new_output_fluxes = xr.DataArray(
            data=data,
            dims=["entity_from", "time", "pool_from"],
            attrs=ds_trees.output_fluxes.attrs,
        )

        # create combined internal fluxes
        data = np.zeros(
            (nr_new_entities, nr_new_entities, nr_new_times, nr_new_pools, nr_new_pools)
        )

        # internal fluxes within trees and within soil and within wood products do not change
        # they stay within their respective entity
        for nr_entity in stand.tree_entity_nrs:
            entity = new_entities[nr_entity]
            data[
                nr_entity, nr_entity, :, tree_pool_slice, tree_pool_slice
            ] = ds_trees.internal_fluxes.sel(entity=entity)

        data[
            soil_entity_nr, soil_entity_nr, :, soil_pool_slice, soil_pool_slice
        ] = ds_soil.internal_fluxes

        data[
            wood_product_entity_nr,
            wood_product_entity_nr,
            :,
            wood_product_pool_slice,
            wood_product_pool_slice,
        ] = ds_wood_product.internal_fluxes

        # move external output fluxes from trees that connect to the soil to internal fluxes
        for nr_entity in stand.tree_entity_nrs:
            entity = new_entities[nr_entity]
            for pool_from, d in tree_soil_interface.items():
                for pool_to, fraction in d.items():
                    #                  print(pool_from, pool_to, new_pools.index(pool_from), new_pools.index(pool_to))
                    data[
                        soil_entity_nr,
                        nr_entity,
                        :,
                        new_pools.index(pool_to),
                        new_pools.index(pool_from),
                    ] = ds_trees.output_fluxes.sel(entity=entity, pool_from=pool_from)

        # add fluxes from cutting trees
        for entity_nr in stand.tree_entity_nrs:
            entity = new_entities[entity_nr]
            tree_nr = entity_nr - stand.tree_entity_nrs[0]
            tree_in_stand = stand.trees[tree_nr]
            for k, d in enumerate(tree_in_stand.cutting_fluxes_list):  # type: ignore
                for (pool_from, pool_to), flux in d.items():
                    if pool_to in soil_pools:
                        data[
                            soil_entity_nr,
                            entity_nr,
                            k,
                            new_pools.index(pool_to),
                            new_pools.index(pool_from),
                        ] += flux.magnitude  # type: ignore
                    elif pool_to in wood_product_pools:
                        data[
                            wood_product_entity_nr,
                            entity_nr,
                            k,
                            new_pools.index(pool_to),
                            new_pools.index(pool_from),
                        ] += flux.magnitude  # type: ignore
                    else:
                        raise (
                            ValueError(
                                "No destination given for pool {pool_from} from cut tree."
                            )
                        )

        new_internal_fluxes = xr.DataArray(
            data=data,
            dims=["entity_to", "entity_from", "time", "pool_to", "pool_from"],
            attrs=ds_trees.internal_fluxes.attrs,
        )

        # create combined dataset
        new_data_vars = {
            "nr_trees": stand.nr_trees,
            "nr_tree_pools": stand.nr_tree_pools,
            "nr_soil_pools": stand.nr_soil_pools,
            "nr_wood_product_pools": nr_wood_product_pools,
            "tree_pool_nrs": stand.tree_pool_nrs,
            "soil_pool_nrs": stand.soil_pool_nrs,
            "wood_product_pool_nrs": stand.wood_product_pool_nrs,
            "tree_entity_nrs": stand.tree_entity_nrs,
            "soil_entity_nr": stand.soil_entity_nr,
            "wood_product_entity_nr": stand.wood_product_entity_nr,
            "species": speciess,
            "N_per_m2": Ns_per_m2,
            "GPP_total": GPP_total,
            "GPP_year": GPP_year,
            "newly_planted_biomass": newly_planted_biomass,
            "stocks": new_stocks,
            "input_fluxes": new_input_fluxes,
            "output_fluxes": new_output_fluxes,
            "internal_fluxes": new_internal_fluxes,
        }

        new_ds = xr.Dataset(
            data_vars=new_data_vars,  # type: ignore
            coords=new_coords,  # type: ignore
        )
        new_ds.time.attrs = {"units": "yr"}

        return new_ds
