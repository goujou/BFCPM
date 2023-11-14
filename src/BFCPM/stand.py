# -*- coding: utf-8 -*-
"""
This module contains the :class:`~.stand.Stand` class, the central part of the model.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from bgc_md2.notebook_helpers import write_to_logfile

from . import Q_
from .management.management_strategy import (Cut, CutWaitAndReplant,
                                             ManagementStrategy, OnStandAges,
                                             Thin)
from .prepare_stand import (load_tree_soil_interface,
                            load_wood_product_interface)
from .productivity.constants import (EPS, MOLAR_MASS_H2O, PAR_TO_UMOL,
                                     micromolCO2_TO_gC)
from .productivity.radiation import canopy_sw_Spitters as sw_model_Spitters
from .productivity.radiation import canopy_sw_ZhaoQualls as sw_model_Zhao
from .productivity.waterbalance import Bucket, Canopywaterbudget
from .soil.soil_c_model_abc import SoilCModelABC
from .trees.mean_tree import MeanTree
from .trees.single_tree_allocation import SingleTree, TreeShrinkError
from .trees.single_tree_C_model import SingleTreeCModel
from .type_aliases import (CuttingFluxes, SpeciesParams, SpeciesSettings,
                           TreeExternalOutputFluxes, TreeSoilInterface,
                           WoodProductInterface)
from .utils import assert_accuracy, cached_property
from .wood_products.wood_product_model_abc import WoodProductModelABC


class Stand:
    """
    A stand is a compilation of trees, a soil model and a wood product model.

    It is used as basis to model microclimatic regimes (light, wind),
    canopy and soil water budget and call tree A-gs computations.
    Trees return their external output fluxes to the stand who in turn
    forwards soil inputs to the soil model.

    The productivity part of the stand runs of half-hourly basis,
    whereas the trees and the soil models are updated annualy.

    Args:
        z: grid of tree height layer boundaries [m]
        loc: location dictionary

            - "lat", "lon": latitude and longitude [deg]

        ctr: control dictionary

            - "phenology": include phenology?
            - "leaf_area": "seasonal LAI cycle?
            - "water_stress": include water stress?

        water_p: water_p dictionary

            - "interception":
                - "LAI": ??? [???]
                - "wmax": ??? [???]
                - "alpha": ??? [???]
            - "snowpack":
                - "Tm": ??? [???]
                - "Km": ??? [???]
                - "Kf": ??? [???]
                - "R": ??? [???]
                - "Tmax": ??? [???]
                - "Tmin": ??? [???]
                - "sliq": ??? [???]
                - "Sice": ??? [???]
            - "organic:layer":
                - "DM": ??? [???]
                - "Wmax": ??? [???]
                - "Wmin": ??? [???]
                - "Wcrit": ??? [???]
                - "alpha": ??? [???]
                - "Wliq": ??? [???]

        soil_p: soil bucket dictionary, can come from
            :mod:`~.params`

            - "depth": ??? [???]
            - "Ksat": ??? [???]
            - "pF": Hyytiala A-horizon pF curve
                - "ThetaS": ??? [???]
                - "ThetaR": ??? [???]
                - "alpha": ??? [???]
                - "n": ??? [???]
            - "MaxPond": ??? [???]
            - "Wliq": ??? [???]

        soil_model: carbon soil model to be used
        tree_soil_interface: connection dictionary, connect 
            key ``tree_pool`` to value ``soil_pool``
        wood_product_model: wood_product model to be used
        wood_product_interface: connection dictionary, connect 
            
            - ``tree_pool`` : {``pool``: ``fraction``, ...} with ``pool``
                from soil or wood product model

    Attributes:
        z
        dz (float): ``z[1]-z[0]``
        loc
        ctr
        soil_model
        tree_soil_interface
        wood_product_model
        wood_product_interface
        trees (List[:class:`~trees.mean_tree.MeanTree`]):
            the trees in the stand
        canopywater (:class:`.productivity.waterbalance.Canopywaterbudget`): ???
        soil (:class:`.productivity.waterbalance.Bucket`): ???
        times: the times at which the stand structure was updated during\
            a simulation run
    """

    def __init__(
        self,
        z: np.ndarray,
        loc: Dict[str, Any],
        ctr: Dict[str, bool],
        water_p: Dict[str, Dict[str, float]],
        soil_p: Dict[str, Any],
        #        soil_model_class: type, #SoilCModelABC,
        soil_model: SoilCModelABC,
        tree_soil_interface: TreeSoilInterface,
        wood_product_model: WoodProductModelABC,
        wood_product_interface: WoodProductInterface,
    ):
        self._yearly_cache: Dict[str, Any] = dict()
        self._daily_cache: Dict[str, Any] = dict()

        self.z = z
        self.dz = z[1] - z[0]
        self.loc = loc
        self.ctr = ctr

        # create empty stand
        self.trees: List[MeanTree] = []

        # add soil model
        self.soil_model = soil_model

        # connect trees and soil model
        self.tree_soil_interface = tree_soil_interface

        # add wood_product model
        self.wood_product_model = wood_product_model

        # connect cut trees and soil and wood product models
        self.wood_product_interface = wood_product_interface

        # water balance sub-model
        self.canopywater = Canopywaterbudget(water_p)

        # soil bucket sub-model
        self.soil = Bucket(soil_p)

        # ------ THESE ADDED 24.11.2021 / SL ---
        #
        self.Tdaily = 0.0
        self.fract_of_day = 0.0

        self.age = Q_("0 yr")

    # OK, HERE YOU CREATE MODEL INSTANCES BUT WHERE IS THIS CALLED?
    # this function is called in the notebook
    # then trees are added based on a desired stand configuration
    # happens also in the notebook
    @classmethod
    def create_empty(
        cls,
        stand_parameters: Dict[str, Any],
        wood_product_interface_name: str = "default",
    ) -> Stand:
        """Create a stand without trees.

        Args:
            stand_parameters: see :data:`.simulation_parameters.stand_params`
        """
        p = stand_parameters

        SingleTreeClass = p["SingleTreeClass"]
        soil_model = p["soil_model"]
        tree_soil_interface = load_tree_soil_interface(SingleTreeClass, soil_model)

        wood_product_model = p["wood_product_model"]
        wood_product_interface_name = p["wood_product_interface_name"]
        wood_product_interface = load_wood_product_interface(
            wood_product_interface_name, SingleTreeClass, soil_model, wood_product_model
        )

        return cls(
            p["z"].to("m").magnitude,
            p["loc"],
            p["ctr"],
            p["water_p"],
            p["soil_p"],
            soil_model,
            tree_soil_interface,
            wood_product_model,
            wood_product_interface,
        )

    def clear_cache(self):
        """Clear the yearly and daily caches."""
        self._yearly_cache: Dict[str, Any] = dict()
        self._daily_cache: Dict[str, Any] = dict()
        for tree_in_stand in self.trees:
            tree_in_stand.clear_cache()

    # TODO: announce tree status, soil model wood product model
    def __str__(self) -> str:
        s = [str(type(self))]
        s += [f"# trees: {len(self.trees)}"]
        N_per_m2 = sum(
            [tree.N_per_m2 for tree in self.trees if tree.current_status != "waiting"]
        )
        s += [f"Trees per m^2: {N_per_m2:5.5f}"]
        for tree in self.trees:
            s += ["\n" + str(tree)]

        return "\n".join(s)

    @property
    def nr_trees(self) -> int:
        """The number of MeanTrees in the stand, all statuses."""
        return len(self.trees)

    @property
    def nr_tree_pools(self) -> int:
        """The number of carbon pools per MeanTree."""
        if self.nr_trees == 0:
            raise ValueError("The stand has no trees associated to it.")

        return self.trees[0].nr_pools

    @property
    def nr_soil_pools(self) -> int:
        """The number of carbon pools in the soil carbon module."""
        return self.soil_model.nr_pools

    @property
    def nr_wood_product_pools(self) -> int:
        """The number of carbon pools in the wood product module."""
        return self.wood_product_model.nr_pools

    @property
    def tree_pool_nrs(self) -> Iterable:
        """The integer numbers of the tree pools in the whole system.

        Since the data set is in dims (entity, time, pool), the pools are
        E, B_L, C_L, ..., B_R, Litter, CWD, SOM, WP_S, WP_L.
        All entities have all the pools, for trees all non-tree pools contain
        nans, for soil, all non-soil pools contain nans, etc.
        """
        return range(0, self.nr_tree_pools, 1)

    @property
    def soil_pool_nrs(self) -> Iterable:
        """The integer numbers associated to the soil module."""
        return range(self.nr_tree_pools, self.nr_tree_pools + self.nr_soil_pools, 1)

    @property
    def wood_product_pool_nrs(self) -> Iterable:
        """The integer numbers associated to the wood product module."""
        return range(
            self.nr_tree_pools + self.nr_soil_pools,
            self.nr_tree_pools + self.nr_soil_pools + self.nr_wood_product_pools,
            1,
        )

    @property
    def tree_entity_nrs(self) -> List:
        """The entity numbers associated to MeanTrees."""
        return list(range(0, self.nr_trees, 1))

    @property
    def soil_entity_nr(self) -> int:
        """The entity number associated to the soil module."""
        return self.nr_trees

    @property
    def wood_product_entity_nr(self) -> int:
        """The entity number associated to the wood product module."""
        return self.soil_entity_nr + 1

    @property
    def living_trees(self) -> List[MeanTree]:
        """Return all living MeanTrees."""
        return [tree for tree in self.trees if tree.is_alive]

    @property
    def non_removed_trees(self) -> List[MeanTree]:
        """Return all trees with status not equal to "removed"."""
        return [tree for tree in self.trees if tree.current_status != "removed"]

    @property
    def basal_area(self) -> Q_[float]:
        """Stand basal area [m2 ha-1]."""
        if len(self.living_trees) == 0:
            return Q_(0, "m^2/ha")

        cross_section_areas = Q_.from_list(
            [
                np.pi * (tree.C_only_tree.tree.dbh.to("m") / 2) ** 2
                for tree in self.living_trees
            ]
        )
        Ns_per_ha = Q_(
            np.array([tree.N_per_m2 for tree in self.living_trees]), "1/m^2"
        ).to("1/ha")

        return (cross_section_areas * Ns_per_ha).sum()

    @cached_property("_yearly_cache")
    def dominant_tree_height(self) -> Q_[float]:
        """Average height of the tallest 100 trees [m]."""
        l = [(tree, tree.H) for tree in self.living_trees]
        l.sort(reverse=True, key=lambda x: x[1])

        #        if len(l) > 0:
        #            if l[0][0].N_per_m2 * 10_000 < 100:
        #                # need to combine more MeanTrees
        #                Ns = [el[0].N_per_m2 * 10_000 for el in l]
        #                if sum(Ns) < 100:
        #                    # less than 100 trees ha-1 in stand
        #                    return l[0][1]
        #
        #                Hs = [el[1] for el in l]
        #                nr = np.argmax(np.cumsum(Ns) >= 100)
        #                return np.average(Hs[: nr + 1], weights=Ns[: nr + 1])
        #
        #            return l[0][1]
        if len(l) > 0:
            if l[0][0].N_per_m2 * 10_000 < 100:
                # tallest MeanTree has less than 100 trees per ha
                Ns = [el[0].N_per_m2 * 10_000 for el in l]
                Hs = [el[1] for el in l]

                if sum(Ns) < 100:
                    # whole stand has less than 100 trees per ha
                    # take weighted average of all trees in stand
                    nr = len(l) - 1
                    return sum([Hs[i] * Ns[i] for i in range(nr + 1)]) / sum(
                        Ns[: nr + 1]
                    )
                else:
                    # there are more than 100 trees per ha in the stand
                    # nr of MeanTrees needed to reach 100 trees per ha
                    nr = np.argmax(np.cumsum(Ns) >= 100)
                    # number of trees in MeanTrees to stay under 100
                    s = sum(Ns[:nr])
                    # number of trees to fill up
                    missing = 100 - s
                    return (
                        sum(Hs[i] * Ns[i] for i in range(nr)) + Hs[nr] * missing
                    ) / 100
            else:
                return l[0][1]
        else:
            return Q_(np.nan, "m")

    @cached_property("_yearly_cache")
    def mean_tree_height(self) -> Q_[float]:
        """Mean tree height in the stand [m]."""
        if len(self.living_trees) == 0:
            return Q_(np.nan, "m")

        num = Q_(0.0, "m")
        denom = 0.0
        for tree in self.living_trees:
            num += tree.H * tree.N_per_m2
            denom += tree.N_per_m2

        return (num / denom).to("m")

    def add_trees_from_setting(
        self, species_settings: SpeciesSettings, custom_species_params: SpeciesParams
    ):
        """Add a prescribed collection of trees to the stand.

        Args:
            species_settings: settings for different tree species
            custom_species_params: tree species parameters (for all species)
        """
        tree_nrs = list()
        trees = list()
        stand_age = self.age.to("yr").magnitude
        for species, data in species_settings.items():
            #            print(species, data)
            for k, (tree_nr, dbh, density, ms, waiting) in enumerate(data):
                tree_nrs.append(tree_nr)

                N_per_m2 = density.to("1/m^2").magnitude

                tree_age = Q_(0, "yr")
                if waiting == "waiting":
                    first_action = ms.trigger_action_list[0][1]
                    if isinstance(first_action, CutWaitAndReplant):
                        tree_age = Q_(first_action.nr_waiting + 1, "yr")

                    status_list = ["waiting"] * stand_age + ["waiting"]
                else:
                    status_list = ["waiting"] * stand_age + ["assigned to: ['plant']"]

                single_tree = SingleTree.from_dbh(
                    species=species,
                    dbh=dbh,
                    Delta_t=Q_(1, "yr"),
                    custom_species_params=custom_species_params,
                    tree_age=tree_age,
                )

                C_only_tree = SingleTreeCModel(single_tree)
                nr_pools = C_only_tree.nr_pools
                tree_in_stand = MeanTree(
                    name=species + str(k),
                    C_only_tree=C_only_tree,
                    z=self.z,
                    loc=self.loc,
                    ctr=self.ctr,
                    management_strategy=ms,
                    base_N_per_m2=N_per_m2,
                    status_list=status_list,
                    N_per_m2_list=[0.0] * stand_age
                    + [N_per_m2 if not waiting else 0.0],
                    output_fluxes_list=[{}] * stand_age,
                    cutting_fluxes_list=[{}] * stand_age,
                    newly_planted_biomass_list=[Q_(np.zeros(nr_pools), "gC/m^2")]
                    * stand_age,
                )
                trees.append(tree_in_stand)

        # permute trees such that they appear in the order as given by
        # the species_settings
        for tree_nr_index in np.argsort(tree_nrs):
            self.trees.append(trees[tree_nr_index])

    # THIS IS CALLED FROM Simulation.run to:
    # 1) compute annual GPP of each plant, total stand GPP
    # 2) grow plants according to allocation models
    # 3) handle management actions, link stand litterfall & soil carbon dynamics
    # 4) check the overall C balance for this year

    def add_final_felling(self, max_stand_age: Q_):
        """Add the final felling (trigger, action) pair to all trees in the stand.

        Args:
            max_stand_age: age of stand at end of simulation
        """
        penultimate_stand_age = max_stand_age.to("yr").magnitude - 1
        for tree_in_stand in self.trees:
            # make space for the C transfer to WP
            trigger = OnStandAges(stand_ages=[Q_(penultimate_stand_age - 1, "yr")])
            action = Cut()
            ms_item = (trigger, action)
            # put final felling at the beginning (highest priority)
            ms = [ms_item] + tree_in_stand.management_strategy.trigger_action_list  # type: ignore
            tree_in_stand.management_strategy = ManagementStrategy(ms)  # type: ignore

    def simulate_year(
        self,
        year: int,
        forcing_of_year: pd.DataFrame,
        sw_model_name: str,
        logfile_path: Path,
    ) -> Tuple[List[str], Q_[float], float]:
        """Run one year of model simulation, the stand is the connecting module.

        Args:
            year [yr]
            forcing_of_year: the forcing data frame of the simulation year
            sw_model_name: light model, element of ["Spitters", "Zhao"]
            logfile_path: path to the logfile

        Returns:
            tuple

            - list with names of trees that could not be updated
            - GPP of the year [gC m-2 yr-1]
            - time: last time stamp from the forcing data
        """
        log = f"\n--- year={year}, stand_age={self.age} ---"
        print("\n" + log)
        write_to_logfile(logfile_path, log)

        dt = forcing_of_year.index[1] - forcing_of_year.index[0]

        # C in the system at the beginning of the time step
        C_system_pre = self.count_C_in_system(ignore=["assigned to: ['plant']"])

        # clear cache before yearly computation
        self._yearly_cache = dict()
        self._daily_cache = dict()

        self.process_cutting(logfile_path)
        self.process_planting(logfile_path)

        # process photosynthesis
        GPP_of_year = Q_(0, "gC/m^2/yr")
        #        for time, row in tqdm(forcing_of_year.iterrows(), total=len(forcing_of_year)):

        # This calls 1/2hourly computations and accumulates stand GPP
        # PRINT or plot: (1) is stand lad developing correctly during the year?
        # (2) is stand LAI == sum(tree LAI's) at all timesteps?
        # (3) is tree annual maximum LAI == tree leaf_mass x tree SLA?
        # (4) is correct N_per_m2 assigned to to each tree?
        # (5) does tree below large tree absorb less PAR per unit leaf area? print:
        #       sum(aPAR_sl x Lsl + aPAR_sh x Lsh) / sum(Lsl + Lsh)
        # (6) does it photosynthesize less per unit leaf area?
        #      sum(An_sl x Lsl + An_sh x Lsh) / sum(Lsl + Lsh)
        # --> did all the checks!

        for time, row in forcing_of_year.iterrows():
            if len(self.living_trees) > 0:
                forc = row.to_dict()
                tmp = self.run(dt.total_seconds(), forc, sw_model_name)
                GPP = tmp[0]
                # STAND GPP (gC m-2 (ground) yr-1)
                GPP_of_year += Q_(GPP * micromolCO2_TO_gC, "gC/m^2/yr")

        log = f"GPP_of_year: {GPP_of_year:2.4f}"
        #        print(log)
        #        print([t.LabileC_assimilated for t in self.trees])
        write_to_logfile(logfile_path, log)

        # IS IT HERE THE TREE GROWTH HAPPENS?
        # yes, here the trees grow if they are alive according to
        # tree_in_stand.LabileC_assimilated which was accmulated in the
        # half hourly run above
        failed_tree_names = self.process_trees_update(logfile_path)

        # check whether something went wrong like a tree cannot sustain itself
        if len(failed_tree_names) == 0:  # everything okay
            # process soil and WPs
            C_external_outputs = self.process_soil_and_WP_update()

            # check total mass balance
            C_external_inputs = GPP_of_year
            if C_external_inputs == 0:
                C_external_inputs = Q_(0, "gC/m^2/yr")
            C_system_post = self.count_C_in_system()

            # aggregate newly planted biomass
            C_newly_planted = sum(
                [sum(tree.newly_planted_biomass_list[-1]) for tree in self.trees]
            ) * Q_("1/yr")

            #            print(546)
            #            print(C_system_pre)
            #            print(C_system_post)
            #            print(C_system_post-C_system_pre)
            #            print()
            #            print(C_external_inputs)
            #            print(C_external_outputs)
            #            print(C_newly_planted)
            #            print(C_external_inputs-C_external_outputs+C_newly_planted)
            assert_accuracy(
                C_system_post - C_system_pre,
                (C_external_inputs - C_external_outputs + C_newly_planted) * Q_("1 yr"),
                1e-08,
            )

            # apply management strategies
            self.post_process_management(logfile_path)

            Delta_t = Q_("1 yr")
            self.age += Delta_t

        # this now returns Stand-level GPP ()
        return (
            failed_tree_names,
            GPP_of_year,
            time,
        )  # pylint: disable=undefined-loop-variable

    @cached_property("_daily_cache")  # depends on tree's relative leaf area
    def lad(self) -> np.ndarray:
        """The stand's leaf area density over ``self.z``, [m2 m-3].
    
        Returns:
            sum of the stand's trees leaf area densities\
            weighted by their abundance ``N_per_m2``, [m2 m-3]
        """
        lad = np.zeros(np.shape(self.z))
        for tree in self.living_trees:
            lad += tree.lad * tree.N_per_m2  # m2m-3

        return lad

    # depends on lad by computation, but actually only on lad_normed
    @cached_property("_yearly_cache")
    def canopy_height(self) -> float:
        """The stand's canopy height.

        Returns:
            highest point in the canopy where leaves can be found [m]
        """
        canopy_height = self.z[
            max(np.where(self.lad > 0)[0])
        ]  # pylint: disable=comparison-with-callable

        return canopy_height

    @cached_property("_daily_cache")  # depends on lad, hence of relative_leaf_area
    def LAI(self) -> float:
        """Stand's leaf area index [m2 (leaf) / m2 (ground)]."""
        return np.sum(self.lad * self.dz)

    @cached_property("_daily_cache")  # depends through LAI on lad
    def PARalbedo(self) -> float:
        """The stand's photosynthetically active radiation albedo [???]."""
        PARalbedo = 0.0
        for tree in self.living_trees:
            PARalbedo += tree.PARalbedo * tree.leaf_area * tree.N_per_m2

        return PARalbedo / self.LAI

    @cached_property("_daily_cache")  # depends through LAI on lad
    def Unorm(self):
        """The stand's ??? [???]."""
        return self.wind_profile(Uo=1.0)

    def run(self, dt: float, forcing: Dict[str, Any], sw_model_name: str) -> tuple:
        """The half-hourly run of the photosynthesis part.

        run for time step ``dt``

        Args:
            dt: time step [s]
            forcing: forcing dictionary

                - doy: day of year [-]
                - Zen: zenith angle [rad]
                - dirPar: direct PAR [umolm-2s-1]
                - diffPar: diffuse PAR [umolm-2s-1]
                - Tair: air temperature [degC]
                - H2O: H2O mixing ratio [mol/mol]
                - CO2: CO2 mixing ratio [ppm]
                - U: wind speed at canopy top [ms-1]
                - Prec: precipitation rate [kg m-2 s-1]
                - P: air pressure [Pa]

            sw_model_name: light model, element of ["Spitters", "Zhao"]

        Returns:
            tuple

                - GPP: ???
                - Gs: ???
                - Gs: ???
                - Ci: ???
                - E: ???
                - Ecan: ???
                - Ebl: ???
                - Trfall: ???
                - Roff: ???
                - Drain: ???
                - Infil: ???
        """
        self.fract_of_day += dt / 86400
        self.Tdaily += forcing["Tair"] * dt / 86400

        # daily loop to update phenology
        if self.fract_of_day >= 1:
            # update tree phenology and leaf-area
            for tree in self.living_trees:
                tree.update_daily(forcing["doy"], Tdaily=self.Tdaily, Rew=self.soil.REW)

            self.fract_of_day = 0.0
            self.Tdaily = 0.0

            self._daily_cache = dict()

        # solve wind profile
        forcing["U"] *= self.wind_profile(Uo=1.0)

        # solve light absorption
        aPar, f_sl, Par = self.light_absorption(
            forcing["Zen"], forcing["dirPar"], forcing["diffPar"], sw_model_name
        )

        forcing["f_sl"] = f_sl

        # --- solve gas-exchange of trees, accumulate to stand level.
        # tree.LabileC_assimilated & tree.LabileC_respired keep track on tree-level C-flux

        GPP, E, Gs, Ci = 0.0, 0.0, 0.0, 0.0

        for k, tree in enumerate(self.living_trees):
            # CHECK THAT k matches correct element in aPar;
            # i.e. is the tree order here same as in self.light_absorption
            # --> both loops go through the list self.living_trees,
            # so it's fine
            forcing.update({"aPar_sl": aPar["sunlit"][k], "aPar_sh": aPar["shaded"][k]})

            # compute Ags
            # A, Rd umol tree-1 s-1, fe mol tree-1 s-1, gs mol tree s-1, ci ppm
            (
                A,
                Rd,
                fe,
                gs,
                ci,
            ) = tree.compute_Ags(forcing)

            # update cumulative tree labile C pools: value is gC per single tree
            # I changed tree_in_stand.compute_Ags() so that now A is net CO2 assimilation,
            # i.e. photosynthesis - Rd, where Rd are respirative costs related to photosynthesis
            # (this should be logical, unless we cover Rd already as part of annual Rmaintenance)
            # --> absolutely perfect, this is how we wanted to have it,
            # no idea why it wasn't this way

            # general C uptake limitation factor
            # can be species dependent
            alpha = tree.C_only_tree.tree.params["alpha"].magnitude

            tree.LabileC_assimilated += alpha * A * micromolCO2_TO_gC * dt
            tree.LabileC_respired += Rd * micromolCO2_TO_gC * dt

            # on stand level, we should multiply them by dt (relevant ones)
            GPP += alpha * A * tree.N_per_m2 * dt  # umol m-2 timestep-1
            E += fe * tree.N_per_m2  # mol m-2 s-1

            # canopy conductance (mol m-2 (ground) s-1)
            Gs += gs * tree.N_per_m2
            # This could be removed if not used elsewhere in the code;
            # its units are arbitrary and we don't really need it
            Ci += ci * tree.N_per_m2

        E = E * MOLAR_MASS_H2O  # stand transpiration rate kg m-2 (ground) s-1

        # --- solve canopy water balance

        # net radiation of ground and canopy is used to model evaporation from interception
        # storages. Revise later; below is Global ~2 x Par, assume 80% of absorbed becomes net.
        Rng = 0.8 * 2.0 * Par[0] / PAR_TO_UMOL
        Rnc = 0.8 * 2.0 * (Par[-1] - Par[0]) / PAR_TO_UMOL

        wforc = {
            "Rnc": Rnc,
            "Rng": Rng,
            "Tair": forcing["Tair"],
            "Prec": forcing["Prec"],
        }

        # water fluxes kg m-2(ground) s-1
        Trfall, Ecan, Ebl = self.canopywater.run(dt, wforc)

        # compute root zone water balance
        Infil, Roff, Drain, mbe = self.soil.run(dt, rr=Trfall, et=E, latflow=0.0)

        return GPP, Gs, Ci, E, Ecan, Ebl, Trfall, Roff, Drain, Infil

    # ******** Microclimate functions **********
    def light_absorption(
        self, Zen: float, Qb0: float, Qd0: float, sw_model_name: str
    ) -> tuple:
        """
        Computes PAR (or NIR if added) absorption within multi-layer canopy

        Args:
            Zen: zenith angle [rad]
            Qb0: direct light at canopy top, in our case [umol m-2 s-1]
            Qd0: diffuse light at canopy top, in our case [umol m-2 s-1]
            sw_model_name: light model, element of ["Spitters", "Zhao"]

        Returns:
            tuple

            - aPAR (List[np.ndarray]): aPAR per each tree [umol m-2 (leaf) s-1]
            - f_sl (np.ndarray): sunlit fraction [-]
            - Q (float): incident PAR [umol m-2(ground) s-1]
        """
        LAIz = self.lad * self.dz

        if sw_model_name == "Spitters":
            # q_sl, q_sh: absorbed light umol m-2(leaf) s-1 at each layer
            Qb, Qd, Qsl, Qsh, q_sl, q_sh, _, f_sl, _ = sw_model_Spitters(
                LAIz,
                Clump=1.0,
                x=1.0,
                ZEN=Zen,
                IbSky=Qb0,
                IdSky=Qd0,
                LeafAlbedo=self.PARalbedo,
                SoilAlbedo=0.0,
            )
        elif sw_model_name == "Zhao":

            # TODO: Add parameter Clump=0.7 (clumping factor)
            # as input parameter instead of beign hard-coded
            Qb, Qd, Qdu, Qsl, Qsh, q_sl, q_sh, q_soil, f_sl, alb = sw_model_Zhao(
                LAIz,
                Clump=0.7,
                x=1.0,
                ZEN=Zen,
                IbSky=Qb0,
                IdSky=Qd0,
                LeafAlbedo=self.PARalbedo,
                SoilAlbedo=0.05,
            )
        else:
            raise ValueError("Unknown SW model")

        aPAR: Dict[str, List[float]] = {
            "sunlit": [],
            "shaded": [],
        }  # absorbed by leaves

        for tree in self.living_trees:
            # relative absorbed by each tree depends also on relative albedos
            f = self.PARalbedo / (
                tree.PARalbedo + EPS
            )  # * tree.lad * tree.N_per_m2 / (self.lad + EPS)
            aPAR["sunlit"].append(f * q_sl)  # per tree, per unit leaf area
            aPAR["shaded"].append(f * q_sh)  # per tree, unit leaf area

        return aPAR, f_sl, Qb + Qd

    def wind_profile(self, Uo: float = 1.0) -> np.ndarray:
        """
        Exponential attenuation of wind within canopy of uniform leaf-area density

        Args:
            Uo: flow at canopy_height [ms-1]

        Returns:
            flow at ``self.z`` [ms-1]
        """
        a = 0.5 * self.LAI  # frontal LAI
        U = Uo * np.exp(a * (self.z / self.canopy_height - 1.0))
        U[self.z >= self.canopy_height] = Uo  # pylint: disable=comparison-with-callable

        return U

    def process_cutting(self, logfile_path: Path):
        """Cut trees that are assigned to cutting. Also do thinning."""
        # interface of tree and wood products
        wood_product_interface = self.wood_product_interface

        #        # {tree_name: {(pool_from, pool_to): flux}}
        #        tree_cutting_fluxes_dict: Dict[str, Dict[tuple[str, str], Q_[float]]] = {}
        for tree_in_stand in self.trees:
            tree_name = tree_in_stand.name
            C_only_tree = tree_in_stand.C_only_tree
            new_N_per_m2 = tree_in_stand.N_per_m2

            do_cutting = False
            do_thinning = False
            current_status = tree_in_stand.current_status
            if current_status == "assigned to: ['cut']":
                do_cutting = True
                N_per_m2_stored = tree_in_stand.N_per_m2
                tree_in_stand.status_list[-1] = "removed"

            elif current_status.find("assigned to: ['cut',") != -1:
                do_cutting = True
                N_per_m2_stored = tree_in_stand.N_per_m2

                # remove 'cut' from the list
                i = current_status.find("[")
                s = current_status[: i + 1] + current_status[i + len("'cut', ") + 1 :]
                tree_in_stand.status_list[-1] = s

            elif current_status == "assigned to: ['thin']":
                do_thinning = True
                tree_in_stand.status_list[-1] = "thinned"

            if do_cutting and do_thinning:
                log = "Ignore thinning because of cutting."
                write_to_logfile(logfile_path, log)
                print(log)
                do_thinning = False

            tree_cutting_fluxes: CuttingFluxes = dict()
            if do_cutting:
                log = f"Cutting tree {tree_name}."
                print(log)
                write_to_logfile(logfile_path, log)
                mean_tree_cutting_fluxes = C_only_tree.get_cutting_fluxes(
                    wood_product_interface, logfile_path
                )

                tree_cutting_fluxes = {
                    k: v * Q_(N_per_m2_stored, "1/m^2")
                    for k, v in mean_tree_cutting_fluxes.items()
                }
                log = f"{tree_cutting_fluxes}"
                print(log)
                write_to_logfile(logfile_path, log)

                new_N_per_m2 *= 0

            if do_thinning:
                N_per_m2 = tree_in_stand.N_per_m2  # type: ignore
                new_N_per_m2 = tree_in_stand._new_N_per_m2  # type: ignore
                del tree_in_stand._new_N_per_m2  # type: ignore[attr-defined]

                log = f"Thinning {tree_name} from {N_per_m2} to {new_N_per_m2}."
                print(log)
                write_to_logfile(logfile_path, log)

                q = 1 - new_N_per_m2 / N_per_m2
                mean_tree_cutting_fluxes = C_only_tree.get_cutting_fluxes(
                    wood_product_interface, logfile_path
                )
                tree_cutting_fluxes = {
                    k: v * Q_(tree_in_stand.N_per_m2, "1/m^2") * q
                    for k, v in mean_tree_cutting_fluxes.items()
                }
                log = f"{tree_cutting_fluxes}"
                print(log)
                write_to_logfile(logfile_path, log)

            # always add cutting fluxes to tree's list
            tree_in_stand.cutting_fluxes_list.append(tree_cutting_fluxes)

            # always add N_per_m2 to tree's list
            tree_in_stand.N_per_m2_list.append(new_N_per_m2)

    def process_planting(self, logfile_path: Path):
        """Replant trees that are assigned to replanting and plant new trees."""
        for tree_nr, tree_in_stand in enumerate(self.trees):
            #            tree_name = tree_in_stand.name
            C_only_tree = tree_in_stand.C_only_tree

            planting_action = None
            if tree_in_stand.current_status == "assigned to: ['plant']":
                planting_action = "do plant"
                tree_in_stand.status_list[-1] = "newly_planted"
            elif tree_in_stand.current_status == "assigned to: ['replant']":
                planting_action = "do replant"
                tree_in_stand.status_list[-1] = "newly_planted"

            tree_newly_planted_biomass = Q_(np.zeros(C_only_tree.nr_pools), "gC/m^2")
            if planting_action == "do plant":
                # plant the tree
                # new tree comes with external C
                # take care of it in C budget
                newly_planted_biomass = Q_(C_only_tree.start_vector, "gC") * Q_(
                    tree_in_stand.base_N_per_m2, "1/m^2"
                )

                tree_newly_planted_biomass += newly_planted_biomass

                self.clear_cache()
                tree_in_stand.clear_cache()

                tree_in_stand.N_per_m2_list[-1] = tree_in_stand.base_N_per_m2
                log = f"Planting new tree\n{tree_in_stand}"
                print(log)
                write_to_logfile(logfile_path, log)

            elif planting_action == "do replant":
                # replant the tree
                single_tree = SingleTree.from_dbh(
                    species=tree_in_stand._new_species,  # type: ignore
                    dbh=tree_in_stand._new_dbh,  # type: ignore[attr-defined]
                    Delta_t=Q_(1, "yr"),
                    custom_species_params=tree_in_stand.C_only_tree.tree.custom_species_params,
                    tree_age=tree_in_stand._new_tree_age,  # type: ignore[attr-defined]
                )
                del tree_in_stand._new_species  # type: ignore[attr-defined]
                del tree_in_stand._new_dbh  # type: ignore[attr-defined]
                del tree_in_stand._new_tree_age  # type: ignore[attr-defined]

                new_C_only_tree = SingleTreeCModel(single_tree)
                # new tree comes with external C
                # take care of it in C budget
                newly_planted_biomass = Q_(new_C_only_tree.start_vector, "gC") * Q_(
                    tree_in_stand._new_N_per_m2, "1/m^2"
                )  # type: ignore[attr-defined]

                new_C_only_tree._xs = C_only_tree.xs.copy()
                new_C_only_tree._Us = C_only_tree.Us.copy()
                new_C_only_tree._Rs = C_only_tree.Rs.copy()
                new_C_only_tree._Fs = C_only_tree.Fs.copy()

                new_N_per_m2_list = tree_in_stand.N_per_m2_list[:-1] + [
                    tree_in_stand._new_N_per_m2  # type: ignore
                ]

                new_tree_in_stand = MeanTree(
                    tree_in_stand.name,
                    new_C_only_tree,
                    tree_in_stand.z,
                    tree_in_stand.loc,
                    tree_in_stand.ctr,
                    # take the old INSTANCES with its current values
                    tree_in_stand.management_strategy,
                    tree_in_stand.base_N_per_m2,
                    tree_in_stand.status_list,
                    new_N_per_m2_list,
                    tree_in_stand.output_fluxes_list,
                    tree_in_stand.cutting_fluxes_list,
                    tree_in_stand.newly_planted_biomass_list,
                )

                del tree_in_stand._new_N_per_m2  # type: ignore[attr-defined]

                log = f"Replanting tree\n{new_tree_in_stand}"
                print(log)
                write_to_logfile(logfile_path, log)
                self.trees[tree_nr] = new_tree_in_stand
                tree_in_stand = self.trees[tree_nr]

                tree_newly_planted_biomass += newly_planted_biomass

                self.clear_cache()

            # always add newly planted biomass to tree's list
            tree_in_stand.newly_planted_biomass_list.append(tree_newly_planted_biomass)

    # UPDATES TREE C POOLS AND GROWS TREES ONCE PER YEAR BASED ON C INPUT
    # and check the C balance for the single trees
    def process_trees_update(self, logfile_path: Path):
        """Update the carbon models of the trees."""

        # TREE CARBON POOLS PER TREE (gC plant-1)
        #
        # interface of tree and soil
        tree_soil_interface = self.tree_soil_interface

        Delta_t = Q_("1 yr")
        # update all trees in the stand

        # LOOP OVER ALL TREES IN STAND
        failed_tree_names = []
        for tree_in_stand in self.trees:
            tree_name = tree_in_stand.name
            # here is the initial state before any allocation of Labile_C_assimilated
            # Print these also after calling tree_in_stand.update_structure()!
            s = [f"{tree_name}:"]
            s += [f"dbh={tree_in_stand.dbh:~2.2f},"]
            s += [f"H={tree_in_stand.H:~2.2f},"]
            s += [
                f"C_alloc={tree_in_stand.LabileC_assimilated*tree_in_stand.N_per_m2:2.2f} gC/m2,"
            ]
            s += [f"C_alloc_per_plant={tree_in_stand.LabileC_assimilated:2.2f} gC,"]
            s += [f"sba={self.basal_area:~2.2f}"]
            s += [f"LA_plant={tree_in_stand.LA:~2.2f}"]
            s += [f"max_LA_cohort={tree_in_stand.max_leaf_area:2.2f}"]
            s += [f"LA_cohort={tree_in_stand.leaf_area:5.5f}"]
            s += [f"N_per_m2={tree_in_stand.N_per_m2:5.5f}"]
            log = " ".join(s)
            print("\n" + log)
            write_to_logfile(logfile_path, log)

            # we try, because a tree update might fail and we have to
            # repeat the previous year with some management action
            try:
                # so these should be per each tree from L1433
                N_per_m2 = Q_(tree_in_stand.N_per_m2, "1/m^2")
                C_only_tree = tree_in_stand.C_only_tree  # type: ignore
                old_status = tree_in_stand.current_status

                # C in the tree_in_stand before the update
                C_tree_pre = Q_(C_only_tree.xs[-1].sum(), C_only_tree.stock_unit) * Q_(
                    tree_in_stand.N_per_m2_list[-2], "1/m^2"
                )

                # HERE WE MOVE FROM "gC plant-1" to "gC m-2 (ground)
                # --> it's also for the C balance of tree_in_stand
                C_tree_input = Q_(tree_in_stand.LabileC_assimilated, "gC/yr") * N_per_m2

                tree_output_fluxes: TreeExternalOutputFluxes = dict()
                current_status = tree_in_stand.current_status

                if current_status in ["alive", "thinned"]:
                    new_status = "alive"
                    # in the end these are per tree (in glucose units) and trees are growing
                    # in the chain of function calls under the hood.
                    # note: uses object state tree_in_stand.LabileC_assimilated
                    # (gC plant-1) directly

                    # the outputs must be in gC plant-1 yr-1
                    mean_tree_output_fluxes = tree_in_stand.update_structure(
                        tree_soil_interface
                    )
                    # converted to gC m-2 ground a-1; i.e. represent fluxes from the tree cohort
                    tree_output_fluxes = {
                        k: v * N_per_m2 for k, v in mean_tree_output_fluxes.items()
                    }

                elif current_status == "newly_planted":
                    new_status = "alive"

                    C_only_tree.xs[-1] = C_only_tree.start_vector

                    mean_tree_output_fluxes = tree_in_stand.update_structure(
                        tree_soil_interface
                    )
                    tree_output_fluxes = {
                        k: v * N_per_m2 for k, v in mean_tree_output_fluxes.items()
                    }

                #                    # the following lines are for capturing C only in the
                #                    # first year; now we try to use C from x[0] in the
                #                    # first year, so  the tree can act regularly
                #                    # this conflicts with trees being removed before in the
                #                    # same time step, so a waiting year is required!
                ##                    newly_planted = tree_in_stand.newly_planted_biomass_list[-1]\
                ##                        / N_per_m2
                ##                    C_only_tree.capture_C_only(
                ##                        newly_planted,
                ##                        tree_in_stand.LabileC_assimilated
                ##                    )
                ##                    tree_in_stand.LabileC_assimilated = 0

                elif current_status == "removed":
                    new_status = "removed"
                    # do nothing, just append dummy values to output lists
                    C_only_tree.simulate_being_removed()

                elif current_status == "waiting":
                    new_status = "waiting"
                    # do nothing, just append dummy values to output lists
                    C_only_tree.simulate_waiting()

                elif current_status.find("assigned to: ['wait',") != -1:
                    tree_in_stand.status_list[-1] = "waiting"
                    # remove the initial 'wait'
                    i = current_status.find("[")
                    new_status = (
                        current_status[: i + 1]
                        + current_status[i + len("'wait', ") + 1 :]
                    )
                    # do nothing, just append dummy values to output lists
                    C_only_tree.simulate_being_removed()

                # always update tree status and output fluxes
                tree_in_stand.status_list.append(new_status)

                tree_in_stand.output_fluxes_list.append(tree_output_fluxes)

                C_tree_output = sum(tree_output_fluxes.values())
                if C_tree_output == 0:
                    C_tree_output = 0 * C_tree_input

                tree_cutting_fluxes: CuttingFluxes = tree_in_stand.cutting_fluxes_list[
                    -1
                ]
                C_tree_cut = sum(tree_cutting_fluxes.values())
                if C_tree_cut == 0:
                    C_tree_cut = 0 * C_tree_input

                C_tree_post = (
                    Q_(C_only_tree.xs[-1].sum(), C_only_tree.stock_unit) * N_per_m2
                )

                tree_newly_planted = tree_in_stand.newly_planted_biomass_list[-1]
                C_tree_newly_planted = tree_newly_planted.sum()
                if C_tree_newly_planted == 0:
                    C_tree_newly_planted = 0 * C_tree_pre

                if old_status == "waiting":
                    C_tree_pre = 0 * C_tree_pre

                #                print(1436)
                #                print(C_tree_pre)
                #                print(C_tree_post)
                #                print(C_tree_newly_planted)
                #                print(C_tree_input)
                #                print(C_tree_output)
                #                print(C_tree_cut)
                #                print(N_per_m2)

                if old_status != "newly_planted":
                    #                    print(1205, old_status)
                    #                    print(tree_in_stand.name)
                    #                    print(C_tree_pre)
                    #                    print(C_tree_post)
                    #                    print(C_tree_input)
                    #                    print(C_tree_output)
                    #                    print(C_tree_cut)
                    #                    print()
                    #                    print(C_tree_post - C_tree_pre)
                    #                    print((C_tree_input - C_tree_output - C_tree_cut) * Delta_t)
                    #                    print()
                    assert_accuracy(
                        C_tree_post - C_tree_pre,
                        (C_tree_input - C_tree_output - C_tree_cut) * Delta_t,
                        1e-08,
                    )
                else:
                    #                    print(tree_in_stand.name)
                    #                    print(C_tree_pre)
                    #                    print(C_tree_post)
                    #                    print(C_tree_input)
                    #                    print(C_tree_output)
                    assert_accuracy(
                        C_tree_post - C_tree_newly_planted,
                        (C_tree_input - C_tree_output) * Delta_t,
                        1e-08,
                    )
            except TreeShrinkError as error:
                # collect the failing trees and report them to
                # the function
                failed_tree_names.append(error.tree_name)

        return failed_tree_names

    def assign_emergency_action(
        self,
        tree_name: str,
        emergency_action_str: str,
        emergency_direction: str,
        emergency_q: float,  # fraction to keep
        nr_of_escalation: int,
        logfile_path: Path,
    ):
        """Decide what to do when a tree cannot sustain itself.

        Args:
            tree_name: the name of the MeanTree that failed to be updated
            emergency_action_str: element of ["Cut", "CutWait3AndReplant", "Thin", "Die"]

                - Cut: Cuts a number of trees from above or below
                - CutWait3AndReplant: Like "Cut" but with delayed replanting.
                - Thin: Thin the entire stand equally.
                - Die: Remove the unsustaibale tree.

            emergency_direction: ``above`` or ``below``, applies only for cutting
            emergency_q: Fraction of trees to keep on ``Thin``
            nr_of_escalation: If cutting one tree is not enough, cut two, and so on.
            logfile_path: path to the logfile
        """
        tree_names = [tree.name for tree in self.trees]
        tree_unsust = self.trees[tree_names.index(tree_name)]

        log = f"{tree_name} cannot sustain itself.\n"
        log += "We apply some management strategy."
        print(log)
        write_to_logfile(logfile_path, log)

        # thinning instead of cutting if only one living tree
        if (emergency_action_str == "Thin") or (len(self.living_trees) == 1):
            # thin them all, intensity depends on nr_of_escalation
            q = 1 - emergency_q**nr_of_escalation  # fraction to remove
            emergency_action = Thin(q)
            selected_trees = self.living_trees

        elif emergency_action_str in ["Cut", "CutWait3AndReplant"]:
            if emergency_action_str == "Cut":
                emergency_action = Cut()  # type: ignore
            if emergency_action_str == "CutWait3AndReplant":
                emergency_action = CutWaitAndReplant(nr_waiting=3)  # type: ignore

            if nr_of_escalation > len(self.living_trees):
                raise ValueError("Not enough trees to cut down for nr_of_escalation")

            living_trees_sorted_by_height_index = np.argsort(
                Q_.from_list([tree.H for tree in self.living_trees])
            )

            # nr of trees to cut down depends on nr_of_escalation
            if emergency_direction == "above":
                selected_living_trees_index = living_trees_sorted_by_height_index[
                    -nr_of_escalation:
                ]
            elif emergency_direction == "below":
                selected_living_trees_index = living_trees_sorted_by_height_index[
                    :nr_of_escalation
                ]
                print(1403, nr_of_escalation, selected_living_trees_index)
            else:
                raise ValueError(f"Unknwon emergency direction {emergency_direction}")

            selected_trees = [self.living_trees[i] for i in selected_living_trees_index]

        elif emergency_action_str == "Die":
            emergency_action = Cut()  # type: ignore
            selected_trees = [tree_unsust]

        else:
            raise ValueError(f"Unknown emergency action {emergency_action}")

        # assign emergency action to all selected trees
        for tree_in_stand in selected_trees:
            actions = emergency_action.do(self, tree_in_stand)
            new_old_status = f"assigned to: {actions}"
            log = f"Assigning tree '{tree_in_stand.name}' to {actions}."
            print(log)
            write_to_logfile(logfile_path, log)
            tree_in_stand.status_list[-1] = new_old_status

    def post_process_management(self, logfile_path: Path):
        """Check managament strategies, assign trees for actions next year."""
        Delta_t = Q_("1 yr")
        for tree_in_stand in self.trees:
            #            tree_name = tree_in_stand.name
            management_strategy = tree_in_stand.management_strategy
            actions, log = management_strategy.post_processing_update(
                self, tree_in_stand, Delta_t
            )
            #            tree_in_stand.actions = actions
            if len(actions):
                write_to_logfile(logfile_path, log)
                tree_in_stand.status_list[-1] = f"assigned to: {actions}"

    def process_soil_and_WP_update(self):
        """Transfer carbon from trees to soil and wood products."""
        # soil and wood product pool names
        soil_pool_names = self.soil_model.pool_names
        wood_product_pool_names = self.wood_product_model.pool_names

        # aggregate cutting fluxes
        tree_cutting_fluxes_dict = {
            tree.name: tree.cutting_fluxes_list[-1] for tree in self.trees
        }

        # pool_to: flux
        trees_cutting_fluxes_dict: Dict[str, Q_] = {}
        for tree_name, d in tree_cutting_fluxes_dict.items():
            for key, flux in d.items():
                _, pool_to = key
                if not pool_to in trees_cutting_fluxes_dict.keys():
                    trees_cutting_fluxes_dict[pool_to] = flux
                else:
                    trees_cutting_fluxes_dict[pool_to] += flux

        # aggregate trees' output fluxes
        tree_output_fluxes_dict = {
            tree.name: tree.output_fluxes_list[-1] for tree in self.trees
        }

        # aggregate tree output fluxes
        # {pool_to: flux}; pool_to = None means external outflux
        trees_output_fluxes_dict: Dict[str, Q_] = {}
        for tree_name, d in tree_output_fluxes_dict.items():
            for key, flux in d.items():
                _, pool_to = key
                if not pool_to in trees_output_fluxes_dict.keys():
                    trees_output_fluxes_dict[pool_to] = flux
                else:
                    trees_output_fluxes_dict[pool_to] += flux

        # add soil inputs together
        C_tree_external_outputs = Q_(0, "gC/m^2/yr")
        soil_input_fluxes: Dict[str, Q_] = {}
        for pool_to, flux in trees_output_fluxes_dict.items():
            if pool_to is not None:
                if not pool_to in soil_pool_names:
                    raise ValueError("Unknown destination for tree output.")

                if not pool_to in soil_input_fluxes.keys():
                    soil_input_fluxes[pool_to] = flux
                else:
                    soil_input_fluxes[pool_to] += flux
            else:
                C_tree_external_outputs += flux

        # move C from cut trees to soil and wood products
        wood_product_input_fluxes: Dict[str, Q_] = {}
        for pool_to, flux in trees_cutting_fluxes_dict.items():
            if pool_to in soil_pool_names:
                if not pool_to in soil_input_fluxes.keys():
                    soil_input_fluxes[pool_to] = flux
                else:
                    soil_input_fluxes[pool_to] += flux
            elif pool_to in wood_product_pool_names:
                if not pool_to in wood_product_input_fluxes.keys():
                    wood_product_input_fluxes[pool_to] = flux
                else:
                    wood_product_input_fluxes[pool_to] += flux
            else:
                raise ValueError(f"Invalid destination given for cut tree: {pool_to}")

        # update soil model
        soil_output_fluxes = self.soil_model.update(soil_input_fluxes)
        C_soil_outputs = sum(soil_output_fluxes.values())

        # update wood product model
        wood_product_output_fluxes = self.wood_product_model.update(
            wood_product_input_fluxes
        )
        C_wood_product_outputs = sum(wood_product_output_fluxes.values())

        return C_tree_external_outputs + C_soil_outputs + C_wood_product_outputs

    def count_C_in_trees(self, ignore: Optional[List[str]] = None) -> Q_[float]:
        """Sum up all carbon in trees, ignore some tree statuses.

        Args:
            ignore: list of tree statuses to be ignored in the sum
        """
        if ignore is None:
            ignore_set: Set[str] = set()
        else:
            ignore_set = set(ignore)

        tree_stock_unit = self.trees[0].C_only_tree.stock_unit
        return sum(
            [
                Q_(tree.C_only_tree.xs[-1].sum(), tree_stock_unit)
                * Q_(tree.N_per_m2, "1/m^2")
                for tree in self.trees
                if tree.current_status not in ignore_set
            ]
        )

    def count_C_in_system(self, ignore: Optional[List[str]] = None) -> Q_[float]:
        """Sum up all carbon in the system, ignore some tree statuses.

        Args:
            ignore: list of tree statuses to be ignored in the sum
        """
        C_system = self.count_C_in_trees(ignore)

        soil_model = self.soil_model
        C_system += Q_(soil_model.xs[-1].sum(), soil_model.stock_unit)

        WP_model = self.wood_product_model
        C_system += Q_(WP_model.xs[-1].sum(), WP_model.stock_unit)

        return C_system
