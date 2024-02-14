"""
Contains the :class:`~.mean_tree.MeanTree` that belong to a :class:`~.stand.Stand`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import xarray as xr

from .. import Q_
from ..management.management_strategy import ManagementStrategy
from ..params import global_tree_params
from ..productivity.constants import (AIR_VISCOSITY, EPS,
                                      MOLECULAR_DIFFUSIVITY_CO2,
                                      MOLECULAR_DIFFUSIVITY_H2O, afact)
from ..productivity.phenology import LAI_cycle, Photo_cycle
from ..productivity.photo import photo_c3_medlyn_farquhar as A_gs
from ..type_aliases import (CuttingFluxes, TreeExternalOutputFluxes,
                            TreeSoilInterface)
from ..utils import cached_property
from .single_tree_allocation import GlucoseBudgetError, TreeShrinkError
from .single_tree_C_model import SingleTreeCModel


class MeanTree:
    """Class for a MeanTree in a stand.

    This is initialized by :meth:`~.stand.Stand.add_tree`
    and adds particular stand parameters to the MeanTree. A MeanTree
    represents ``N_per_m2`` identical trees in the stand.

    Args:
        name: name of the MeanTree
        C_only_tree: :class:`~.single_tree_C_model.SingleTreeCModel`,
        z: grid of tree height layer boundaries [m]
        loc: location dictionary

            - "lat", "lon": latitude and longitude [deg]

        ctr: control dictionary

            - "phenology": include phenology?
            - "leaf_area": "seasonal LAI cycle?
            - "water_stress": include water stress?

        management_strategy: instance of
            :class:`~..management.management_strategy.ManagamentStrategy`
        base_N_per_m2: number of trees per ha to be planted usually
            represented [m2-1]
        status_list: list of tree history, values allowed:

            - "waiting": not yet to be planted, combine with planting action
            - "assigned to: ['plant']": to be planted
            - "assigned to: ['wait', ..., 'plant']: wait and then pant

        output_fluxes_list: history of external output fluxes
        cutting_fluxes_list: history of fluxes caused by cutting
        newly_planted_biomass_list: history of newly planted biomass
    """

    # I dont understand why a tree object has lists as parameters?
    # --> it's a little weird but at the moment it gives the flexible
    # option to initialize a tree with a fake history of a certain length
    def __init__(
        self,
        name: str,
        C_only_tree: SingleTreeCModel,
        z: np.ndarray,
        loc: Dict[str, Any],
        ctr: Dict[str, Any],
        management_strategy: ManagementStrategy,
        base_N_per_m2: float,
        status_list: List[str],
        N_per_m2_list: List[float],  # [ha-1]
        output_fluxes_list: List[TreeExternalOutputFluxes] = None,
        cutting_fluxes_list: List[CuttingFluxes] = None,
        newly_planted_biomass_list: List[Q_] = None,
        custom_global_tree_params: Dict[str, Any] = None,
    ):
        self._cache: Dict[str, Any] = dict()
        self.daily_cache: Dict[str, Any] = dict()

        self.name = name
        self.C_only_tree = C_only_tree
        #        self.params = global_tree_params[self.species]

        self._custom_global_tree_params = custom_global_tree_params

        if custom_global_tree_params is None:
            self.params = global_tree_params[self.species]
        else:
            self.params = custom_global_tree_params[self.species]

        self.clear_cache()

        self.z = z
        self.dz = z[1] - z[0]
        self.loc = loc
        self.ctr = ctr

        p = self.params
        # phenology model
        self.Pheno_Model = Optional[Photo_cycle]
        if self.ctr["phenology"]:
            #            # phenology model instance
            #            self.Pheno_Model = Photo_cycle(p['phenop']) # type: ignore
            #            # phenology state [0...1]
            #            self.tree._pheno_state = self.tree.Pheno_Model.f # type: ignore
            self.Pheno_Model = Photo_cycle(p["phenop"])
        else:
            #            self.tree._pheno_state = 1.0 # type: ignore
            self.Pheno_Model = None

        # seasonality of leaf area
        self.LAI_Model = Optional[LAI_cycle]
        if self.ctr["leaf_area"]:
            #            # LAI model instance
            #            self.tree._LAI_Model = LAI_cycle(p['LAIp'], loc) # type: ignore
            #            # LAI relative to annual maximum [0...1]
            #            self.tree._relative_leaf_area = self.tree.LAI_Model.f # type: ignore
            self.LAI_Model = LAI_cycle(p["laip"], loc)
        else:
            #            self.tree._relative_leaf_area = 1.0 # type: ignore
            self.LAI_Model = None

        # A-gs parameters at pheno_state = 1.0 (dict)
        self.photop0: Dict[str, Any] = self.params["photop"]

        # this can vary with phenology and root zone water stress
        self.photop = self.photop0.copy()  # current A-gs parameters (dict)

        # cumulative flux to/from labile pool (gC); two separate fluxes to
        # deal with mass balance in SingleTree
        self.LabileC_assimilated = 0.0
        self.LabileC_respired = 0.0

        self.management_strategy = management_strategy

        self.base_N_per_m2 = base_N_per_m2
        self.N_per_m2_list = N_per_m2_list
        self.status_list = status_list
        if output_fluxes_list is not None:
            self.output_fluxes_list = output_fluxes_list
        else:
            self.output_fluxes_list = list()

        if cutting_fluxes_list is not None:
            self.cutting_fluxes_list = cutting_fluxes_list
        else:
            self.cutting_fluxes_list = list()

        if newly_planted_biomass_list is not None:
            self.newly_planted_biomass_list = newly_planted_biomass_list
        else:
            self.newly_planted_biomass_list = list()

        # create internal variables needed when replanting
        self._new_N_per_m2: Union[None, float] = None
        self._new_species: Union[None, str] = None
        self._new_dbh: Union[None, float] = None
        self._new_tree_age: Union[None, float] = None

    def clear_cache(self):
        """Clear daily and yearly caches."""
        self._daily_cache: Dict[str, Any] = dict()
        self._yearly_cache: Dict[str, Any] = dict()
        self.C_only_tree.tree.clear_cache()

    def __str__(self):
        s = [str(type(self))]
        s += [f"name: {self.name}"]
        s += [f"dbh: {self.C_only_tree.tree.dbh:~P2.3f}"]
        s += [f"height: {self.C_only_tree.tree.H:~P2.3f}"]
        N = Q_(self.N_per_m2, "1/m^2")
        s += [f"N_per_m2: {N:2.3f}"]
        base_N = Q_(self.base_N_per_m2, "1/m^2")
        s += [f"base_N_per_m2: {base_N:2.3f}"]
        mleaf = self.C_only_tree.tree.B_L
        s += [f"leaf biomass: {mleaf:5.5f} per plant, {(mleaf*N):5.5f}"]
        LA = self.C_only_tree.tree.LA
        s += [
            f"leaf area: {LA:5.5f} per plant, {(LA*N).magnitude:5.5f} per tree cohort"
        ]
        s += [f"current status: {self.current_status}"]
        s += [str(self.management_strategy)]

        ##        s += [""] + ["C_only_tree"] + ["-----------"] + [""]
        #        s += [reindent(str(self.C_only_tree), 4)]
        ##        s += [str(self.C_only_tree.tree)]
        return "\n".join(s)

    @property
    def nr_pools(self):
        """Number of carbon pools."""
        return self.C_only_tree.nr_pools

    @property
    def live_statuses(self) -> List[str]:
        """All tree status that count the MeanTree as being alive."""
        return ["alive", "newly_planted", "thinned", "assigned to: ['thin']"]

    @property
    def is_alive(self) -> bool:
        """MeanTree is alive or not?"""
        return (self.current_status in self.live_statuses) or (
            self.current_status.find("assigned to: ['cut'") != -1
        )

    @property
    def r(self) -> Q_[float]:
        """Tree radius at trunk base if MeanTree is alive [m]."""
        return self.C_only_tree.tree.r * float(self.is_alive)

    @property
    def dbh(self) -> Q_[float]:
        """Diameter at breast height if Meantree is alive [cm]."""
        return self.C_only_tree.tree.dbh * float(self.is_alive)

    @property
    def LA(self) -> Q_[float]:
        """Leaf area if MeanTree is alive [m2 ha-1]."""
        return self.C_only_tree.tree.LA * float(self.is_alive)

    @property
    def species(self) -> str:
        """MeanTree species."""
        return self.C_only_tree.species

    @property  # not (yearly) cachable at thinning events
    def N_per_m2(self) -> float:
        """Number of identical trees represented by MeanTreem if alive [m-2]."""
        return self.N_per_m2_list[-1] * float(self.is_alive)

    #    @cached_property("_yearly_cache")
    #    def SLA(self) -> float:
    #        """Specific leaf area [m2 kg_dw-1]."""
    #        SLA = self.C_only_tree.SLA
    #        return SLA.to("m^2/kg_dw").magnitude

    @property
    def basal_area(self) -> Q_[float]:
        """Stand basal area associated to this MeanTree [m2 ha-1]."""
        cross_section_area = np.pi * (self.C_only_tree.tree.dbh.to("m") / 2) ** 2
        N_per_ha = Q_(self.N_per_m2, "1/m^2").to("1/ha")

        return cross_section_area * N_per_ha

    @cached_property("_yearly_cache")
    def mleaf(self) -> float:
        """Leave biomass [kg_dw]."""
        mleaf = self.C_only_tree.B_L
        return mleaf.to("kg_dw").magnitude * float(self.is_alive)

    @cached_property("_yearly_cache")
    def lad_normed(self) -> np.ndarray:
        """Normalized leaf area density over grid ``z``, integrates to 1, [m-1].

        Args:
            z: grid of tree height layer boundaries [m]

        Returns:
            normalized leaf area density of grid ``z``, [m-1]
        """
        z = self.z
        lad_n = self.C_only_tree.lad_normed(z)
        return lad_n.magnitude * float(self.is_alive)

    @property  # don't cache!
    def H(self) -> Q_[float]:
        """Tree height [m]."""
        return self.C_only_tree.H.to("m") * float(self.is_alive)

    @cached_property("_daily_cache")  # depends through leaf_area on relative_leaf_area
    def lad(self) -> np.ndarray:
        """Leaf area density ofer the height grid [m2 m-1]."""
        lad = self.leaf_area * self.lad_normed
        return lad * float(self.is_alive)

    @cached_property("_yearly_cache")
    def max_leaf_area(self) -> float:
        """Maximum leaf area based on leaf biomass [m2]."""
        #        return self.SLA * self.mleaf
        return self.C_only_tree.LA.to("m^2").magnitude

    @cached_property("_daily_cache")
    def relative_leaf_area(self):
        """Relative leaf area based on LAI model [-]."""
        return self.LAI_Model.f if self.LAI_Model is not None else 1.0

    @cached_property("_daily_cache")  # depends on relative_leaf_area
    def leaf_area(self):
        """Actual leaf area [m2]."""
        return self.max_leaf_area * self.relative_leaf_area

    @cached_property("_yearly_cache")
    def leaf_size(self) -> float:
        """Leaf length [m]."""
        return self.params["leaf"]["length"]

    @property
    def current_status(self) -> str:
        """Current status of the MeanTree."""
        return self.status_list[-1]

    @cached_property("_daily_cache")
    def pheno_state(self) -> float:
        """Phenology state [0...1]"""
        return self.Pheno_Model.f if self.Pheno_Model is not None else 1.0  # type: ignore

    @cached_property("_yearly_cache")
    def PARalbedo(self) -> float:
        """The tree's photosynthetically active radiation albedo [???]."""
        return self.params["leaf"]["PARalbedo"]

    # THIS IS CALLED FROM stand ONCE PER YEAR AND TREE C POOLS AND NEW DIMENSIONS ARE COMPUTED
    # --> if the tree is alive
    def update_structure(
        self, tree_soil_interface: TreeSoilInterface
    ) -> TreeExternalOutputFluxes:
        """Update the tree structure and return output fluxes.

        Args:
            tree_soil_interface: interface between the tree and soil modules

        Returns:
            external output fluxes dictionary, stand can then distribute them to soil
        """
        try:
            # CALL TO UDATE C POOLS AND GROW
            tree_output_fluxes = self.C_only_tree.update_gC_no_unit(
                self.LabileC_assimilated, tree_soil_interface  # THIS IS IN gC plant-1
            )
        except GlucoseBudgetError:
            print("GBE in TIS")
        except TreeShrinkError as error:
            raise error.__class__(self.name)
        except AssertionError as error:
            msg = f"Failed to update tree '{self.name}'"
            raise AssertionError(msg) from error

        # store for recording during simulation
        self._LabileC_assimilated = self.LabileC_assimilated
        self._LabileC_respired = self.LabileC_respired

        # reset for next year
        self.LabileC_assimilated = 0.0
        self.LabileC_respired = 0.0

        # --- at begining of each year, set phenology and LAI models to initial stage
        self.Pheno_Model.reset()  # type: ignore[attr-defined]
        self.LAI_Model.reset()  # type: ignore[attr-defined]

        # clear cache for new tree structure
        self.clear_cache()

        return tree_output_fluxes

    def update_daily(self, doy: int, Tdaily: float, Rew: float):
        """Update phenological state and water stress.

        Effects on gas-exchange parameters, seasonal leaf area, etc.

        Args:
            doy: day of year
            Tdaily: daily temperature
            Rew: relatively extractable water
        """
        if self.ctr["phenology"]:
            # update photosynthetic parameters
            # updates self.Pheno_Model.f and hence selph.pheno_state
            self.Pheno_Model.run(Tdaily)  # type: ignore[attr-defined]

            # scale photosynthetic capacity using vertical N gradient
            f = 1.0
            if "kn" in self.photop0:
                kn = self.photop0["kn"]

                # Holger: DO WE NEED lad per plant or per m2 here???
                Lc = np.flipud(np.cumsum(np.flipud(self.lad * self.dz)))
                Lc = Lc / np.maximum(Lc[0], EPS)
                f = np.exp(-kn * Lc)

            # preserve proportionality of Jmax and Rd to Vcmax
            self.photop["Vcmax"] = f * self.pheno_state * self.photop0["Vcmax"]
            self.photop["Jmax"] = f * self.pheno_state * self.photop0["Jmax"]
            self.photop["Rd"] = f * self.pheno_state * self.photop0["Rd"]

        if self.ctr["leaf_area"]:
            # update leaf_area development during the year
            # update self.LAI_model.f and hence self.relative_leaf_area
            self.LAI_Model.run(doy, Tdaily)  # type: ignore[attr-defined]

        if self.ctr["water_stress"]:
            # drought responses from Hyde scots pine shoot chambers, 2006; for 'Medlyn - model' only
            # stomatal parameter
            b = self.photop["drp"]
            fm = np.minimum(1.0, (Rew / b[0]) ** b[1])
            self.photop["g1"] = fm * self.photop0["g1"]

            # apparent Vcmax decrease with Rew but proportionality is preserved.
            fv = np.minimum(1.0, (Rew / b[2]) ** b[3])
            self.photop["Vcmax"] *= fv
            self.photop["Jmax"] *= fv
            self.photop["Rd"] *= fv

        #        if self.Switch_WaterStress == 'PsiL':
        #            PsiL = np.minimum(-1e-5, PsiL)
        #            b = self.photop0['drp']
        #
        #            # medlyn g1-model, decrease with decreasing Psi
        #            self.photop['g1'] = self.photop0['g1'] * np.maximum(0.05, np.exp(b*PsiL))
        #
        #            # Vmax and Jmax responses to leaf water potential. Kellomäki & Wang, 1996.
        #            # (Note! mistake in paper eq's, these correspond to their figure)
        #            fv = 1.0 / (1.0 + (PsiL / - 2.04)**2.78)  # vcmax
        #            fj = 1.0 / (1.0 + (PsiL / - 1.56)**3.94)  # jmax
        #            fr = 1.0 / (1.0 + (PsiL / - 2.53)**6.07)  # rd
        #            self.photop['Vcmax'] *= fv
        #            self.photop['Jmax'] *= fj
        #            self.photop['Rd'] *= fr

        self._daily_cache = dict()

    def compute_Ags(self, forcing: Dict[str, Any]) -> tuple:
        """
        computes leaf gas-exchange

        Args:
            forcing (dict):

                - aPar_sl (array): absorbed Par on sunlit [umol m-2(leaf) s-1
                - aPar_sh (array): absorbed Par on shaded [umol m-2(leaf) s-1]
                - f_sl: sunlit fraction [-]
                - T: (leaf)temperature [degC]
                - D: vapor pressure deficit [mol/mol]
                - Ca: CO2 mixing ratio [ppm]]
                - P: atm. pressure [Pa]
                - U: wind speed [ms-1]
        Returns:
            tuple

            - A: photosynthesis [umol (tree) s-1]
            - Rd: maintenance respiration [umol (tree) s-1]
            - E: transpiration rate [mol (tree) s-1]
            - Gs: stomatal conductance [mol (tree) s-1]
            - Ci: leaf internal CO:math:`_2` [-]
        """
        aPar_sl = forcing["aPar_sl"]
        aPar_sh = forcing["aPar_sh"]
        f_sl = forcing["f_sl"]
        T = forcing["Tair"]
        D = forcing["D"]
        P = forcing["P"]
        Ca = forcing["CO2"]
        U = np.maximum(forcing["U"], EPS)

        # boundary-layer conductance for CO2 and H2O

        factor1 = 1.4  # *2  #1.4 is correction for turbulent flow

        # -- Adjust diffusivity, viscosity, and air density to pressure/temp.
        t_adj = (101300.0 / P) * ((T + 273.15) / 293.16) ** 1.75
        Da_v = MOLECULAR_DIFFUSIVITY_H2O * t_adj
        Da_c = MOLECULAR_DIFFUSIVITY_CO2 * t_adj
        va = AIR_VISCOSITY * t_adj
        rho_air = 44.6 * (P / 101300.0) * (273.15 / (T + 273.13))  # [mol/m3]

        # ----- Compute the leaf-level dimensionless groups
        Re = U * self.leaf_size / va  # Reynolds number
        Sc_v = va / Da_v  # Schmid numbers for water
        Sc_c = va / Da_c  # Schmid numbers for CO2

        gb_c = (
            factor1
            * (0.664 * rho_air * Da_c * Re**0.5 * (Sc_c) ** 0.33)
            / self.leaf_size
        )  # [mol/m2/s]
        gb_v = (
            factor1
            * (0.664 * rho_air * Da_v * Re**0.5 * (Sc_v) ** 0.33)
            / self.leaf_size
        )  # [mol/m2/s]

        # solve leaf gas-exchange for sunlit & shaded leaves at each layer, integrate to tree level
        # uses current state of tree.photop;
        # thus do adjustments for water stress and phenology externally

        # print(aPar_sl, T, D, Ca)
        # A_gs returns fluxes per m-2 (leaf) s-1
        # CONSISTENCY CHECK: must be an_sh <= an_sl
        an_sl, rd_sl, fe_sl, gsc_sl, ci_sl, _ = A_gs(
            self.photop, aPar_sl, T, D, Ca, gb_c, gb_v, P=101300.0
        )
        an_sh, rd_sh, fe_sh, gsc_sh, ci_sh, _ = A_gs(
            self.photop, aPar_sh, T, D, Ca, gb_c, gb_v, P=101300.0
        )

        assert np.all(an_sh <= an_sl + 1e-12)  # give it some tolerance

        #        plt.figure(99)
        #        plt.subplot(221); plt.plot(an_sl)
        #        plt.subplot(222); plt.plot(rd_sl, 'o') # this is float
        # unless T or photop['Rd'] is array
        #        plt.subplot(223); plt.plot(gsc_sl)
        #        plt.subplot(224); plt.plot(ci_sl)

        # upscale to tree-level
        # sunlit (Lsl) and shaded (Lsh) leaf area (m2) in layers
        #        dz = z[1] - z[0]
        #        Lsl = f_sl * self.lad(z) * dz
        #        Lsh = (1 - f_sl) * self.lad(z) * dz
        Lsl = f_sl * self.lad * self.dz
        Lsh = (1 - f_sl) * self.lad * self.dz

        An = sum(Lsl * an_sl + Lsh * an_sh)  # net CO2 umol (tree) s-1
        Rd = sum(Lsl * rd_sl + Lsh * rd_sh)  # respired

        # CONSISTENCY CHECK: FOR EACH TREE, PRINT An / (sum(Lsl + Lsh)).
        # This must be smaller for shaded trees

        # SL changed 01.02.2022
        # A = max(0.0, An+Rd) # assimilated CO2 umol (tree) s-1
        A = max(0.0, An)  # net CO2 umol (tree) s-1,
        # now respirative costs associated with photosynthesis are accounted for.
        E = sum(Lsl * fe_sl + Lsh * fe_sh)

        # big-leaf stomatal conductance for CO2 & Ci per tree
        Gs = E / (D * afact + EPS)  # mol s-1
        Ci = min(Ca, Ca - An / (Gs + EPS))  # ppm

        return A, Rd, E, Gs, Ci

    def get_stocks_and_fluxes_dataset(self, stand_age: int = 0) -> xr.Dataset:
        """Multiply mean tree stocks and fluxes by ``N_per_m2``.

        Call :meth:`~single_tree_C_model.SingleTreeCModel.get_stocks_and_fluxes_dataset`
        and multiply all stocks and fluxes with ``N_per_m2``.
        Add Ns_per_m2.

        Returns:
            stocks and fluxes dataset on stand level, Ns_per_m2

            - stock unit: gC m-2
            - flux unit: gC m-2 yr-1
            - Ns_per_m2 unit: m-2
        """
        ds_mean_tree = self.C_only_tree.get_stocks_and_fluxes_dataset()

        data_vars = dict()
        Ns_per_m2 = Q_(np.array(self.N_per_m2_list), "1/m^2")
        for name, variable in ds_mean_tree.data_vars.items():
            if name == "stocks":
                v = variable.copy()
                q = Q_(v.data, v.attrs["units"])
                tup = (-1,) + (1,) * (q.ndim - 1)

                # extend if tree was planted later
                empty_q = np.zeros((len(Ns_per_m2) - q.shape[0], q.shape[1]))
                empty_q = Q_(empty_q, q.units)
                extended_q = np.concatenate([empty_q, q])
                new_q = extended_q * Ns_per_m2.reshape(tup)

                attrs = v.attrs.copy()
                attrs["units"] = new_q.get_netcdf_unit()
                new_v = xr.DataArray(data=new_q.magnitude, dims=v.dims, attrs=attrs)
                #                v.data = new_q.magnitude
                #                v.attrs["units"] = new_q.get_netcdf_unit()
                data_vars[name] = new_v
            else:
                v = variable.copy()
                q = Q_(v.data, v.attrs["units"])
                Ns_per_m2_shifted = Q_(
                    np.roll(Ns_per_m2.magnitude, shift=-1), Ns_per_m2.units
                )
                tup = (-1,) + (1,) * (q.ndim - 1)

                # extend if tree was planted later
                empty_q = np.zeros(
                    (len(Ns_per_m2) - q.shape[0], *[q.shape[1]] * (len(tup) - 1))
                )
                empty_q = Q_(empty_q, q.units)
                extended_q = np.concatenate([empty_q, q])
                new_q = extended_q * Ns_per_m2_shifted.reshape(tup)

                #                v.data = new_q.magnitude
                #                v.attrs["units"] = new_q.get_netcdf_unit()
                attrs = v.attrs.copy()
                attrs["units"] = new_q.get_netcdf_unit()
                new_v = xr.DataArray(data=new_q.magnitude, dims=v.dims, attrs=attrs)
                data_vars[name] = new_v

        data_vars["N_per_m2"] = xr.DataArray(
            data=Ns_per_m2.magnitude,
            dims=["timestep"],
            attrs={"units": Ns_per_m2.get_netcdf_unit()},
        )
        new_times = np.arange(len(data_vars["N_per_m2"]))

        #        new_coords = copy(ds_mean_tree.coords)
        new_coords = {k: v for k, v in ds_mean_tree.coords.items()}
        new_coords["timestep"] = new_times
        #        ds_mean_tree.assign_coords(new_coords)
        #        ds_mean_tree.assign({"timestep": new_times})

        new_ds_mean_tree = xr.Dataset(
            data_vars=data_vars,
            coords=new_coords,
        )
        #        return ds_mean_tree.assign(data_vars)
        return new_ds_mean_tree
