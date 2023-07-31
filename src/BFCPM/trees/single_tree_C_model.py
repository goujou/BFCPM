"""
This module contains a class for a C only model connected to a :class:`~.mean_tree.MeanTree`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, List

import numpy as np
import xarray as xr
from bgc_md2.models.BFCPM.__init__ import (GL, GPP, GR, GS, ML, MR, MR_C, MS,
                                           B_S_star, delta_W, f_CS, f_L, f_O,
                                           f_R, f_T, v_O, v_T)
from bgc_md2.models.BFCPM.source import srm as srm_BFCPM
from bgc_md2.notebook_helpers import write_to_logfile
from CompartmentalSystems.helpers_reservoir import \
    numerical_function_from_expression
from CompartmentalSystems.smooth_reservoir_model import \
    SmoothReservoirModel  # pylint: disable=unused-import

from .. import Q_, ureg, zeta_dw, zeta_gluc
from ..type_aliases import (CuttingFluxes, TreeExternalOutputFluxes,
                            TreeSoilInterface, WoodProductInterface)
from ..utils import assert_accuracy, reindent
from .single_tree_allocation import SingleTree, TreeShrinkError


class NegativityError(Exception):
    """Raised when either a stock or a flux is negative.

    In case the tree is in shrinking state, this is a reason to
    kill the tree by raising :exc:`TreeShrinkError`, otherwise
    just kill the simulation, because something is wrong.
    """

    def __init__(self, message):
        self.message = message
        super().__init__(self.message)


class SingleTreeCModel:
    """
    Class for a carbon only model connected with a
    :class:`~mean_tree.MeanTree`.

    All carbon units such as g_gluc and g_dw are converted to gC, so this
    class deals with gC only.
    An instance of this class can be used to initialize a
    :class`~.mean_tree.MeanTree` instance which in turn communicates with
    :class:`~.stand.Stand`.

    Args:
        tree: underlying :class:`~.single_tree_allocation.SingleTree` tree
            with its specific allometries
        stock_unit: the carbon unit to be used [gC]
        Delta_t: the time step used [yr]
        tol: absolute tolerance for consistency check with ``SingleTree`` model [-]

    Attributes:
        tree: underlying :class:`~.single_tree_allocation.SingleTree` tree
            with its specific allometries
        Delta_t: the time step used
        tol: absolute tolerance for consistency check with ``SingleTree`` model
        srm (SmoothReservoirModel): the underlying symbolic model
    """

    def __init__(
        self,
        tree: SingleTree,
        stock_unit: Q_[str] = Q_("1 gC"),
        Delta_t: Q_[str] = Q_("1 yr"),
        tol: float = 1e-08,
    ):
        self.tree = tree
        params = self.tree.params

        self._stock_unit = stock_unit
        self.Delta_t = Delta_t
        self._flux_unit = stock_unit / Delta_t

        srm = srm_BFCPM
        self.srm = srm

        # create start values
        sv = np.nan * np.ones(self.nr_pools)

        # leaves
        sv[1] = tree.B_L.to(stock_unit).magnitude  # B_L
        sv[2] = (tree.params["delta_L"] * tree.B_L).to(stock_unit).magnitude  # C_L

        # other
        sv[3] = tree.B_OS.to(stock_unit).magnitude  # B_OS
        sv[4] = tree.B_OH.to(stock_unit).magnitude  # B_OH

        # C_S
        if tree.C_S is None:
            sv[5] = 0.0  # C_S
        else:
            sv[5] = tree.C_S.to(stock_unit).magnitude  # C_S

        # trunk
        sv[7] = tree.B_TS.to(stock_unit).magnitude  # B_TS
        sv[6] = tree.B_TH.to(stock_unit).magnitude  # B_TH

        # roots
        sv[9] = tree.B_R.to(stock_unit).magnitude  # B_R
        sv[8] = (tree.params["delta_R"] * tree.B_R).to(stock_unit).magnitude  # C_R

        # create U_func
        self._U_func = self._create_U_func()
        self._Us: List[np.ndarray] = []

        # prepare state vector, parameters and functions

        # note that all rate parameters are in yr-1, so the time unit should
        # be yr as well
        # otherwise the C model becomes inconsistent and stops
        par_dict = {
            "zeta_gluc": zeta_gluc.magnitude,
            "zeta_dw": zeta_dw.magnitude,
            # leaves
            "delta_L": params["delta_L"].magnitude,
            "R_mL": params["R_mL"].magnitude,
            "S_L": params["S_L"].magnitude,
            "C_gL": params["C_gL"].magnitude,
            # roots
            "delta_R": params["delta_R"].magnitude,
            "R_mR": params["R_mR"].magnitude,
            "S_R": params["S_R"].magnitude,
            "C_gR": params["C_gR"].magnitude,
            # other (coarse roots + branches) and trunk
            "R_mS": params["R_mS"].magnitude,
            "S_O": params["S_O"].magnitude,
            "C_gW": params["C_gW"].magnitude,
            "C_gHW": params["C_gHW"].magnitude,
        }
        func_dict: Dict[str, Callable] = dict()
        free_variables = tuple(srm.state_vector) + (
            f_L,
            f_R,
            f_O,
            f_T,
            delta_W,
            v_O,
            v_T,
            B_S_star,
            MR_C,
            f_CS,
        )

        # create internal flux functions
        internal_flux_funcs = dict()
        for (pool_from, pool_to), expr in srm.internal_fluxes.items():
            internal_flux_funcs[
                (pool_from, pool_to)
            ] = numerical_function_from_expression(
                expr, free_variables, par_dict, func_dict
            )
        self.internal_flux_funcs = internal_flux_funcs
        self._Fs: List[np.ndarray] = []

        # create R_func
        self.R_func = numerical_function_from_expression(
            srm.external_outputs, free_variables, par_dict, func_dict
        )
        self._Rs: List[np.ndarray] = []

        # create maintencance respiration function
        B_L = srm.state_vector[1]
        B_R = srm.state_vector[9]
        R_M_func = numerical_function_from_expression(
            ML + MR + MS + MR_C, (B_L, B_R, B_S_star, MR_C), par_dict, func_dict
        )
        self.maintenance_respiration_flux_func = R_M_func
        self.ML_func = numerical_function_from_expression(
            ML, free_variables, par_dict, func_dict
        )
        self.MR_func = numerical_function_from_expression(
            MR, free_variables, par_dict, func_dict
        )
        self.MS_func = numerical_function_from_expression(
            MS, free_variables, par_dict, func_dict
        )

        self.GL_func = numerical_function_from_expression(
            GL, free_variables, par_dict, func_dict
        )
        self.GR_func = numerical_function_from_expression(
            GR, free_variables, par_dict, func_dict
        )
        self.GS_func = numerical_function_from_expression(
            GS, free_variables, par_dict, func_dict
        )

        # give the tree some initial transient C to work with
        # in its initial time step so that in can grow
        # C required for growth (minimum_E) and maintenance (M_R)
        _B_S_star = self.tree.B_S_star.to(self.stock_unit)
        args = (sv[1], sv[9], _B_S_star.magnitude, 0.0)
        M_R = Q_(self.maintenance_respiration_flux_func(*args), self.flux_unit)
        M_R_gC = (M_R * self.Delta_t).magnitude
        minimum_E = tree.minimum_E_to_resolve_C_budget()
        minimum_E_gC = (minimum_E * zeta_gluc * self.Delta_t).magnitude
        sv[0] = minimum_E_gC + M_R_gC

        # define the start vector based on previous computations
        self._start_vector = sv

        # create carbon stocks
        self._xs = [0 * sv]

        self.tol = tol

    def __str__(self) -> str:
        s = [f"tree type: {type(self)}"]
        s += [f"# C pools: {self.nr_pools}"]
        s += [reindent(str(self.tree), 4)]

        return "\n".join(s)

    @property
    def stock_unit(self) -> Q_[float]:
        """Unit of carbon stocks."""
        return self._stock_unit

    @property
    def flux_unit(self) -> Q_[float]:
        """Unit of carbon fluxes."""
        return self._flux_unit

    @property
    def nr_pools(self) -> int:
        """Number of carbon pools."""
        return self.srm.nr_pools

    @property
    def xs(self) -> List[np.ndarray]:
        """List of state vectors."""
        return self._xs

    @property
    def Us(self) -> List[np.ndarray]:
        """List of external input vectors."""
        return self._Us

    @property
    def Fs(self) -> List[np.ndarray]:
        """List of internal flux matrices."""
        return self._Fs

    @property
    def Rs(self) -> List[np.ndarray]:
        """List of external output vectors."""
        return self._Rs

    @property
    def pool_names(self) -> List[str]:
        """List of pool names."""
        return [sv.name for sv in self.srm.state_vector]

    @property
    def start_vector(self) -> np.ndarray:
        """Initial stocks."""
        return self._start_vector

    @property
    def species(self) -> str:
        """Tree species."""
        return self.tree.species

    @property
    def SLA(self) -> Q_[float]:
        """Specific leaf area [m2 kg_dw-1]."""
        return self.tree.SLA

    @property
    def B_L(self) -> Q_[float]:
        """Leave biomass [kg_dw]."""
        return self.tree.B_L

    @property
    def LA(self) -> Q_[float]:
        """Total leaf area [m2]."""
        return self.tree.LA

    def lad_normed(self, z: np.ndarray) -> Q_[np.ndarray]:
        """Normalized leaf area density over grid ``z``, integrates to 1, [m-1].

        Args:
            z: grid of tree height layer boundaries [m]

        Returns:
            normalized leaf area density of grid ``z``, [m-1]
        """
        return self.tree.lad_normed(z)

    @property
    def H(self) -> Q_[float]:
        """Tree height [m]."""
        return self.tree.H.to("m")

    # THIS CALLS FUNCTION TO GROW TREE. WHY THE WRAPPER IS NEEDED.
    # FIRST gC plant-1 yr-1 is converted
    # to glucose units and then immediately back to gC plant-1?
    # --> mostly to introduce the time step: gC plant-1 time-1
    # --> this should be done more automatically one level up
    def update_gC_no_unit(
        self, An: float, tree_soil_interface: TreeSoilInterface
    ) -> TreeExternalOutputFluxes:
        r"""Wrapper for ``self.update``.

        Function called by :meth:`~.stand.Stand.update_structure`,
        reports back external outfluxes dictionary.
        Converts gC to g_gluc/Delta_t and forwards to :meth:`~single_tree_C_model.SingleTreeCModel.update`.

        Args:
            An: accumulated :math:`A_{\mathrm{net}}` [gC]
            tree_soil_interface: dictionary

                - pool_from: {pool_to_1: proportion_1, ...}, proportions must sum to 1

        Returns:
            external output fluxes dictionary, stand can then distribute them to soil

            - (pool_from, pool_to): flux
        """
        An_gluc = Q_(An, "gC").to("g_gluc") * 1 / self.Delta_t
        return self.update(An_gluc, tree_soil_interface)

    @ureg.check(None, "[mass_glucose]/[time]", None)
    def update(
        self, An_gluc: Q_[float], tree_soil_interface: TreeSoilInterface
    ) -> TreeExternalOutputFluxes:
        r"""Update the tree.

        Args:
            An_gluc: accumulated :math:`A_{\mathrm{net}}` [g_gluc yr-1]
            tree_soil_interface: dictionary

                - pool_from: {pool_to_1: proportion_1, ...} proportions must sum to 1

        Returns:
            external output fluxes dictionary, stand can then distribute them to soil

            - (pool_from, pool_to): flux
        """
        nr_pools = self.nr_pools

        # convert to g_carbon/yr
        An = (An_gluc * zeta_gluc).to(self.flux_unit)

        # U, external inputs
        U = self._U_func(An.magnitude).reshape(nr_pools)
        self._Us.append(U)

        _B_S_star = self.tree.B_S_star.to(self.stock_unit)

        # in self.tree.update the correct growth rate is computed to
        # match the allometries, and the allocation coefficients to the
        # different tree organs are returned as well as some other parameters
        # needed to plug them into the symbolic model.
        #  maintenance respiration
        # is computed symbolically before, but to do so
        # B_S_star from self.tree is required (before the update)
        # growth costs are only implicityly computed by the symbolic model
        # and go into the external output fluxes

        # update the tree
        x = self.xs[-1]
        args = (x[1], x[9], _B_S_star.magnitude, 0.0)
        M_R = Q_(
            self.maintenance_respiration_flux_func(*args), self.flux_unit
        )  # gC yr-1

        # this E is the amount available for allocation to tree organs
        # it consists of C in the storage pool (x[0], E in the symbolic model)
        # minus the maintenance respiration that leaves from x[0]
        # newly incoming C (An=GPP) will at the end of the time step be moved
        # to the x[0] and is available only in the next time step
        # this way no C can be respired at age zero
        Delta_t = self.Delta_t
        # E represents A_net in the manuscript
        E = Q_(x[0], self.stock_unit) / Delta_t - M_R  # gC yr-1

        C_S = (
            self.tree.C_S if self.tree.C_S is not None else Q_(0.0, "g_gluc")
        )  # g_gluc
        free_vars = self.tree.update(E / zeta_gluc)
        (
            f_L_times_E,  # g_gluc yr-1
            f_R_times_E,  # g_gluc yr-1
            f_O_times_E,  # g_gluc yr-1
            f_T_times_E,  # g_gluc yr-1
            _v_O,  # yr-1
            _v_T,  # yr-1
            _delta_W,  # g_gluc g_dw-1
            f_CS_times_CS,  # g_gluc yr-1
        ) = free_vars

        # convert from glucose to C
        _f_L = zeta_gluc * f_L_times_E / E  # [-]: gC/g_gluc * g_gluc/yr / (gC/yr)
        _f_R = zeta_gluc * f_R_times_E / E
        _f_O = zeta_gluc * f_O_times_E / E
        _f_T = zeta_gluc * f_T_times_E / E
        if C_S != 0:
            #            print(360, _f_CS)  # [-]: gC/g_gluc * g_gluc/yr / (gC/yr)
            _f_CS = f_CS_times_CS / (C_S / Delta_t)  # [-]: g_gluc/yr / (g_gluc/yr)
        else:
            _f_CS = 0 * _f_L
        #            print(365, _f_CS)

        # for reporting purposes only
        self._f_L_times_E = _f_L * E  # pylint: disable=attribute-defined-outside-init
        self._f_R_times_E = _f_R * E  # pylint: disable=attribute-defined-outside-init
        self._f_O_times_E = _f_O * E  # pylint: disable=attribute-defined-outside-init
        self._f_T_times_E = _f_T * E  # pylint: disable=attribute-defined-outside-init
        self._f_CS_times_CS = (
            _f_CS * C_S
        )  # pylint: disable=attribute-defined-outside-init

        # in case of imprecise solution of the root algorithm, correct by
        # flexibly adapting maintenance respiration
        coefficient_sum = _f_L + _f_R + _f_O + _f_T

        #        R_M_correction = (1.0 - coefficient_sum) * E
        if coefficient_sum >= 1.0:
            R_M_correction = (1.0 - coefficient_sum) * E
        else:
            R_M_correction = 0 * E

        coefficient_sum_tol = 0.1
        #        if not np.abs(coefficient_sum - 1.0) < coefficient_sum_tol:
        if coefficient_sum > 1.0 + coefficient_sum_tol:
            print()
            print("--------- ERROR --------")
            print(coefficient_sum)
            print(_f_L, f_L_times_E)
            print(_f_R, f_R_times_E)
            print(_f_T, f_T_times_E)
            print(_f_O, f_O_times_E)
            print(_f_CS, f_CS_times_CS)
            print(R_M_correction)
            print([x[0] for x in self.xs])
            print(E, Q_((M_R * self.Delta_t).magnitude, self.flux_unit))
        #        assert np.abs(coefficient_sum - 1.0) < coefficient_sum_tol
        assert coefficient_sum <= 1.0 + coefficient_sum_tol

        self._coefficient_sum = (
            coefficient_sum  # pylint: disable=attribute-defined-outside-init
        )
        self._R_M_correction = (
            R_M_correction  # pylint: disable=attribute-defined-outside-init
        )

        # scaling from available C E=Anet to the transient pool E=x[0]
        _f_L_rate = _f_L * E / Q_(x[0], self.stock_unit)  # [yr-1]: gC/yr / gC
        _f_R_rate = _f_R * E / Q_(x[0], self.stock_unit)  # [yr-1]: gC/yr / gC
        _f_O_rate = _f_O * E / Q_(x[0], self.stock_unit)  # [yr-1]: gC/yr / gC
        _f_T_rate = _f_T * E / Q_(x[0], self.stock_unit)  # [yr-1]: gC/yr / gC
        # [y-1]: gC/g_gluc * (g_gluc/yr) / gC
        _f_CS_rate = _f_CS * zeta_gluc * (C_S / Delta_t) / Q_(x[5], self.stock_unit)

        # prepare function arguments for calling the symbolic model and
        # obtain all the fluxes from it (gC yr-1)
        f_tup = (
            _f_L_rate.magnitude,
            _f_R_rate.magnitude,
            _f_O_rate.magnitude,
            _f_T_rate.magnitude,
            _delta_W.magnitude,
            _v_O.magnitude,
            _v_T.magnitude,
            _B_S_star.magnitude,
            R_M_correction.magnitude,
            _f_CS_rate.magnitude,
        )
        args = tuple((float(y) for y in x)) + f_tup  # type: ignore

        # F, internal fluxes
        F = np.zeros((nr_pools, nr_pools))
        for (pool_from, pool_to), func in self.internal_flux_funcs.items():
            F[pool_to, pool_from] = func(*args)
        self._Fs.append(F)

        # growth respiration fluxes leaves, roots, sapwood
        # they are not really used here, but nice to have
        self._GL = self.GL_func(*args)  # pylint: disable=attribute-defined-outside-init
        self._GR = self.GR_func(*args)  # pylint: disable=attribute-defined-outside-init
        self._GS = self.GS_func(*args)  # pylint: disable=attribute-defined-outside-init

        # R, external outfluxes
        R = self.R_func(*args).reshape(nr_pools)
        self._Rs.append(R)

        # maintenance respiration fluxes leaves, roots, sapwood
        # they are not really used here, but nice to have
        self._ML = self.ML_func(*args)  # pylint: disable=attribute-defined-outside-init
        self._MR = self.MR_func(*args)  # pylint: disable=attribute-defined-outside-init
        self._MS = self.MS_func(*args)  # pylint: disable=attribute-defined-outside-init

        # update state vector, in U we find the allocated An,
        # which moves now to x[0]
        Delta_x_positive_part = U + F.sum(axis=1)
        Delta_x_negative_part = F.sum(axis=0) + R
        x = x - Delta_x_negative_part * self.Delta_t.magnitude  # does not change xs[-1]

        # check that not more mass leaves any pool than is in there
        # BEFORE new mass comes in: important for DMR construction
        try:
            assert np.all(x >= -1e-08)
        except AssertionError as error:
            for i in np.where(x < 0):
                print("Too high outfluxes from pool", i)
                print(
                    self.xs[-1][i], x[i], U[i], F.sum(axis=1)[i], F.sum(axis=0)[i], R[i]
                )
                print()
            raise error

        # add inputs to the pools
        x = x + Delta_x_positive_part * self.Delta_t.magnitude

        self.xs.append(x)
        # check that stocks and fluxes are nonnegative
        try:
            self._check_nonnegativity()
        except NegativityError as err:
            if self.tree.tree_status == "shrinking":
                raise TreeShrinkError
            else:
                raise err
        # check consistency with tree model
        self._check_consistency_with_tree()

        # compute the external output fluxes (leaving the tree) and allocate them
        # based on tree_soil_interface to the soil pools
        # the real moving of these fluxes in done by the stand depending
        # on N_per_m2, but based on this per plant information
        state_vector = self.srm.state_vector

        # output_fluxes[(pool_from, pool_to)] = gC plant-1 year-1
        output_fluxes: TreeExternalOutputFluxes = dict()

        for pool_nr in range(nr_pools):
            pool_from = state_vector[pool_nr].name
            total_flux = Q_(R[pool_nr], self.flux_unit)

            if pool_from in tree_soil_interface.keys():
                fraction_sum = 0.0
                d = tree_soil_interface[pool_from]
                for pool_to, fraction in d.items():
                    fraction_sum += fraction
                    key = (pool_from, pool_to)
                    if key not in output_fluxes.keys():
                        output_fluxes[key] = total_flux * fraction
                    else:
                        output_fluxes[key] += total_flux * fraction

                assert_accuracy(1.0, fraction_sum, 1e-08)
            else:
                key = (pool_from, None)
                output_fluxes[key] = total_flux

        # output fluxes is list of sub-fluxes in gC plant-1 yr-1?
        # --> fluxes leaving the plant toward the atmosphere or the soil
        # in gC plant-1 yr-1
        return output_fluxes

    def get_cutting_fluxes(
        self, wood_product_interface: WoodProductInterface, logfile_path: Path
    ) -> CuttingFluxes:
        """Distribute biomass to soil and wood products on cutting the tree."""
        state_vector = self.srm.state_vector
        tree = self.tree

        vf_func = wood_product_interface["_trunk_fate_func"]
        vf_dict = vf_func(
            tree.dbh.to("cm").magnitude, tree.H.to("m").magnitude, tree.species
        )
        log = [f"dbh={tree.dbh:2.2f}, H={tree.H:2.2f}, {tree.species}"]
        log += [f"{vf_dict}"]
        log_str = "\n".join(log)
        print(log_str)
        write_to_logfile(logfile_path, log_str)

        cutting_fluxes: CuttingFluxes = dict()
        for pool_nr in range(self.nr_pools):
            pool_from = state_vector[pool_nr].name
            total_flux = Q_(self.xs[-1][pool_nr], self.flux_unit)
            fraction_sum = 0.0

            if pool_from not in wood_product_interface.keys():
                raise ValueError(
                    f"No destination given for pool {pool_from} from cut tree."
                )

            if pool_from in ["C_S", "B_TH", "B_TS"]:

                if pool_from == "C_S":
                    d = wood_product_interface[pool_from]
                    fs = {"other": tree.B_OS / tree.B_S, "trunk": tree.B_TS / tree.B_S}

                    sub_pool_name = "other"
                    f = fs["other"]

                    for k, v in d[sub_pool_name].items():
                        pool_to, fraction = k, v
                        fraction = fraction * f.magnitude
                        fraction_sum += fraction

                        key = (pool_from, pool_to)
                        if key not in cutting_fluxes.keys():
                            cutting_fluxes[key] = total_flux * fraction
                        else:
                            cutting_fluxes[key] += total_flux * fraction

                    sub_pool_name = "trunk"
                    f = fs["trunk"]
                    for typ, pool_to in d[sub_pool_name].items():
                        fraction = vf_dict[typ] * f.magnitude
                        fraction_sum += fraction

                        key = (pool_from, pool_to)
                        if key not in cutting_fluxes.keys():
                            cutting_fluxes[key] = total_flux * fraction
                        else:
                            cutting_fluxes[key] += total_flux * fraction

                else:
                    for typ, pool_to in wood_product_interface[pool_from].items():
                        fraction = vf_dict[typ]
                        fraction_sum += fraction

                        key = (pool_from, pool_to)
                        if key not in cutting_fluxes.keys():
                            cutting_fluxes[key] = total_flux * fraction
                        else:
                            cutting_fluxes[key] += total_flux * fraction

            else:
                for k, v in wood_product_interface[pool_from].items():
                    pool_to, fraction = k, v
                    fraction_sum += fraction

                    key = (pool_from, pool_to)
                    if key not in cutting_fluxes.keys():
                        cutting_fluxes[key] = total_flux * fraction
                    else:
                        cutting_fluxes[key] += total_flux * fraction

            assert_accuracy(1.0, fraction_sum, 1e-08)

        return cutting_fluxes

    def simulate_being_removed(self):
        """Do nothing, keep the lists the correct length."""
        nr_pools = self.nr_pools
        self._xs.append(np.zeros(nr_pools))
        self._Us.append(np.zeros(nr_pools))
        self._Rs.append(np.zeros(nr_pools))
        self._Fs.append(np.zeros((nr_pools, nr_pools)))

    def simulate_waiting(self):
        """Wait until being planted, keep the lists the correrct length."""
        self.simulate_being_removed()

    def get_stocks_and_fluxes_dataset(
        self,
    ) -> xr.Dataset:  # pylint: disable=missing-function-docstring
        """Return dataset of stocks and fluxes after a simulation."""
        nr_pools = self.nr_pools

        timesteps = np.arange(len(self.xs))
        pool_names = self.pool_names
        coords = {
            "timestep": timesteps,
            "pool": pool_names,
            "pool_to": pool_names,
            "pool_from": pool_names,
        }

        data_vars = dict()

        # stocks
        data_vars["stocks"] = xr.DataArray(
            data=np.array(self.xs),
            dims=["timestep", "pool"],
            attrs={
                "units": Q_.get_netcdf_unit(self.stock_unit),
                "cell_methods": "time: instantaneous",
            },
        )

        # fluxes
        flux_attrs = {
            "units": Q_.get_netcdf_unit(self.flux_unit),
            #            "cell_methods": "time: mean"
            "cell_methods": "time: total",
        }

        # input fluxes
        Us = Q_(np.array(self.Us + [np.nan * np.ones(nr_pools)]), self.stock_unit)
        #        Us_mean = Us / self.Delta_t
        data_vars["input_fluxes"] = xr.DataArray(
            #            data=Us_mean.magnitude,
            data=Us.magnitude,
            dims=["timestep", "pool_to"],
            attrs=flux_attrs,
        )

        # output fluxes
        Rs = Q_(np.array(self.Rs + [np.nan * np.ones(nr_pools)]), self.stock_unit)
        #        Rs_mean = Rs / self.Delta_t
        data_vars["output_fluxes"] = xr.DataArray(
            #            data=Rs_mean.magnitude,
            data=Rs.magnitude,
            dims=["timestep", "pool_from"],
            attrs=flux_attrs,
        )

        # internal fluxes
        Fs = Q_(
            np.array(self.Fs + [np.nan * np.ones((nr_pools, nr_pools))]),
            self.stock_unit,
        )
        #        Fs_mean = Fs / self.Delta_t
        data_vars["internal_fluxes"] = xr.DataArray(
            #            data=Fs_mean.magnitude,
            data=Fs.magnitude,
            dims=["timestep", "pool_to", "pool_from"],
            attrs=flux_attrs,
        )

        # create dataset
        ds = xr.Dataset(
            data_vars=data_vars,  # type: ignore
            coords=coords,  # type: ignore
        )

        return ds

    ###########################################################################

    def _check_nonnegativity(self):
        """Check that stocks and fluxes are nonnegative."""

        def check_nonnegative_one_pool(vs: List[np.ndarray], s: str):
            """Check that `v[-1]` is nonnegative.

            Args:
                vs: stocks (xs), input fluxes (Us), or output fluxes (Rs)
                s: short explanatory string whether stock or flux

            Raises:
                AssertionError: if `v[-1]` has a negative entry
            """
            v = vs[-1]
            try:
                assert np.all(v >= 0)
            except AssertionError as er:
                neg_idx = np.where(v < 0)
                msg = [f"Negative {s} in pools {neg_idx}"]
                msg += [f"values: {v[neg_idx]}"]
                msg += [f"timestep: {len(vs)}"]
                msg += [f"history: {np.array(vs)[:, neg_idx]}"]
                msg_str = ", ".join(msg)

                raise NegativityError(msg_str) from er

        # check that no stock, input flux, and output flux is negative
        for vs, s in zip(
            [self.xs, self.Us, self.Rs], ["stock", "input flux", "output flux"]
        ):
            check_nonnegative_one_pool(vs, s)

        # check that no internal Flux is negative
        F = self.Fs[-1]
        try:
            assert np.all(F >= 0)
        except AssertionError as er:
            neg_flux_to, neg_flux_from = np.where(F < 0)
            msg = [f"Negative internal flux from: {neg_flux_from} to {neg_flux_to}"]
            msg += [f"values: {F[neg_flux_to, neg_flux_from]}"]
            msg += [f"timestep: {len(self.Fs)}"]
            msg += [f"history: {np.array(self.Fs)[:, neg_flux_to, neg_flux_from]}"]
            msg = ", ".join(msg)

            raise NegativityError(msg) from er

    def _check_consistency_with_tree(self):
        """Check whether carbon stocks are consistent with ``SingleTree``."""
        x = self.xs[-1]
        tol = self.tol

        stock_unit = self.stock_unit

        # global labile carbon pool
        # remains constant as long as glucose budget is zero
        # --> not anymore, it contains stored C from the last time step

        # leaves
        assert_accuracy(Q_(x[1], stock_unit), self.tree.B_L.to(stock_unit), tol)  # B_L
        assert_accuracy(
            Q_(x[2], stock_unit),  # C_L
            (self.tree.params["delta_L"] * self.tree.B_L).to(stock_unit),
            tol,
        )

        # other
        try:
            assert_accuracy(
                Q_(x[3], stock_unit), self.tree.B_OS.to(stock_unit), tol  # B_OS
            )
            assert_accuracy(
                Q_(x[4], stock_unit), self.tree.B_OH.to(stock_unit), tol  # B_OH
            )

            # C_S
            assert_accuracy(
                Q_(x[5], stock_unit), self.tree.C_S.to(stock_unit), tol  # C_S
            )
        except AssertionError as error:
            print(774)
            print(Q_(x[3], stock_unit).to("g_dw"))
            print(self.tree.B_OS)
            print(777)
            print(Q_(x[5], stock_unit).to("g_gluc"))
            print(self.tree.C_S)
            print([Q_(x[5], stock_unit).to("g_gluc") for x in self.xs])
            F = self.Fs[-1]
            print(Q_(F[3, 0], stock_unit).to("g_gluc"))
            print(Q_(F[3, 5], stock_unit).to("g_gluc"))
            print(Q_(F[4, 5], stock_unit).to("g_gluc"))
            print(Q_(F[5, 0], stock_unit).to("g_gluc"))
            R = self.Rs[-1]
            print(Q_(R[5], stock_unit).to("g_gluc"))
            print(self.tree._Delta_C_S)
            print()
            print([Q_(x[3], stock_unit).to("g_gluc") for x in self.xs])
            print([Q_(x[4], stock_unit).to("g_gluc") for x in self.xs])
            print([Q_(x[7], stock_unit).to("g_gluc") for x in self.xs])
            print([Q_(x[6], stock_unit).to("g_gluc") for x in self.xs])
            raise error

        # trunk
        assert_accuracy(
            Q_(x[7], stock_unit), self.tree.B_TS.to(stock_unit), tol  # B_TS
        )
        assert_accuracy(
            Q_(x[6], stock_unit), self.tree.B_TH.to(stock_unit), tol  # B_TH
        )

        # roots
        assert_accuracy(Q_(x[9], stock_unit), self.tree.B_R.to(stock_unit), tol)  # B_R
        assert_accuracy(
            Q_(x[8], stock_unit),  # C_R
            (self.tree.params["delta_R"] * self.tree.B_R).to(stock_unit),
            tol,
        )

    ###########################################################################

    def _create_U_func(self) -> Callable:
        return numerical_function_from_expression(
            self.srm.external_inputs, (GPP,), {}, {}
        )
