"""
This module contains the class
:class:`~single_tree_allocation.SingleTree`.

The class combines eco-physiological principles and tree allometries
to model C allocation to tree organs.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import brentq, root

from .. import Q_, ureg, zeta_dw, zeta_gluc
from ..type_aliases import SpeciesParams
from . import tree_utils
from .allometry import allometries_repola_and_lehtonen as allometries
from .single_tree_params import initialize_params, species_params
from .single_tree_vars import assign, var_infos

# from ..utils import cached_property


class TreeShrinkError(Exception):
    """Raised when the tree wants to shrink its radius at trunk base.

    This happends when the tree's :math:`C_S'` pool is too depleted
    such that the tree cannot survive.
    """

    def __init__(self, tree_name=None):
        super().__init__()
        self._tree_name = tree_name

    @property
    def tree_name(self):
        """Name of the tree that raised the problem."""
        return self._tree_name


class GlucoseBudgetError(TreeShrinkError):
    """Raised when the glucose budget could not be resolved.

    This needs to be resolved numerically in order to find the
    right ``Delta_r`` for radial growth given an amount of available
    glucose ``E``. It usually does not work when ``E`` is too small, which
    seems to happen for small spruces in the year after planting.
    By catching this exception, the stand can make the tree to enjoy
    another year of doing nothing and collect the captured carbon
    in ``x[0]`` of the
    :class:`~single_tree_C_model.SingleTreeCModel`.
    Next year, next try.

    Currently not being used.
    """


class SingleTree:
    """
    Single tree class comprising internal cycling and external allometries.

    An instance of this class can be used to initialize an instance of
    :class:`~single_tree_C_model.SingleTreeCModel`
    which can then be used to initialize an instance of
    :class:`~.mean_tree.MeanTree` which in turn
    can be added to a :class:`~.stand.Stand`.

    Args:
        species: tree species, element of ``["pine", "spruce", "birch"]``
        r: radius at trunk base [m]
        Delta_t: time step [yr]
        C_S: initial value of ``C_S pool``, required if tree is
            big enough to have heartwood, otherwise defaults to ``Q_("0 g_gluc")``
        B_TS: initial value of ``B_TS`` pool, optional
        B_TH: initial value of ``B_TH`` pool, required if tree is
            big enough to have heartwood, otherwise defaults to ``Q_("0 g_dw")``
        custom_species_params: tree species parameters (for all species)
        tree_age: age of the tree [yr]
        tol: absolute tolerance in check for self-consistency

    Attributes:
        species: element of ``["pine", "spruce", "birch"]``
        r (Q_[float]): radius at trunk base [m]
        Delta_t (Q_[float]): time step [yr]
        params (dict[str, Any]): tree parameters, mostly allometric
        tol (float): absolute tolerance in check for self-consistency [-]
    """

    def __init__(
        self,
        species: str,
        r: Q_[float],
        Delta_t: Q_[float] = Q_("1 yr"),
        C_S: Q_[float] = None,
        B_TS: Q_[float] = None,
        B_TH: Q_[float] = None,
        custom_species_params: SpeciesParams = None,
        tree_age=Q_(0, "yr"),
        tol: float = 1e-08,
    ):
        self.clear_cache()
        self.tree_status = "healthy"

        self.species = species
        if r == 0:
            raise ValueError("Cannot create a tree with radius zero.")
        self.r = assign("r", r)

        if custom_species_params is None:
            params = initialize_params(species_params[species])
            self.custom_species_params = species_params
        else:
            params = initialize_params(custom_species_params[species])
            self.custom_species_params = custom_species_params

        self.params = params

        self.Delta_t = Delta_t
        self.tree_age = tree_age

        dbh = self.dbh
        H = self.H_func(dbh)

        if B_TS is None:
            if self.V_TH != Q_(0, "m^3"):  # pylint: disable=comparison-with-callable
                raise NotImplementedError("Tree too big to be initialized.")
            self.B_TH = assign("B_TH", 0)
        else:
            self.B_TH = B_TH

        # compute initial B_TS and C_S

        # this is how it would be done following Ogle and Pacala
        # unfortunately, the results do not match allometric equations
        if B_TS is None:
            #            rho_W0 = self.compute_rho_W(
            #                dbh, H, Q_("0.0 m")
            #            )  # , Delta_V_TH=Q_(0, "m^3"))

            # make sure C_S_star >= 0, multiply with 0.5 to have at leas some C_S at the beginning
            rho_W0 = assign("rho_W", (1 - params["gamma_X"]) / params["gamma_W"] * 0.5)
            print(rho_W0)
            print(rho_W0.__repr__())
            print(self.V_TS)
            print(self.V_TS.__repr__())

            self.B_TS = assign("B_TS", rho_W0 * self.V_TS)
        else:
            self.B_TS = B_TS

        if self.B_TH == 0:
            self.SW = assign("SW", self.r)
        else:
            self.SW = self.SW_func(self.r, self.tree_age)

        if C_S is None:
            self.C_S = assign("C_S", self.C_S_star)
        else:
            self.C_S = C_S

        # living sapwood biomass, depends on B_TS and V_TS
        assert self.B_S_star >= 0

        self.tol = tol

        # store initial dbh for potential replanting
        self.initial_dbh: Q_ = dbh

    @classmethod
    def from_dbh(
        cls,
        species: str,
        dbh: Q_[float],
        Delta_t: Q_[float] = Q_("1 yr"),
        C_S: Q_[float] = None,
        B_TS: Q_[float] = None,
        B_TH: Q_[float] = None,
        custom_species_params: SpeciesParams = None,
        tree_age: Q_[float] = Q_(0, "yr"),
        tol: float = 1e-08,
    ):
        """Constructor from dbh instead of r.

        Args:
            species: element of ["pine", "spruce", "birch"]
            dbh: diameter at breast height [cm]
            Delta_t: time step [yr]
            C_S: initial value of ``C_S pool``, required if tree is
                big enough to have heartwood, otherwise defaults to ``Q_("0 g_gluc")``
            B_TS: initial value of ``B_TS`` pool, optional
            B_TH: initial value of ``B_TH`` pool, required if tree is
                big enough to have heartwood, otherwise defaults to ``Q_("0 g_dw")``
            custom_species_params: tree species parameters (for all species)
            tree_age: age of the tree [yr]
            tol: absolute tolerance in check for self-consistency [-]
        """
        if custom_species_params is None:
            custom_species_params = initialize_params(species_params[species])

        dbh_cm = dbh.to("cm").magnitude
        r_at_trunk_base = tree_utils.r_from_dbh(dbh_cm, species, custom_species_params)

        check_dbh_cm = tree_utils.dbh_from_r(
            r_at_trunk_base, species, custom_species_params
        )

        assert np.abs(dbh_cm - check_dbh_cm) < 1e-08

        return cls(
            species,
            r_at_trunk_base,
            Delta_t=Delta_t,
            C_S=C_S,
            B_TS=B_TS,
            B_TH=B_TH,
            custom_species_params=custom_species_params,
            tree_age=tree_age,
            tol=tol,
        )

    @classmethod
    def from_mleaf(
        cls,
        species: str,
        mleaf: Q_[float],
        Delta_t: Q_[float] = Q_("1 yr"),
        C_S: Q_[float] = None,
        B_TS: Q_[float] = None,
        B_TH: Q_[float] = None,
        custom_species_params: SpeciesParams = None,
        tree_age: Q_[float] = Q_(0, "yr"),
        tol: float = 1e-08,
    ) -> SingleTree:
        """
        Initialize a class instant based on leaf biomass.

        Args:
            species: tree species, element of ["pine", "spruce", "birch"]
            mleaf: leaf biomass [g_dw]
            Delta_t: time step [yr]
            C_S: initial value of ``C_S pool``, required if tree is
                big enough to have heartwood, otherwise defaults to ``Q_("0 g_gluc")``
            B_TS: initial value of ``B_TS`` pool, optional
            B_TH: initial value of ``B_TH`` pool, required if tree is
                big enough to have heartwood, otherwise defaults to ``Q_("0 g_dw")``
            custom_species_params: tree species parameters (for all species)
            tree_age: age of the tree [yr]
            tol: absolute tolerance in check for self-consistency [-]

        Returns:
            class instance
        """
        if custom_species_params is None:
            custom_species_params = initialize_params(species_params[species])

        r = Q_(
            tree_utils.r_from_mleaf(
                mleaf.to("kg_dw").magnitude,
                species,
                allometries,
                custom_species_params=custom_species_params,
            ),
            "m",
        )

        instance = cls(
            species,
            r,
            Delta_t=Delta_t,
            C_S=C_S,
            B_TS=B_TS,
            B_TH=B_TH,
            custom_species_params=custom_species_params,
            tree_age=tree_age,
            tol=tol,
        )
        return instance

    def clear_cache(self):
        """Clear the cache."""
        self._cache: dict[str, Any] = dict()

    def __str__(self):
        s = [f"Single tree type: {type(self)}"]
        s += [f"dbh: {self.dbh:~.4}"]
        s += [f"height: {self.H:~.4}"]
        return "\n".join(s)

    def lad_normed(self, z: np.ndarray) -> Q_[np.ndarray]:
        """Normalized leaf area density [m2 m-1] over grid ``z``, integrates to 1.

        Args:
            z: grid of tree height layer boundaries [m]

        Returns:
            normalized leaf area density of grid ``z``, [m-1]
        """
        lad_n = tree_utils.lad_normed_func(
            self.dbh.to("cm").magnitude, z, self.species  # pylint: disable=no-member
        )
        return Q_(lad_n, "1/m")

    ###########################################################################

    def solve_glucose_budget(
        self,
        dr: float,
        E: Q_[float],
        tree_status: str = "healthy",
        tree_init: bool = False,
    ) -> float:
        """
        Optimization function based on the radius change ``dr``.

        The goal is to choose ``dr`` such that ``Anet_gluc`` is used up
        during the tree's structural update, based on its allometries.

        Args:
            dr: potential change in tree radius ``r`` at trunk base [m]
            E: parameter of available glucose to distribute [g_gluc yr-1]
            tree_status: if "static" or "shrinking", no trunk growth
            tree_init: if True, we are still in the planting process

        Returns:
            res: relative discrepancy in carbon budget [-]
        """
        dr = float(getattr(dr, "_value", dr))  # it might come in as a one-element array

        if (not tree_init) and (dr < 0):
            return -dr

        Delta_r = assign("Delta_r", dr)
        next_r = self.r + Delta_r
        try:
            tup = self._update_with_next_r(next_r, tree_status)
        except ValueError as e:
            # don't stop trying to solve the glucose budget
            # just the currently tried r is technically too small
            # to compute the dbh, try another one, dont give up completely
            return -1 / Delta_r.magnitude if Delta_r.magnitude != 0 else np.inf

        f_L_times_E, f_R_times_E, f_T_times_E, f_O_times_E, _, _, _, _, _ = tup
        res = E - f_L_times_E - f_R_times_E - f_T_times_E - f_O_times_E

        #        res /= E # seems way more robust # or not?
        #        print(367, dr, E, res)
        #        print(f_L_times_E, f_R_times_E, f_T_times_E, f_O_times_E)
        return res.magnitude

    def minimum_E_to_resolve_C_budget(self) -> Q_[float]:
        """Determine the minimum amount of glucose to resolve budget.

        During initialization of the associated
        :class:`~.single_tree_C_model.SingleTreeCModel`
        a minimum amount of ``E`` is necessary to resolve the glucose budget
        in the first time step when there are yet no inputs by photosynthesis.
        So this method is called to determine how much ``E`` (glucose) is needed
        such that the tree can resolve its C budget for its first radial growth.

        Returns:
            ``E[0]``: initially needed amount for first radial growth [g_gluc yr-1]
        """
        # clear cache before computations
        self.clear_cache()

        def g(E: float):
            if E == 0:
                return -1000

            _Delta_r = getattr(self, "_Delta_r", Q_(1e-02, "m"))
            try:
                root_results = root(
                    self.solve_glucose_budget,
                    x0=_Delta_r.magnitude,
                    args=(Q_(E, "g_gluc/yr"), "healthy", True),
                )
                res = root_results.x
                success = root_results.success
                fun = root_results.fun

            #                res, root_results = brentq(
            #                    self.solve_glucose_budget,
            #                    -0.05,
            #                    0.05,
            #                    args=(E, "healthy", True),
            ##                    xtol=1e-05,
            ##                    rtol=1e-05,
            #                    full_output=True,
            #                )
            #                success = root_results.converged
            #                print(res, E)
            #                print(root_results)
            except ValueError as e:
                return -1000

            if (not success) and (np.abs(fun) > 1e-03):
                return -2000

            Delta_r = assign("Delta_r", float(res))
            #            print("E", E, "Delta_r", Delta_r, Delta_r/self.r)
            # The 0.01 here are super arbitrary but seem to work
            # at the moment. With a previous spinup version 0.02 worked
            # and with the current spinup 0.02 does not work anymore.
            if Delta_r / self.r < 0.01:  # or (Delta_r < Q_("0.01 cm")):
                return np.minimum(Delta_r.magnitude, -Delta_r / self.r)

            return np.abs(fun)

        E1 = 10  # g_gluc/yr
        while g(E1) < 0:
            E1 *= 2

        E0 = 0
        res, root_results = brentq(g, E0, E1, xtol=1e-05, rtol=1e-05, full_output=True)
        if not root_results.converged:
            raise GlucoseBudgetError

        return Q_(res, "g_gluc/yr")

    @ureg.check(None, "[length]", None)
    def _update_with_next_r(
        self, next_r: Q_[float], tree_status: str = "healthy"
    ) -> tuple:
        """Internal method used to find the right next ``r``.

        The return values will be used in the nonlinear optimization
        to find the right trunk growth ``Delta_r`` such that the
        photosynthetically captured glucose is optimally used.

        Args:
            next_r: potential radius at trunk base of next year's tree[m]
            tree_status: if "static" or "shrinking", no trunk growth

        Returns: tuple containing

            - f_L_times_E: glucose allocated to leaves [g_gluc yr-1]
            - f_R_times_E: glucose allocated to fine roots [g_gluc yr-1]
            - f_T_times_E: glucose allocated to trunk [g_gluc yr-1]
            - f_O_times_E: glucose allocated to other [g_gluc yr-1]
            - v_T: rate of trunk heartwood production from sapwood [yr-1]
            - v_O: rate of other heartwood production from sapwood [yr-1]
            - Delta_B_TH: growth in trunk heartwood biomass [g_dw]
            - Delta_B_TS: growth in trunk sapwood biomass [g_dw]
            - rho_W: density of newly produced sapwood [g_dw m-3]
        """
        #        print("_update_with_next_r")
        #        print(next_r, tree_status, self.tree_age)

        species = self.species
        next_dbh = Q_(
            tree_utils.dbh_from_r(
                next_r.to("m").magnitude, species, self.custom_species_params
            ),
            "cm",
        )
        if next_dbh < 0:
            raise ValueError("next_dbh < 0")

        next_H = self.H_func(next_dbh)

        next_B_L = self.B_L_func(next_dbh, next_H)
        f_L_times_E = self.compute_f_L_times_E(next_B_L)

        next_B_R = self.B_R_func(next_dbh, next_H)
        f_R_times_E = self.compute_f_R_times_E(next_B_R)

        if tree_status == "healthy":
            next_SW = self.SW_func(next_r, self.tree_age + self.Delta_t)
        else:
            next_SW = self.SW
        #        print(482, "next_SW", next_SW)

        next_V_TH = self.V_TH_func(next_dbh, next_H, next_SW)
        v_T = self.compute_v_T(next_V_TH)
        if tree_status in ["static", "shrinking"]:
            # no sapwood to heartwood conversion
            v_T *= 0

        next_V_T = self.V_T_func(next_dbh, next_H)

        #            Delta_V_TH = next_V_TH - self.V_TH
        Delta_r = next_r - self.r
        rho_W_healthy = self.compute_rho_W(next_dbh, next_H, Delta_r)  # , Delta_V_TH)
        if (tree_status == "healthy") and (next_dbh != self.dbh):
            rho_W = rho_W_healthy
        else:
            # if the density is too low, then sum(f_X) might be > 1 even though
            # in healthy state the tree wants to shrink, then we would
            # have a negative flux f_CS, which leads to problems
            #            rho_W = np.maximum(self._rho_W, rho_W_healthy)
            rho_W = getattr(self, "_rho_W", rho_W_healthy)
        #            print(
        #                480,
        #                "rho_W",
        #                tree_status,
        #                rho_W,
        #                rho_W_healthy,
        #                getattr(self, "_rho_W", "no _rho_W"),
        #            )

        delta_W = self.compute_delta_W(rho_W)
        f_T_times_E = self.compute_f_T_times_E(v_T, next_V_T, rho_W, delta_W)
        if tree_status in ["static", "shrinking"]:
            f_T_times_E *= 0

        delta_S = self.delta_S
        Delta_B_TH = (
            (1 + delta_S / self.params["C_gHW"]) * v_T * self.B_TS * self.Delta_t
        )
        next_B_TH = self.B_TH + Delta_B_TH

        Delta_B_TS = (
            f_T_times_E / (self.params["C_gW"] + delta_W) - v_T * self.B_TS
        ) * self.Delta_t
        next_B_TS = self.B_TS + Delta_B_TS

        next_B_OH = self.B_OH_func(next_dbh, next_H, next_B_TH)
        v_O = self.compute_v_O(next_B_OH)

        next_B_OS = self.B_OS_func(next_dbh, next_H, next_B_TS)
        f_O_times_E = self.compute_f_O_times_E(v_O, next_B_OS, delta_W)

        return (
            f_L_times_E,
            f_R_times_E,
            f_T_times_E,
            f_O_times_E,
            v_T,
            v_O,
            Delta_B_TH,
            Delta_B_TS,
            rho_W,
        )

    @ureg.check(None, "[mass_glucose]/[time]")
    def update(self, E: Q_[float]) -> tuple:
        """
        Update the tree according to the acccumulated incoming ``E=A_net``.

        Args:
            E: incoming accumulated net radiation as glucose [g_gluc/yr]

        Returns:
            Tuple of fluxes and rates.

            - f_L_times_E (Q_[float]): glucose allocated to leaves [g_gluc yr-1]
            - f_R_times_E (Q_[float]): glucose allocated to fine roots [g_gluc yr-1]
            - f_O_times_E (Q_[float]): glucose allocated to coarse roots and branches [g_gluc yr-1]
            - f_T_times_E Q_[float]: glucose allocated to trunk [g_gluc]
            - v_O (Q_[float]): sapwood to heartwood conversion rate
                of coarse roots and branches [yr-1]
            - v_T (Q_[float]): sapwood to heartwood conversion rate
                of the trunk [yr-1]
            - delta_W Q_[float]): maximum labile C storage capacity of
                newly produces sapwood [g_gluc/g_dw]
            - f_CS_times_CS (Q_[float]): in "static" and "shrinking" state amount of glucose used
                from storage [g_gluc yr-1]
        """
        species = self.species
        params = self.params

        # clear cache before computations
        self.clear_cache()

        tree_status = self.tree_status

        # solve for Delta_r to equalize glucose budget

        #        _, root_res = brentq(self.solve_glucose_budget, 0, 1, full_output=True)
        #        print(root_res)

        _Delta_r = getattr(self, "_Delta_r", Q_(0, "m"))
        if _Delta_r == Q_(0, "m"):
            _Delta_r = Q_(0.05, "m")

        root_results = root(
            self.solve_glucose_budget,
            x0=_Delta_r.magnitude,
            # healthy is important here, we want the theoretically optimal result
            args=(E, "healthy", False),
            # xtol=1e-05,
            #            args=(E, tree_status)
            #            method="lm" # don't use it, it's search ends with crazy results
        )
        res = root_results.x
        success = root_results.success
        Delta_r = assign("Delta_r", float(res))
        print(601, Delta_r, success)

        coefficient_sum_tol = 0.1

        if success:
            next_r_healthy = self.r + Delta_r
            tup = self._update_with_next_r(next_r_healthy, tree_status="healthy")
            (
                f_L_times_E,
                f_R_times_E,
                f_T_times_E,
                f_O_times_E,
                v_T,
                v_O,
                Delta_B_TH,
                Delta_B_TS,
                _,  # rho_W
            ) = tup
            print(620, "healthy", tup)

            fraction_E_required_for_healthy = (
                f_L_times_E + f_R_times_E + f_T_times_E + f_O_times_E
            ) / E

            if np.abs(fraction_E_required_for_healthy - 1.0) > coefficient_sum_tol:
                success = False

        # just in case, give it a second shot
        if not success:
            print(603, "'root' did not converge")
            print(root_results)
            # maybe remove tolerances?
            # Why the hell does it say it converged
            # when it is actually far away from a root?
            res, root_results = brentq(
                self.solve_glucose_budget,
                -0.05,
                0.05,
                args=(E, "healthy", False),
                xtol=1e-05,
                rtol=1e-05,
                full_output=True,
            )
            success = root_results.converged
            print(success)
            print(res, root_results)

        Delta_r = assign("Delta_r", float(res))

        next_r_healthy = self.r + Delta_r
        tup = self._update_with_next_r(next_r_healthy, tree_status="healthy")
        (
            f_L_times_E,
            f_R_times_E,
            f_T_times_E,
            f_O_times_E,
            v_T,
            v_O,
            Delta_B_TH,
            Delta_B_TS,
            rho_W,  # rho_W
        ) = tup
        #        print(661, "healthy", tup)
        if rho_W < 0 * rho_W:
            print(663, "negative wood density", rho_W)
            success = False

        fraction_E_required_for_healthy = (
            f_L_times_E + f_R_times_E + f_T_times_E + f_O_times_E
        ) / E

        if np.abs(fraction_E_required_for_healthy - 1.0) > coefficient_sum_tol:
            print("No convergence!")
            success = False

        print(674, "healthy", E, fraction_E_required_for_healthy, success)
        #        print(tree_status, tup)

        #        from scipy.optimize import minimize, Bounds
        #        root_res = minimize(
        #            self.solve_glucose_budget,
        #            x0=np.array([_Delta_r.magnitude]),
        #            bounds=Bounds(np.array([0]), np.array([np.inf]), keep_feasible=True),
        #            method="L-BFGS-B",# SLSQP
        ##Powell
        ##CG
        ##Newton-CG
        ##TNC
        ##COBYLA
        ##SLSQP
        ##trust-constr
        ##dogleg
        ##trust-ncg
        ##trust-exact
        ##trust-krylov
        #        )
        #        print(root_res)

        #        if (not root_res.success) and (np.abs(root_res.fun) > 1e-02) :
        ##        if (not root_res.success) and (np.abs(root_res.fun) > 0.1) :
        ##        if not root_res.converged:
        #            print(root_res)
        #            raise ValueError("Glucose budget could not be resolved.")
        ##            raise GlucoseBudgetError()

        #        Delta_r = assign("Delta_r", float(root_res.x))

        #        if (Delta_r <= 0) or (fraction_E_required_for_healthy > 1.1):
        if (Delta_r <= 0) or (not success):
            dbh = getattr(self, "dbh", None)
            msg = f"Tree wants to shrink! H={self.H}, dbh={dbh}"
            print(msg, Delta_r, fraction_E_required_for_healthy)

            print("710, healthy", tup)
            tup = self._update_with_next_r(self.r, tree_status="static")
            (
                f_L_times_E,
                f_R_times_E,
                f_T_times_E,
                f_O_times_E,
                v_T,
                v_O,
                Delta_B_TH,
                Delta_B_TS,
                _,  # rho_W
            ) = tup
            print(723, tree_status, tup)

            if f_T_times_E != 0:
                raise ValueError("Trunk should not get anything.")

            if E - f_L_times_E - f_R_times_E - f_O_times_E >= 0:
                # for instance, tree is in "static" from last year and now cannot
                # find a Delta_r for growth, root search returns Delta_r = 0,
                # but fraction_E_required_for_healthy < 1
                # there is some E still available! What to do with it?
                #                raise ValueError("Tree is not really trying to shrink??")
                print("Some C is kept in E, it cannot be used now for growth.")

            tree_status = "static"
            Delta_r = 0 * Delta_r

            if E < f_L_times_E + f_R_times_E:
                print(
                    "Not enough E to support leaves and roots, losing leaves and roots."
                )
                tree_status = "shrinking"
        else:
            tree_status = "healthy"

        next_r = self.r + Delta_r
        next_dbh = Q_(
            tree_utils.dbh_from_r(
                next_r.to("m").magnitude, species, self.custom_species_params
            ),
            "cm",
        )
        # double check the found next_dbh
        check_r = Q_(
            tree_utils.r_from_dbh(
                next_dbh.magnitude, species, self.custom_species_params
            ),
            "m",
        )
        assert np.abs(next_r - check_r) < Q_(1e-08, "m")

        # there was an error in the H_TH formula in the SD
        # of Ogle and Pacala, can in special situations be caught by:
        next_H = self.H_func(next_dbh)

        if tree_status == "healthy":
            next_SW = self.SW_func(next_r, self.tree_age)
        else:
            next_SW = self.SW
        next_H_TH = self.H_TH_func(next_r, next_H, next_SW)
        next_V_TH = self.V_TH_func(next_dbh, next_H, next_SW)

        assert self.H_TH <= next_H_TH + Q_(1e-05, "m")
        if self.V_TH > next_V_TH + Q_(1e-05, "m^3"):
            print(709, self.dbh, next_dbh, self.H, next_H)
            next_V_T = self.V_T_func(next_dbh, next_H)
            print(self.V_T, next_V_T, self.V_TH, next_V_TH)
        assert self.V_TH <= next_V_TH + Q_(1e-05, "m^3")

        tup = self._update_with_next_r(next_r, tree_status=tree_status)
        (
            f_L_times_E,
            f_R_times_E,
            f_T_times_E,
            f_O_times_E,
            v_T,
            v_O,
            Delta_B_TH,
            Delta_B_TS,
            rho_W,
        ) = tup
        #        if rho_W < 0 * rho_W:
        #            print(790, "negative wood density", rho_W)
        #            tree_status = "static"
        #            Delta_r = 0 * Delta_r
        #            next_r = self.r + Delta_R
        #
        #            tup = self._update_with_next_r(next_r, tree_status=tree_status)
        #            (
        #                f_L_times_E,
        #                f_R_times_E,
        #                f_T_times_E,
        #                f_O_times_E,
        #                v_T,
        #                v_O,
        #                Delta_B_TH,
        #                Delta_B_TS,
        #                rho_W,
        #            ) = tup

        print(737, tree_status, tup)

        fraction_E_required_for_status = (
            f_L_times_E + f_R_times_E + f_T_times_E + f_O_times_E
        ) / E
        print(742, tree_status, fraction_E_required_for_status)
        print(self.C_S, self.C_S_star, self.C_S / self.C_S_star)
        print(self.delta_S, 0.02 * zeta_dw / zeta_gluc)

        B_L_old = self.B_L
        B_R_old = self.B_R
        B_OS_old = self.B_OS
        B_OH_old = self.B_OH
        B_TS_old = self.B_TS
        B_TH_old = self.B_TH
        delta_S = self.delta_S

        delta_W = self.compute_delta_W(rho_W)
        C_S_star = self.C_S_star

        # for recording in simulation
        self._rho_W: Q_[float] = rho_W  # pylint: disable=attribute-defined-outside-init
        self._delta_W = delta_W  # pylint: disable=attribute-defined-outside-init

        print("Tree status:", tree_status, self.tree_status)
        if tree_status == "healthy":
            f_CS_times_CS = 0 * f_O_times_E

        if tree_status == "static":
            # glucose flux from E to B_OS and C_S and growth respiration
            #            if (not root_res.success) and (fraction_E_required_for_healthy) < 1.0:
            if (not success) and (fraction_E_required_for_status) < 1.0:
                print(774, "cannot use all E")
                f_O_star_times_E = f_O_times_E
                f_CS_times_CS = 0 * f_O_times_E
            else:
                f_O_star_times_E = E - f_L_times_E - f_R_times_E
                f_CS_times_CS = (params["S_O"] + v_O) * B_OS_old * params[
                    "C_gW"
                ] - f_O_star_times_E * params["C_gW"] / (params["C_gW"] + delta_W)
                print(782, f_CS_times_CS)

            if f_CS_times_CS < 0 * f_CS_times_CS:
                print("Flux from CS is negative, we set it to zero.")
                print("WE SHOULD NEVER BE HERE")

                #                f_O_star_times_E = f_O_star_times_E + f_CS_times_CS

                # make sure, Delta_B_OS == 0 by removing the part that would
                # go to C_S, adapt the different costs,
                # note that f_CS_times_CS is negative
                f_O_star_times_E += (
                    f_CS_times_CS / params["C_gW"] * (params["C_gW"] + delta_W)
                )
                f_CS_times_CS *= 0
                print(797, f_CS_times_CS)

            if f_CS_times_CS * self.Delta_t > self.C_S:
                tree_status = "shrinking"
                print("Changing from 'static' to 'shrinking'.")
            else:
                # update actual fluxes from E to other
                f_O_times_E = f_O_star_times_E

                Delta_B_OS_ = (
                    f_O_times_E / (params["C_gW"] + delta_W)
                    + f_CS_times_CS / params["C_gW"]
                    - (params["S_O"] + v_O) * B_OS_old
                ) * self.Delta_t
                print(818, self.r, self.dbh, self.H, self.tree_age, self.SW, self.C_S)
                print(tree_status)
                print(f_O_times_E, f_CS_times_CS)
                print(
                    self.B_OS,
                    B_OS_old,
                    self.B_OS - B_OS_old,
                    Delta_B_OS_,
                    f_CS_times_CS,
                )
                print(self.B_OH, B_OH_old, self.B_OH - B_OH_old)
                print(self.B_TS, B_TS_old, self.B_TS - B_TS_old)
                print(self.B_TH, B_TH_old, self.B_TH - B_TH_old)

        if tree_status == "shrinking":
            # scale down allocation equally
            x, y, z = f_L_times_E, f_R_times_E, f_O_times_E
            f_L_times_E = x / (x + y + z) * E
            f_R_times_E = y / (x + y + z) * E
            f_O_times_E = z / (x + y + z) * E

            f_CS_times_CS = (params["S_O"] + v_O) * B_OS_old * params[
                "C_gW"
            ] - f_O_times_E * params["C_gW"] / (params["C_gW"] + delta_W)
            print(816, f_CS_times_CS)
            if f_CS_times_CS < 0 * f_CS_times_CS:
                print("Flux from CS is negative, we set it to zero.")
                f_CS_times_CS *= 0

            # adapt leave and root changes, store the values, they will be
            # needed in the properties as long as the tree is shrinking
            Delta_B_L = (
                f_L_times_E / (params["C_gL"] + params["delta_L"])
                - params["S_L"] * B_L_old
            ) * self.Delta_t
            self._B_L = B_L_old + Delta_B_L

            Delta_B_R = (
                f_R_times_E / (params["C_gR"] + params["delta_R"])
                - params["S_R"] * B_R_old
            ) * self.Delta_t
            self._B_R = B_R_old + Delta_B_R

        #            if f_CS_times_CS * self.Delta_t > self.C_S:
        #                msg = (
        #                    "While shrinking, tree has not enough reserves to "
        #                    "sustain its coarse roots and branches. Emergency!"
        #                    ""
        #                )
        #                print(msg)
        #                raise TreeShrinkError()

        self.tree_status = tree_status

        # update C_S, B_TH, B_TS
        # Eq. (1D)
        self.update_C_S(E, delta_W, f_T_times_E, f_O_times_E, v_T, v_O, f_CS_times_CS)

        # check if C_S is depleted in unhealthy state
        #        if (self.tree_status in ["static", "shrinking"]) and (
        #            self.delta_S < 0.02 * zeta_dw / zeta_gluc
        #        ):
        #        if (self.tree_status in ["static", "shrinking"]) and (
        #            self.C_S < 0.5 * self.C_S_star
        #        ):
        #            self.delta_S < 0.5 * delta_W
        #        ):
        if (self.tree_status in ["static", "shrinking"]) and (self.C_S < 0 * self.C_S):
            msg = f"delta_S = {self.delta_S} < {0.02*zeta_dw/zeta_gluc}, emergency"
            print(msg)
            print(self.C_S, self.C_S_star, self.C_S / self.C_S_star)
            print("951, old barrier", self.delta_S, 0.02 * zeta_dw / zeta_gluc)
            print("new_barrier", delta_W, self.delta_S, 0.5 * delta_W)
            raise TreeShrinkError()

        self.B_TH = self.B_TH + Delta_B_TH
        self.B_TS = self.B_TS + Delta_B_TS

        # update remaining tree status
        self.r = next_r

        # increase the tree age
        self.tree_age += self.Delta_t

        # self._Delta_r is stored as a starting point for next year's
        # optimization
        self._Delta_r = Delta_r  # pylint: disable=attribute-defined-outside-init

        # clear cache to avoid using old tree data for future computations
        self.clear_cache()
        if self.tree_status == "healthy":
            self.SW = self.SW_func(self.r, self.tree_age)

        # check internal consistency
        tol = self.tol

        # Eq. (1A)
        Delta_B_L = (
            f_L_times_E / (params["C_gL"] + params["delta_L"]) - params["S_L"] * B_L_old
        ) * self.Delta_t
        assert np.abs(self.B_L - B_L_old - Delta_B_L) < Q_(tol, "g_dw")

        # Eq. (1B)
        Delta_B_R = (
            f_R_times_E / (params["C_gR"] + params["delta_R"]) - params["S_R"] * B_R_old
        ) * self.Delta_t
        assert np.abs(self.B_R - B_R_old - Delta_B_R) < Q_(tol, "g_dw")

        # Eq. (1E)
        Delta_B_OS = (
            f_O_times_E / (params["C_gW"] + delta_W)
            + f_CS_times_CS / params["C_gW"]
            - (params["S_O"] + v_O) * B_OS_old
        ) * self.Delta_t
        #        print(917, self.r, self.dbh, self.H, self.tree_age, self.SW, self.C_S)
        if not (np.abs(self.B_OS - B_OS_old - Delta_B_OS) < Q_(tol, "g_dw")):
            print(tree_status)
            print(f_O_times_E, f_CS_times_CS)
            print(self.B_OS, B_OS_old, self.B_OS - B_OS_old, Delta_B_OS, f_CS_times_CS)
            print(self.B_OH, B_OH_old, self.B_OH - B_OH_old)
            print(self.B_TS, B_TS_old, self.B_TS - B_TS_old)
            print(self.B_TH, B_TH_old, self.B_TH - B_TH_old)
        assert np.abs(self.B_OS - B_OS_old - Delta_B_OS) < Q_(tol, "g_dw")

        # Eq. (1F)
        Delta_B_OH = (
            (1 + delta_S / params["C_gHW"]) * v_O * B_OS_old - params["S_O"] * B_OH_old
        ) * self.Delta_t
        assert np.abs(self.B_OH - B_OH_old - Delta_B_OH) < Q_(tol, "g_dw")

        assert np.abs(self.B_TH - B_TH_old - Delta_B_TH) < Q_(tol, "g_dw")
        assert np.abs(self.B_TS - B_TS_old - Delta_B_TS) < Q_(tol, "g_dw")

        return (
            f_L_times_E,
            f_R_times_E,
            f_O_times_E,
            f_T_times_E,
            v_O,
            v_T,
            delta_W,
            f_CS_times_CS,
        )

    @property
    def dbh(self) -> Q_:
        """Diameter at breast height [cm]."""
        return assign(
            "dbh",
            tree_utils.dbh_from_r(
                self.r.to("m").magnitude, self.species, self.custom_species_params
            ),
        )

    @property
    def H(self) -> Q_[float]:
        """Tree height [m]."""
        return assign("H", self.H_func(self.dbh))

    @property
    def V_T(self) -> Q_[float]:
        """Trunk volume as computed by ACGCA [m3]."""
        return assign("V_T", self.V_T_func(self.dbh, self.H))

    @property
    def V_TS(self) -> Q_[float]:
        """Volume of trunk sapwood [m3]."""
        if not hasattr(self, "B_TH") or self.B_TH == 0:
            return assign("V_TH", self.V_T.magnitude)

        return assign("V_TS", self.V_TS_func(self.dbh, self.H, self.SW))

    @property
    def V_TH(self) -> Q_[float]:
        """Volume of trunk heartwood section [m3]."""
        if not hasattr(self, "B_TH") or self.B_TH == 0:
            return assign(
                "V_TH",
                0.0,
            )

        return assign("V_TH", self.V_TH_func(self.dbh, self.H, self.SW))

    @property
    def H_TH(self) -> Q_[float]:
        """Height of trunk heartwood section [m]."""
        return assign("H_TH", self.H_TH_func(self.r, self.H, self.SW))

    @property
    def LA(self) -> Q_[float]:
        """Total leaf area [m2]."""
        #        return assign("LA", self.LA_func(self.dbh, self.H))
        return assign("LA", self.LA_func(self.B_L))

    @property
    def SLA(self) -> Q_[float]:
        """Specific leaf area [m2 kg_dw-1]."""
        return self.SLA_func()

    @property
    def B_L(self) -> Q_[float]:
        """Leaf biomass [g_dw]."""
        if self.tree_status != "shrinking":
            return assign("B_L", self.B_L_func(self.dbh, self.H))
        else:
            return assign("B_L", self._B_L)

    @property
    def B_R(self) -> Q_[float]:
        """Total fine root area [m2]."""
        if self.tree_status != "shrinking":
            return assign("B_R", self.B_R_func(self.dbh, self.H))
        else:
            return assign("B_R", self._B_R)

    @property
    def B_OS(self) -> Q_[float]:
        """Coarse roots and branches (other) sapwood biomass [g_dw]."""
        return self.B_OS_func(self.dbh, self.H, self.B_TS)

    @property
    def B_S(self) -> Q_[float]:
        """Sapwood biomass (coarse roots and branches + trunk) [g_dw]."""
        return assign("B_S", self.B_S_func(self.dbh, self.H, self.B_TS))

    @property
    def B_T(self) -> Q_[float]:
        """Trunk biomass (sapwood + heartwood) [g_dw]."""
        if hasattr(self, "B_TS"):
            return assign(
                "B_T", self.B_T_func(self.B_TS, self.B_TH, self.C_S, self.B_OS)
            )
        else:
            return assign("B_T", 0)

    @property
    def B_OH(self) -> Q_[float]:
        """Coarse roots and branches heartwood biomass [g_dw]."""
        return assign("B_OH", self.B_OH_func(self.dbh, self.H, self.B_TH))

    # Is computed before it enters this tree here, not part of E
    # Used here only for output purposes and does not take into account
    # corrections that are necessary to close the GlucodeBudget
    @property
    def R_M(self) -> Q_[float]:
        """Maintenance respiration [g_gluc yr-1]."""
        params = self.params
        return assign(
            "R_M",
            (
                params["R_mL"] * self.B_L
                + params["R_mR"] * self.B_R
                + params["R_mS"] * self.B_S_star
            ),
        )

    @property
    def B_S_star(self) -> Q_[float]:
        """Biomass of 'living' sapwood [g_dw]."""
        params = self.params

        B_S_star_ = assign(
            "B_S_star",
            (
                (
                    1
                    - params["gamma_W"]
                    / (1 - params["gamma_X"])
                    * self.B_TS
                    / self.V_TS
                )
                * (self.B_S + self.C_S / params["gamma_C"] * self.B_TS / self.V_TS)
            ),
        )

        assert B_S_star_ >= 0
        return B_S_star_

    @property
    def C_S_star(self) -> Q_[float]:
        """Max. potential amount of labile carbon storage of bulk sapwood [g_gluc]."""
        return assign(
            "C_S_star", self.C_S_star_func(self.dbh, self.H, self.B_TS, self.SW)
        )

    @property
    def delta_S(self):
        """Concentration of labile carbon storage of bulk sapwood [g_gluc g_dw-1]."""
        return assign("delta_S", self.C_S / self.B_S)

    ###############################################################################

    @ureg.check(None, "[mass_dryweight]/[volume]")
    def compute_delta_W(self, rho_W: Q_[float]) -> Q_[float]:
        """Maximum labile C storage capacity of newly produced sapwood [g_gluc g_dw-1].

        Args:
            rho_W: density of newly produced sapwood [g_dw m-3]
        """
        params = self.params

        delta_W_ = assign(
            "delta_W",
            params["gamma_C"]
            * (1 - params["gamma_X"] - params["gamma_W"] * rho_W)
            / rho_W,
        )
        # don't check here, it will be negative for negative
        # rho_W during the process of solving the glucose budget
        #        assert delta_W_ > 0
        return delta_W_

    @ureg.check(None, "[length]", "[length]", None)
    def compute_rho_from_allometry(
        self, dbh: Q_[float], H: Q_[float], V_T: Q_[float] = None
    ) -> Q_[float]:
        """Wood density as predicted by allometries [g_dw m-3].

        Args:
            dbh: diameter at breas height [cm]
            H: tree height [m]
            V_T (optional): trunk volume [m3], default: self.V_T
        """
        mtrunk = self.mtrunk_func(dbh, H).to("g_dw")
        if V_T is None:
            V_T = self.V_T_func(dbh, H)

        return mtrunk / V_T

    @ureg.check(None, "[length]", "[length]", "[length]")
    def compute_rho_W(
        self, next_dbh: Q_[float], next_H: Q_[float], Delta_r: Q_[float]
    ) -> Q_[float]:
        """Density of newly produced sapwood [g_dw m-3].

        Args:
            next_dbh: dbh AFTER tree growth has taken place [cm]
            next_H: tree height AFTER tree growth has taken place [m]
        """
        next_V_T = self.V_T_func(next_dbh, next_H)
        Delta_V_T = next_V_T - self.V_T

        next_mtrunk = self.mtrunk_func(next_dbh, next_H).to("g_dw")

        rho_Wmin = self.params["rho_Wmin"]
        rho_Wmax = self.params["rho_Wmax"]

        delta_W_unit = var_infos["delta_W"]["target_unit"]  # g_gluc / g_dw
        delta_W = getattr(self, "_delta_W", Q_(0.0, delta_W_unit))

        if Delta_V_T != 0 * Delta_V_T:
            correction_for_C_ST = Q_(1, delta_W_unit) + delta_W
            # keep the followinf line for the wrong version in the manuscript
            # TODO: change the manuscript, remove the line, run everything again
            #            correction_for_C_ST = Q_(1, delta_W_unit)

            unit_correction = zeta_dw / zeta_gluc
            rho_W = (next_mtrunk - self.B_T) / Delta_V_T / correction_for_C_ST
            rho_W *= unit_correction  # make it to g_dw / m^3
        else:
            rho_W = rho_Wmax

        rho_W = np.minimum(rho_W, rho_Wmax)
        rho_W = np.maximum(rho_W, rho_Wmin)

        return assign("rho_W", rho_W)

    @ureg.check(None, "[mass_dryweight]")
    def compute_f_L_times_E(self, next_B_L: Q_[float]) -> Q_[float]:
        """Flux from transient pool to leaves [g_gluc yr-1].

        Args:
            next_B_L: leaf biomass of tree AFTER growth has taken place [g_dw]
        """
        params = self.params
        Delta_t = self.Delta_t
        Delta_B_L = next_B_L - self.B_L

        x = assign(
            "f_L_times_E",
            (
                (Delta_B_L + params["S_L"] * self.B_L * Delta_t)
                * (params["C_gL"] + params["delta_L"])
                / Delta_t
            ),
        )
        return x

    @ureg.check(None, "[mass_dryweight]")
    def compute_f_R_times_E(self, next_B_R: Q_[float]) -> Q_[float]:
        """Flux from transient pool to fine roots [g_gluc yr-1].

        Args:
            next_B_R: fine root biomass AFTER growth [g_dw]
        """
        params = self.params
        Delta_t = self.Delta_t
        Delta_B_R = next_B_R - self.B_R

        return assign(
            "f_R_times_E",
            (
                (Delta_B_R + params["S_R"] * self.B_R * Delta_t)
                * (params["C_gR"] + params["delta_R"])
                / Delta_t
            ),
        )

    @ureg.check(None, "[volume]")
    def compute_v_T(self, next_V_TH: Q_[float]) -> Q_[float]:
        """Sapwood to heartwood conversion rate of trunk [yr-1].

        Args:
            next_V_TH: trunk volume AFTER growth [m3]

        Returns:
            ``v_T``
        """
        Delta_t = self.Delta_t
        Delta_V_TH = next_V_TH - self.V_TH

        v_T = assign("v_T", Delta_V_TH / self.V_TS / Delta_t)
        return v_T

    @ureg.check(
        None,
        "1/[time]",
        "[volume]",
        "[mass_dryweight]/[volume]",
        "[mass_glucose]/[mass_dryweight]",
    )
    def compute_f_T_times_E(
        self, v_T: Q_[float], next_V_T: Q_[float], rho_W: Q_[float], delta_W: Q_[float]
    ) -> Q_[float]:
        """Flux from labile pool to trunk [g_gluc yr-1].

        Args:
            v_T: sapwood to heartwood conversion rate of trunk [yr-1]
            next_V_T: trunk volume AFTER growth [m3]
            rho_W: density of newly produced sapwood [g_dw m-3]
            delta_W: maximum labile C storage capacity of newly
                produced sapwood [g_gluc g_dw-1]
        """
        Delta_t = self.Delta_t
        Delta_V_T = next_V_T - self.V_T

        B_TS = self.B_TS

        # delta_S(t) looks like it should come from 'next_tree', but
        # the definition of delta_S (Ogle and Pacala, SD) shows otherwise
        # this is inconsistent with the rest of their notation
        delta_S = self.delta_S

        f_T_times_E_ = assign(
            "f_T_times_E",
            (
                (
                    rho_W
                    * Delta_V_T
                    #                    - delta_S / self.params["C_gHW"] * v_T * B_TS * Delta_t
                )
                * (self.params["C_gW"] + delta_W)
                / Delta_t
            ),
        )

        # the ACGCA formula allows this to become negative if there is sapwood to
        # heartwood conversion but (almost) no trunk volume growth
        # adapted formula, so this here should be superfluous
        if f_T_times_E_ < 0 * f_T_times_E_:
            f_T_times_E_ = 0 * f_T_times_E_

        return f_T_times_E_

    @ureg.check(None, "1/[time]", "[mass_dryweight]", "[mass_glucose]/[mass_dryweight]")
    def compute_f_O_times_E(
        self, v_O: Q_[float], next_B_OS: Q_[float], delta_W: Q_[float]
    ) -> Q_[float]:
        """Flux from labile pool to coarse roots and branches [g_gluc yr-1].

        Args:
            v_O: sapwood to heartwood conversion rate of other [yr-1]
            next_B_OS: other sapwood biomass AFTER growth [g_dw]
            delta_W: maximum labile C storage capacity of newly
                produced sapwood [g_gluc g_dw-1]
            next_tree (same class): Tree with allometries AFTER solving
                the carbon budget for ``Delta_r``.
        """
        params = self.params
        B_OS = self.B_OS
        Delta_B_OS = next_B_OS - B_OS
        Delta_t = self.Delta_t

        return assign(
            "f_O_times_E",
            (Delta_B_OS + Delta_t * (params["S_O"] + v_O) * B_OS)
            / Delta_t
            * (params["C_gW"] + delta_W),
        )

    @ureg.check(
        None,
        "[mass_glucose]/[time]",
        "[mass_glucose]/[mass_dryweight]",
        "[mass_glucose]/[time]",
        "[mass_glucose]/[time]",
        "1/[time]",
        "1/[time]",
        "[mass_glucose]/[time]",
    )
    def update_C_S(
        self,
        E: Q_[float],
        delta_W: Q_[float],
        f_T_times_E: Q_[float],
        f_O_times_E: Q_[float],
        v_T: Q_[float],
        v_O: Q_[float],
        f_CS_times_CS: Q_[float],
    ):
        """Update C_S after carbon budget is solved for ``Delta_r``.

        Args:
            Anet_gluc: incoming net radiaton [g_gluc yr-1]
            delta_W: maximum labile C storage capacity of newly
                produced sapwood [g_gluc g_dw-1]
            f_T_times_E: flux from labile pool to trunk [g_gluc yr-1]
            f_O_times_E: flux from labile pool to other [g_gluc yr-1]
            v_T: sapwood to heartwood conversion rate of trunk [yr-1]
            v_O: sapwood to heartwood conversion rate of other [yr-1]
            f_CS_times_CS: in "static" state amount of glucose to use from reserves
        """
        params = self.params

        Delta_t = self.Delta_t

        B_OS = self.B_OS
        B_TS = self.B_TS

        f_T = f_T_times_E / E
        f_O = f_O_times_E / E

        delta_S = self.delta_S

        Delta_C_S_positive_part = (
            E * delta_W / (params["C_gW"] + delta_W) * (f_T + f_O)
        ) * Delta_t
        #        print(1347, Delta_C_S_positive_part)
        Delta_C_S_negative_part_without_B_OS = (
            +zeta_dw / zeta_gluc * delta_S / params["C_gHW"] * v_O * B_OS * Delta_t
            + delta_S * params["S_O"] * B_OS * Delta_t
            + zeta_dw / zeta_gluc * delta_S / params["C_gHW"] * v_T * B_TS * Delta_t
        )
        Delta_C_S_negative_part_B_OS = +f_CS_times_CS * Delta_t
        Delta_C_S_negative_part = (
            Delta_C_S_negative_part_without_B_OS + Delta_C_S_negative_part_B_OS
        )
        #        print(1357, Delta_C_S_negative_part)

        # check if more C_S is trying to be used than is actually available
        # BEFORE new glucose comes in from the transient pool E
        try:
            assert self.C_S - Delta_C_S_negative_part >= 0
        except AssertionError:
            # Too much C_S is being tried to use.

            # if this fails, then something is seriously wrong
            print(1469, self.C_S, Delta_C_S_negative_part_without_B_OS)
            print(f_T, f_O, v_T, v_O, B_TS, B_OS)
            assert self.C_S - Delta_C_S_negative_part_without_B_OS >= 0

            # if we end up here, then we need too much C_S for regrowth of "other"
            # we call emergency
            raise TreeShrinkError()

        Delta_C_S = Delta_C_S_positive_part - Delta_C_S_negative_part

        # store it to use it as an approximation of the new value
        # in computation of rho_W
        self._Delta_C_S = Delta_C_S  # pylint: disable=attribute-defined-outside-init
        #        print(1378, self.C_S, Delta_C_S)
        self.C_S = assign("C_S", self.C_S + Delta_C_S)
        assert self.C_S >= 0

    @ureg.check(None, "[mass_dryweight]")
    def compute_v_O(self, next_B_OH: Q_[float]) -> Q_[float]:
        """Sapwood to heartwood conversion rate of coarse roots and branches [yr-1].

        Args:
            next_B_OH: other heartwood biomass AFTER growth [g_dw]

        Returns:
            ``v_O``
        """
        params = self.params
        B_OH = self.B_OH
        B_OS = self.B_OS
        Delta_B_OH = next_B_OH - B_OH
        Delta_t = self.Delta_t
        v_O = assign(
            "v_O",
            (Delta_B_OH + Delta_t * params["S_O"] * B_OH)
            / (Delta_t * (1 + self.delta_S / params["C_gHW"]) * B_OS),
        )
        return v_O

    def SLA_func(self) -> Q_[float]:
        """Specific leaf area [m2 kg_dw-1]."""
        SLA = self.params["SLA"]
        return SLA

    @ureg.check(None, "[length]", "[length]")
    def B_L_func(self, dbh: Q_[float], H: Q_[float]) -> Q_[float]:
        """Leaf biomass [g_dw]."""

        dbh_cm = dbh.to("cm").magnitude
        H_m = H.to("m").magnitude
        species = self.species
        components = ["leaves"]
        numerator = Q_(
            allometries(dbh_cm, H_m, species, components, self.custom_species_params),
            "kg_dw",
        ).to("g_dw")

        B_L_g_dw = numerator / (1 + self.params["delta_L"] * zeta_gluc / zeta_dw)

        return assign("B_L", B_L_g_dw)  # g_dw

    @ureg.check(None, "[length]", "[length]")
    def mtrunk_func(self, dbh: Q_[float], H: Q_[float]) -> Q_[float]:
        """Trunk biomass from external allometries [g_dw]."""
        dbh_cm = dbh.to("cm").magnitude
        H_m = H.to("m").magnitude
        components = ["stemwood", "stembark", "stump"]
        mtrunk = Q_(
            allometries(
                dbh_cm, H_m, self.species, components, self.custom_species_params
            ),
            "kg_dw",
        )

        return mtrunk.to("g_dw")

    @ureg.check(None, "[length]", "[length]")
    def mother_func(self, dbh: Q_[float], H: Q_[float]) -> Q_[float]:
        """Biomass of coarse roots and branches from external allometries [g_dw]."""
        dbh_cm = dbh.to("cm").magnitude
        H_m = H.to("m").magnitude
        components = ["livingbranches", "roots"]
        mother = Q_(
            allometries(
                dbh_cm, H_m, self.species, components, self.custom_species_params
            ),
            "kg_dw",
        )

        return mother.to("g_dw")

    @ureg.check(None, "[length]", "[length]")
    def mstump_func(self, dbh: Q_[float], H: Q_[float]) -> Q_[float]:
        """Stump biomass from external allometries [g_dw]."""
        dbh_cm = dbh.to("cm").magnitude
        H_m = H.to("m").magnitude
        components = ["stump"]
        mstump = Q_(
            allometries(
                dbh_cm, H_m, self.species, components, self.custom_species_params
            ),
            "kg_dw",
        )

        return mstump.to("g_dw")

    #    @ureg.check(None, "[length]", "[length]")
    #    def LA_func(self, dbh: Q_[float], H: Q_[float]) -> Q_[float]:
    @ureg.check(None, "[mass_dryweight]")
    def LA_func(self, B_L: Q_[float]) -> Q_[float]:
        """Total leaf area [m2]."""
        SLA = self.SLA_func()
        #        B_L = self.B_L_func(dbh, H)
        return assign("LA", SLA * B_L)

    @ureg.check(None, "[length]", "[length]")
    def B_R_func(self, dbh: Q_[float], H: Q_[float]) -> Q_[float]:
        """Fine root biomass [g_dw]."""
        params = self.params
        B_L = self.B_L_func(dbh, H)
        return assign("B_R", params["rho_RL"] * B_L)

    @ureg.check(None, "[length]", "[length]")
    def lambda_func(self, dbh: Q_[float], H: Q_[float]) -> float:
        """Ratio of other (branches + coarse roots) to trunk.

        This ratio comes from external allometries based on
        given diameter at breast height ``dbh`` and tree height ``H``.
        """
        mother = self.mother_func(dbh, H).to("g_dw")
        mtrunk = self.mtrunk_func(dbh, H).to("g_dw")

        return mother / mtrunk

    @ureg.check(None, "[length]", "[length]")
    def lambda_H_func(self, dbh: Q_[float], H: Q_[float]) -> float:
        """Ratio of other heartwood to trunk heartwood by external allometries."""
        return self.lambda_func(dbh, H)

    @ureg.check(None, "[length]", "[length]")
    def lambda_S_func(self, dbh: Q_[float], H: Q_[float]) -> float:
        """Ratio of other sapwood to trunk sapwood by external allometries."""
        return self.lambda_func(dbh, H)

    @ureg.check(None, "[length]", "[length]", "[mass_dryweight]")
    def B_OS_func(self, dbh: Q_[float], H: Q_[float], B_TS: Q_[float]) -> Q_[float]:
        """Coarse roots and branches (other) sapwood biomass [g_dw]."""
        lambda_S = self.lambda_S_func(dbh, H)
        return assign("B_OS", lambda_S * B_TS)

    @ureg.check(None, "[length]", "[length]", "[mass_dryweight]")
    def B_S_func(self, dbh: Q_[float], H: Q_[float], B_TS: Q_[float]) -> Q_[float]:
        """Sapwood biomass (coarse roots and branches + trunk) [g_dw]."""
        B_OS = self.B_OS_func(dbh, H, B_TS)
        return assign("B_S", B_TS + B_OS)

    @ureg.check(None, "[length]", "[length]", "[mass_dryweight]")
    def B_OH_func(self, dbh: Q_[float], H: Q_[float], B_TH: Q_[float]) -> Q_[float]:
        """Coarse roots and branches (other) heartwood biomass [g_dw]."""
        lambda_H = self.lambda_H_func(dbh, H)
        return assign("B_OH", lambda_H * B_TH)

    @ureg.check(
        None,
        "[mass_dryweight]",
        "[mass_dryweight]",
        "[mass_glucose]",
        "[mass_dryweight]",
    )
    def B_T_func(
        self, B_TS: Q_[float], B_TH: Q_[float], C_S: Q_[float], B_OS: Q_[float]
    ) -> Q_[float]:
        """Trunk biomass [g_dw].

        ``B_T = B_TS + B_TH`` plus ``C_S`` associated to trunk sapwood.
        """
        return assign("B_T", B_TS + B_TH + B_TS / (B_TS + B_OS) * C_S.to(B_TS.units))

    @ureg.check(None, "[length]")
    def H_func(self, dbh: Q_[float]) -> Q_[float]:
        """Tree height from diameter at breast height [m].

        Based on Näslund equation parameterized using Hyytiälä stand inventory
        from 2008.
        """
        return assign("H", tree_utils.H_func(dbh.to("cm").magnitude, self.species))

    @ureg.check(None, "[length]")
    def h_B_func(self, H: Q_[float]) -> Q_[float]:
        """Height at which trunk transitions from a neiloid to a paraboloid [m].

        Args:
            H: tree height [m]
        """
        eta_B = self.params["eta_B"]
        return assign("h_B", tree_utils.h_B_func(H.to("m").magnitude, eta_B.magnitude))

    @ureg.check(None, "[length]")
    def h_C_func(self, H: Q_[float]) -> Q_[float]:
        """Height at which trunk transitions from a paraboloid to a cone [m].

        Args:
            H: tree height [m]
        """
        eta_C = self.params["eta_C"]
        return assign("h_C", tree_utils.h_C_func(H.to("m").magnitude, eta_C.magnitude))

    @ureg.check(None, "[length]", "[length]")
    def r_B_func(self, r: Q_[float], H: Q_[float]) -> Q_[float]:
        """Radius at which trunk transitions from a neiloid to a paraboloid [m].

        Args:
            r: tree radois at trunk base [m]
            H: tree height [m]
        """
        eta_B = self.params["eta_B"]
        return assign(
            "r_B",
            tree_utils.r_B_func(
                r.to("m").magnitude, H.to("m").magnitude, eta_B.magnitude
            ),
        )

    @ureg.check(None, "[length]", "[length]")
    def r_C_func(self, r: Q_[float], H: Q_[float]) -> Q_[float]:
        """Radius at which trunk transitions from a paraboloid to a cone [m].

        Args:
            r: tree radois at trunk base [m]
            H: tree height [m]
        """
        eta_B = self.params["eta_B"]
        eta_C = self.params["eta_C"]
        return assign(
            "r_C",
            tree_utils.r_C_func(
                r.to("m").magnitude,
                H.to("m").magnitude,
                eta_B.magnitude,
                eta_C.magnitude,
            ),
        )

    @ureg.check(None, "[length]", "[length]")
    def V_T_func(self, dbh: Q_[float], H: Q_[float]) -> Q_[float]:
        """Trunk volume as computed by ACGCA [m3]."""
        dbh = dbh.to("cm").magnitude
        H = H.to("m").magnitude

        species = self.species
        custom_species_params = self.custom_species_params
        #        V_T = Q_(tree_utils.V_T_func(dbh, species, custom_species_params), "m^3")

        V_T = Q_(tree_utils.V_T_func(dbh, H, species, custom_species_params), "m^3")

        return assign("V_T", V_T)

    @ureg.check(None, "[length]", "[length]", None)
    def H_TH_func(
        self, r: Q_[float], H: Q_[float], SW: Q_[float]  # = None
    ) -> Q_[float]:
        """Height of trunk heartwood section [m]."""
        r = r.to("m").magnitude
        H = H.to("m").magnitude

        H_TH = Q_(
            tree_utils.H_TH_func(
                SW.magnitude, r, H, self.species, self.custom_species_params
            ),
            "m",
        )
        return assign("H_TH", H_TH)

    @ureg.check(None, "[length]", "[length]", "[length]")
    def V_TH_func(
        self, dbh: Q_[float], H: Q_[float], SW: Q_[float]
    ) -> Q_[float]:  # = None
        """Volume of trunk heartwood section [m3]."""
        dbh = dbh.to("cm").magnitude
        H = H.to("m").magnitude

        V_TH = Q_(
            tree_utils.V_TH_func(
                SW.magnitude, dbh, H, self.species, self.custom_species_params
            ),
            "m^3",
        )
        return assign("V_TH", V_TH)

    @ureg.check(None, "[length]", "[length]", "[length]")
    def V_TS_func(
        self, dbh: Q_[float], H: Q_[float], SW: Q_[float] = None
    ) -> Q_[float]:
        """Volume of trunk sapwood [m3]."""
        V_T = self.V_T_func(dbh, H)
        V_TH = self.V_TH_func(dbh, H, SW)
        return assign("V_TS", V_T - V_TH)

    @ureg.check(None, "[length]", "[length]", "[mass_dryweight]", None)
    def C_S_star_func(
        self, dbh: Q_[float], H: Q_[float], B_TS: Q_[float], SW: Q_[float] = None
    ) -> Q_[float]:
        """Max. potential amount of labile carbon storage of bulk sapwood [g_gluc]."""
        params = self.params
        V_TS = self.V_TS_func(dbh, H, SW)
        B_S = self.B_S_func(dbh, H, B_TS)
        C_S_star_ = (
            params["gamma_C"]
            * ((1 - params["gamma_X"]) * V_TS / B_TS - params["gamma_W"])
            * B_S
        )
        assert C_S_star_ >= 0
        return assign("C_S_star", C_S_star_)

    @ureg.check(None, "[length]", "[time]")
    def SW_func(self, r: Q_[float], tree_age: Q_[float]) -> Q_[float]:
        """Sapwood radius at trunk base [m].

        Args:
            r: radius at trunk base [m]
            tree_age: age of the tree [yr]
        """
        dbh = assign(
            "dbh",
            tree_utils.dbh_from_r(
                r.to("m").magnitude, self.species, self.custom_species_params
            ),
        )

        H = self.H_func(dbh)

        if hasattr(self, "B_TS") and hasattr(self, "C_S"):
            B_TS = self.B_TS.to("kg_dw")
            V_TS = self.V_TS.to("m^3")
            B_TH = self.B_TH.to("kg_dw")
            V_T = self.V_T.to("m^3")

            rho_TS = B_TS / V_TS
            rho_T = (B_TS + B_TH) / V_T
        else:
            # at tree init: the value is not important,
            # it just needs to be the same and then cancels out
            rho_TS = Q_(1.0, "kg_dw/m^3")
            rho_T = rho_TS

        SW = Q_(
            tree_utils.solve_for_SW(
                r.to("m").magnitude,
                dbh.to("cm").magnitude,
                H.to("m").magnitude,
                #                self.tree_age.to("yr").magnitude,
                tree_age.to("yr").magnitude,
                rho_TS.magnitude,
                rho_T.magnitude,
                self.species,
                self.custom_species_params,
            ),
            "m",
        )
        if np.abs(SW - r) < Q_(1e-03, "m"):
            SW = r

        # (trunk) heartwood cannot grow back
        if r > self.r:
            if self.V_TH_func(dbh, H, SW) < self.V_TH:
                SW = self.SW
            if hasattr(self, "SW"):
                if self.H_TH_func(r, H, SW) < self.H_TH:
                    SW = self.SW

        return assign("SW", r if r < SW else SW)
