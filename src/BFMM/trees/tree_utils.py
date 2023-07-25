"""Helper functions for tree properties.

Source:

    .. [1] Näslund (1936), "Skogsförsöksanstaltens gallringsförsök i 
        tallskog", Meddelanden från Statens Skogsförsöksanstalt 29. 169

    .. [2] Siipilehto J. (2000). A comparison of two parameter
        prediction methods for stand structure in Finland.
        Silva Fennica vol. 34 no. 4 article id 617.

    .. [3] Tavahnainen & Forss (2008), For. Ecol. Manag., Fig. 4
        https://doi.org/10.14214/sf.617

    .. [4] Ogle and Pacala (2009), Tree Physiology


    .. [5] Laasenaho (1988)

    .. [6] Vanninen and Mäkelä (2005), Tree Physiology

    .. [7] Sellin (1994), Canadian Journal of Forest Research
"""
from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root

from .. import Q_
from ..params import BREAST_HEIGHT
from ..productivity.vegetation import \
    crown_biomass_distr as _crown_biomass_distr
from ..type_aliases import SpeciesParams
from ..utils import make_lru_cacheable
from .single_tree_params import initialize_params, species_params

params_crown = {
    "pine": {"k": 2.0, "a": 0.894, "b": 0.185, "c": 3.0, "d": 0.76, "e": 0},
    "spruce": {"k": 3.0, "a": 1.811, "b": 0.308, "c": 1.99, "d": 0.51, "e": -7.7e-03},
}
"""
Tree height (Näslund height curve, [1]_) and crown base height parameters:

- tree height (k, a, b): [2]_ 
- crown base height (c, d, e): [3]_

"""


def compute_crown_boundaries(dbh: float, species: str) -> Tuple[float, float]:
    """Compute tree height and crown base height based on dbh.

    Height model based on Näslund equation parameterized by
    Siipilehto & Kangas (2015) using Hyytiälä stand inventory from 2008.
    Crown base model based on Tahvanainen & Forss, 2008 For. Ecol.
    Manag. Fig 4.

    Args:
        dbh: diameter in breast height [cm]
        species: tree species, element of ["pine", "spruce", "birch"]

    Returns:
        Tuple of tree height and crown base height.

        - htop: tree height [m]
        - hbase: crown base height [m]

    Notes:
        Source: [1]_
    """
    assert isinstance(dbh, float)

    p = params_crown[species]
    k, a, b, c, d, e = p["k"], p["a"], p["b"], p["c"], p["d"], p["e"]
    #    k, a, b, c, d = p["k"], p["a"], p["b"], p["c"], p["d"]
    htop = 1.3 + dbh**k / (a + b * dbh) ** k
    hbase = np.maximum(0, -c + d * htop + e * htop**2.0)
    #    hbase = np.maximum(0, -c + d * htop)
    return htop, hbase


def crown_biomass_distr(
    htop: float,
    hbase: float,
    z: np.ndarray,
    species: str,
) -> np.ndarray:
    """
    Normalized leaf area density on height grid ``z`` [m-1].

    Args:
        htop: tree height [m]
        hbase: tree crown base height [m]
        z: grid of tree height layer boundaries [m]
        species: element of ["pine", "spruce", "birch"]
    """
    assert isinstance(htop, float)
    assert isinstance(hbase, float)

    lad_normalized, _ = _crown_biomass_distr(species, z, htop, hbase)
    return lad_normalized


# def freeze_crown_biomass_distr_from_dbh(dbh, z, species):
#    """
#    Can be used as freezing function in the ``make_lru_cachable`` decorator.
#    """
#    if isinstance(dbh, Q_):
#        dbh = dbh.to("cm").magnitude
#
#    if not isinstance(dbh, float):
#        dbh = tuple(dbh)
#
#    if isinstance(z, Q_):
#        z = z.to("m").magnitude
#
#    return (dbh, tuple(z), species), dict()


# @make_lru_cacheable(freeze_crown_biomass_distr_from_dbh)
def crown_biomass_distr_from_dbh(dbh: float, z: np.ndarray, species: str) -> np.ndarray:
    """
    Return normalized leaf area density on height grid ``z`` [m-1].

    Args:
        dbh: tree diameter in breast height [cm]
        z: grid of tree height layer boundaries [m]
        species: element of ["pine", "spruce", "birch"]
    """
    assert isinstance(dbh, float)
    assert isinstance(z, np.ndarray)

    htop, hbase = compute_crown_boundaries(dbh, species)
    lad_normalized = crown_biomass_distr(htop, hbase, z, species)

    return lad_normalized


def lad_normed_func(dbh: float, z: np.ndarray, species: str) -> np.ndarray:
    """
    Normalized leaf area density over ``z``, integrates to 1 [m-1].

    Args:
        species: tree species, element of ["pine", "spruce", "birch"]
        z: grid of tree height layer boundaries [m]
        dbh: diameter at breast height [cm]
    """
    assert isinstance(dbh, float)
    return crown_biomass_distr_from_dbh(dbh, z, species)


def H_func(dbh: float, species: str) -> float:
    """Compute tree height from diameter at breast height.

    Based on Näslund equation parameterized using Hyytiälä stand inventory
    from 2008.

    Args:
        dbh: diameter in breast height [cm]
        species: element of ["pine", "spruce", "birch"]
    """
    assert isinstance(dbh, float)
    return compute_crown_boundaries(dbh, species)[0]


def V_T_func_ACGCA(
    dbh: float, H: float, species: str, custom_species_params: SpeciesParams
) -> float:
    """Trunk volume as computed by ACGCA [m3].

    Args:
        dbh: diameter in breast height [cm]
        H: tree height [m]
        species: element of ["pine", "spruce", "birch"]
        custom_species_params: tree species parameters (for all species)

    Notes:
        Source: [4]_ (SI, Eq.(9))
    """
    #    H = H_func(dbh, species)
    r = r_from_dbh(
        dbh,
        species,
        custom_species_params,
    )

    params = initialize_params(custom_species_params[species])

    eta_B = params["eta_B"].magnitude
    eta_C = params["eta_C"].magnitude
    h_B = h_B_func(H, eta_B)
    h_C = h_C_func(H, eta_C)
    r_B = r_B_func(r, H, eta_B)
    r_C = r_C_func(r, H, eta_B, eta_C)

    V_neiloid = np.pi / 4 * r**2 / H**3 * (H**4 - (H - h_B) ** 4)
    V_paraboloid = (
        np.pi / 2 * r_B**2 / (H - h_B) * (h_B**2 - h_C**2 + 2 * H * (h_C - h_B))
    )
    V_cone = np.pi / 3 * r_C**2 * (H - h_C)

    V_T = V_neiloid + V_paraboloid + V_cone

    return V_T


def V_T_func_Laasenaho(dbh: float, H: float, species: str):
    """Trunk volume as computed by Laasenaho [m3].

    Args:
        dbh: diameter in breast height [cm]
        H: tree height [m]
        species: element of ["pine", "spruce", "birch"]

    Notes:
        Source: [5]_, Table (61.3), Eq. (52.5)
    """
    if species == "pine":
        intercept = -3.32176
        b1 = 2.01395
        b2 = 2.07025
        b3 = -1.07209
        b4 = -0.0032473

    elif species == "spruce":
        intercept = -3.77543
        b1 = 1.91505
        b2 = 2.82541
        b3 = -1.53547
        b4 = -0.0085726

    else:
        raise ValueError(f"Unkwonw species: {species}")

    return (
        np.exp(
            intercept
            + b1 * np.log(dbh)
            + b2 * np.log(H)
            + b3 * np.log(H - 1.3)
            + b4 * dbh
        )
        * 1e-3  # from dm^3 to m^3
    )


def V_T_func(dbh: float, H: float, species: str, custom_species_params: SpeciesParams):
    """Trunk volume [m3].

    Args:
        dbh: diameter in breast height [cm]
        H: tree height [m]
        species: element of ["pine", "spruce", "birch"]
        custom_species_params: tree species parameters (for all species)
    """
    # for small dbhs Laasenaho volume formula is useless
    # use ACGCA volume formula and make it match Laasenaho at ``dbh_M``
    params = initialize_params(custom_species_params[species])
    dbh_M = params["dbh_M"].to("cm").magnitude
    if dbh <= dbh_M:
        #        H_M = H_func(dbh_M, species)
        #        V_T_Laasenaho_at_dbh_M = V_T_func_Laasenaho(dbh_M, H_M, species)
        #        V_T_ACGCA_at_dbh_M = V_T_func_ACGCA(dbh_M, H_M, species, custom_species_params)
        #        scale_factor = V_T_Laasenaho_at_dbh_M / V_T_ACGCA_at_dbh_M
        #        return scale_factor * V_T_func_ACGCA(dbh, H, species, custom_species_params)
        def f(x):
            H_x = H_func(x, species)
            return V_T_func_Laasenaho(x, H_x, species)

        y_M = f(dbh_M)
        y_M_prime = (f(dbh_M + 0.01) - f(dbh_M - 0.01)) / 0.02

        #            print(species, x1, y1, y1_prime)
        r = dbh_M * y_M_prime / y_M
        a = (y_M - y_M_prime) / (dbh_M**r - r * dbh_M ** (r - 1))
        y = a * dbh**r
        return y

    return V_T_func_Laasenaho(dbh, H, species)


def H_TH_func(
    SW: float, r: float, H: float, species: str, custom_species_params: SpeciesParams
) -> Q_[float]:
    """Height of trunk heartwood section [m].

    The formula is from Ogle and Pacala (SI, Eq. (13)) with one correction.

    Args:
        SW: sapwood widht [m]
        r: tree radius at trunk base [m]
        H: tree height [m]
        species: element of ["pine", "spruce", "birch"]
        custom_species_params: tree species parameters (for all species)

    Notes:
        Source: [4]_
    """
    params = initialize_params(custom_species_params[species])

    eta_B = params["eta_B"].magnitude
    eta_C = params["eta_C"].magnitude
    h_B = h_B_func(H, eta_B)
    h_C = h_C_func(H, eta_C)
    r_B = r_B_func(r, H, eta_B)
    r_C = r_C_func(r, H, eta_B, eta_C)

    if r - SW <= 0:
        H_TH = 0.0
    elif r_B - SW <= 0:
        H_TH = H - H * (SW / r) ** (2 / 3)  # wrong in Ogle and Pacala
    elif r_C - SW <= 0:
        H_TH = H - (H - h_B) * (SW / r_B) ** 2
    else:
        H_TH = np.minimum(H - SW, (H * (r_C - SW) + h_C * SW) / r_C)

    return H_TH


def V_TH_func_ACGCA(
    SW: float, dbh: float, species: str, custom_species_params: SpeciesParams
) -> Q_[float]:
    """Volume of trunk heartwood section [m3].

    Args:
        SW: sapwood widht [m]
        dbh: tree diameter ar breast height [cm]
        species: element of ["pine", "spruce", "birch"]
        custom_species_params: tree species parameters (for all species)

    Notes:
        Source: [4]_ (SI, Eq. (14))
    """
    r = r_from_dbh(dbh, species, custom_species_params)

    params = initialize_params(custom_species_params[species])

    H = H_func(dbh, species)
    eta_B = params["eta_B"].magnitude
    eta_C = params["eta_C"].magnitude
    h_B = h_B_func(H, eta_B)
    h_C = h_C_func(H, eta_C)
    r_B = r_B_func(r, H, eta_B)
    r_C = r_C_func(r, H, eta_B, eta_C)
    H_TH = H_TH_func(SW, r, H, species, custom_species_params)

    if r - SW <= 0:
        V_TH = 0
    elif r_B - SW <= 0:
        V_TH = np.pi / 4 * H_TH * (r - SW) ** 2
    elif r_C - SW <= 0:
        V_TH = np.pi / 4 * (r - SW) ** 2 * (
            (H_TH**4 - (H_TH - h_B) ** 4) / H_TH**3
        ) + np.pi / 2 * (r_B - SW) ** 2 * (H_TH - h_B)
    else:
        V_TH = (
            np.pi / 4 * (r - SW) ** 2 / H_TH**3 * (H_TH**4 - (H_TH - h_B) ** 4)
            + np.pi
            / 2
            * (r_B - SW) ** 2
            / (H_TH - h_B)
            * (h_B**2 - h_C**2 + 2 * H_TH * (h_C - h_B))
            + np.pi / 3 * (r_C - SW) ** 2 * (H_TH - h_C)
        )

    return V_TH


def V_TH_func(
    SW: float, dbh: float, H: float, species: str, custom_species_params: SpeciesParams
) -> Q_[float]:
    """Volume of trunk heartwood section [m3].

    Args:
        SW: sapwood widht [m]
        dbh: tree diameter ar breast height [cm]
        H: tree height [m]
        species: element of ["pine", "spruce", "birch"]
        custom_species_params: tree species parameters (for all species)

    Notes:
        Source: [4]_ (SI, Eq. (14))
    """
    V_TH = V_TH_func_ACGCA(SW, dbh, species, custom_species_params)
    # scale down to Laasenaho
    V_T = V_T_func(dbh, H, species, custom_species_params)
    #    V_T_Laasenaho = V_T_func_Laasenaho(dbh, H, species)
    V_T_ACGCA = V_T_func_ACGCA(dbh, H, species, custom_species_params)
    #    if dbh > params["dbh_M"].to("cm").magnitude:
    #        scaling_factor = V_T_Laasenaho / V_T_ACGCA
    #    else:
    #        scaling_factor = 1.0

    scaling_factor = V_T / V_T_ACGCA
    return scaling_factor * V_TH


def H_crown(dbh: float, species: str) -> float:
    """Compute height of crown base from diameter at breast height [m].

    Based on Tahvanainen & Forss, 2008 For. Ecol. Manag. Fig 4.

    Args:
        dbh: diameter in breast height [cm]
        species: element of ["pine", "spruce", "birch"]
    """
    return compute_crown_boundaries(dbh, species)[1]


def dbh_from_mleaf(
    mleaf: float,
    species: str,
    allometries: Callable,
    custom_species_params: SpeciesParams,
) -> float:
    """Compute diameter at breast height from leaf biomass [cm].

    Args:
        mleaf: leaf biomass [kg_dw]
        species: element of ["pine", "spruce", "birch"]
        allometries: e.g. :func:`~.allometry.allometries_repola_and_lehtonen`
        custom_species_params: tree species parameters (for all species)
    """
    dbh0 = 5.0

    def g(dbh):
        dbh = float(dbh)
        H = H_func(dbh, species)
        mleaf_tmp = allometries(dbh, H, species, ["leaves"], custom_species_params)
        return mleaf - mleaf_tmp

    root_res = root(g, dbh0)
    if not root_res.success:
        raise ValueError("Could not determine dbh from mleaf")

    return float(root_res.x)


def r_from_mleaf(
    mleaf: float,
    species: str,
    allometries: Callable,
    custom_species_params: SpeciesParams,
):
    """Compute radius at trunk base height from leaf biomass [m].

    Args:
        mleaf: leaf biomass [kg_dw]
        species: element of ["pine", "spruce", "birch"]
        allometries: e.g. :func:`~.allometry.allometries_repola_lehtonen`
        custom_species_params: tree species parameters (for all species)
    """
    dbh = dbh_from_mleaf(mleaf, species, allometries, custom_species_params)
    return r_from_dbh(dbh, species, custom_species_params)


def h_B_func(H: float, eta_B: float):
    """Height at which trunk transitions from a neiloid to a paraboloid [m].

    Args:
        H: tree height [m]
        eta_B: relative height at which trunk transitions from a neiloid
            to a paraboloid [-]
    """
    return eta_B * H


def h_C_func(H: float, eta_C: float):
    """Height at which trunk transitions from a paraboloid to a cone [m].

    Args:
        H: tree height [m]
        eta_C: relative height at which trunk transitions from
            a paraboloid to a cone [-]
    """
    return eta_C * H


def r_B_func(r: float, H: float, eta_B: float) -> float:
    """Compute radius at which trunk transitions from a neiloid to a paraboloid [m].

    Args:
        r: tree radois at trunk base [m]
        H: tree height [m]
        eta_B: relative height at which trunk transitions from a neiloid
            to a paraboloid [-]
    """
    if H == 0.0:
        r_B = 0.0
    else:
        r_B = r * np.sqrt(((H - h_B_func(H, eta_B)) / H) ** 3)

    return r_B


def r_C_func(r: float, H: float, eta_B: float, eta_C: float) -> float:
    """Compute radius at which trunk transitions from a paraboloid to a cone [m].

    Args:
        r: radius at trunk base [m]
        H: tree height [m]
        eta_B: relative height at which trunk transitions from a neiloid
            to a paraboloid [-]
        eta_C: relative height at which trunk transitions from
            a paraboloid to a cone [-]
    """
    D = (H - h_C_func(H, eta_C)) / (H - h_B_func(H, eta_B))
    return r_B_func(r, H, eta_B) * np.sqrt(D)


def r_BH_func(r: float, H: float, BH: float, eta_B: float, eta_C: float) -> float:
    """Compute radius at breast height from radius at trunk base and tree height [m].

    Args:
        r: radius at trunk base [m]
        H: tree height [m]
        BH: breast height [m]
        eta_B: relative height at which trunk transitions from a neiloid
            to a paraboloid [-]
        eta_C: relative height at which trunk transitions from
            a paraboloid to a cone [-]

    Notes:
        Source: [4]_ (SI, Eq.(34))
    """
    r_B = r_B_func(r, H, eta_B)
    r_C = r_C_func(r, H, eta_B, eta_C)
    h_B = h_B_func(H, eta_B)
    h_C = h_C_func(H, eta_C)

    if BH <= h_B:
        r_BH = r * ((H - BH) / H) ** (3 / 2)
    elif BH <= h_C:
        r_BH = r_B * ((H - BH) / (H - h_B)) ** (1 / 2)
    elif BH < H:
        r_BH = r_C * (H - BH) / (H - h_C)
    else:
        r_BH = 0.0

    return r_BH


# @lru_cache(maxsize=50_000)
# @make_lru_cacheable(maxsize=100) # caching does not seem to speed up
def r_from_dbh(dbh: float, species: str, custom_species_params: SpeciesParams) -> float:
    """Compute the trunk base radius r for :class:`~single_tree_allocation.SingleTreeAllocation` for given dbh [m].

    Args:
        dbh: diameter at breast height to match [cm]
        species: element of ["pine", "spruce", "birch"]
        custom_species_params: tree species parameters (for all species)

    Returns:
        trunk base radius of tree such that 2*tree.r_BH = dbh [m]
    """
    assert isinstance(dbh, float)

    if dbh == 0:
        return 0.0

    if dbh < 1.0:
        # dbh too small
        return -5.0 * dbh

    if dbh < 3.0:
        # dbh small, we use
        # Laasasenaho, 1982; Repola, 2009, p. 631
        # from dbh to r at trunk base
        f = interp1d(
            [1.0, 3.0],
            [(2 + 1.25 * 1.0) / 2 / 100, r_from_dbh(3.0, species, species_params)],
        )
        return float(f(dbh))

    dbh_m = dbh * 0.01

    H = H_func(dbh, species)

    params = initialize_params(custom_species_params[species])
    #    r0 = 0.5 * dbh_m
    r0 = 0.001

    def g(r):
        """
        Args:
            r [m]: tree radius at trunk base

        Returns:
            res [m]: dbh - 2*r_BH(r)
        """
        r_BH = r_BH_func(
            r,
            H,
            BREAST_HEIGHT.to("m").magnitude,
            params["eta_B"].magnitude,
            params["eta_C"].magnitude,
        )
        res = dbh_m - 2 * r_BH

        #        # add a penalty if r > dbh to prevent stupid behavior for small dbh's
        #        penalty = np.maximum(0, r-dbh_m)
        return res

    res = root(g, r0)
    if (not res.success) and (np.abs(res.fun) > 1e-08):  # cm
        msg = "trunk base radius computation did not converge: "
        msg += "dbh = {} cm too small?".format(dbh)
        raise ValueError(msg)

    r = float(res.x)
    return r


# @lru_cache(maxsize=50_000) # caching does not seem to speed up
# @make_lru_cacheable(maxsize=100)
def dbh_from_r(r: float, species: str, custom_species_params: SpeciesParams) -> float:
    """
    Compute dbh for :class:`~.single_tree_allocation.SingleTreeAllocation` for given trunk base rasius r [cm].

    Args:
        r: radius at trunk base to match [m]
        species: element of ["pine", "spruce", "birch"]
        custom_species_params: tree species parameters (for all species)
    """
    assert isinstance(r, float)
    if r == 0:
        return 0.0

    # only for finding initial C for transient pool E
    if r < (2 + 1.25 * 1.0) / 2 / 100:
        # r too small
        return -5 * r

    if r < r_from_dbh(3.0, species, species_params):
        # r small, we use
        # Laasasenaho, 1982; Repola, 2009, p. 631
        # from dbh to r at trunk base
        f = interp1d(
            [(2 + 1.25 * 1.0) / 2 / 100, r_from_dbh(3.0, species, species_params)],
            [1.0, 3.0],
        )
        return float(f(r))

    # initialize a tree to use its computational facilities
    dbh0_m = 2 * r

    # set up a root solve algorithm for trunk base radius r
    def g(dbh_cm):
        if dbh_cm < 0:
            return -10 * dbh_cm
        res = r - r_from_dbh(float(dbh_cm), species, custom_species_params)
        return res

    root_res = root(g, dbh0_m * 100)
    if (not root_res.success) or (root_res.x < 0):  # and (root_res.fun > 1e-05): # cm
        msg = "dbh computation did not converge: "
        msg += "r = {} m too small?".format(r)
        # whatever works is fine...
        #        return float(res.x)
        raise ValueError(msg)

    return float(root_res.x)  # cm


# @make_lru_cacheable(maxsize=100)
def solve_for_SW(
    r_at_trunk_base: float,  # m
    dbh: float,  # cm
    H: float,  # m
    tree_age: float,  # yr
    rho_TS: float,  # kg/m^3
    rho_T: float,  # kg/m^3
    species: str,
    custom_species_params: SpeciesParams,
) -> float:  # m
    """Compute sapwood width such that SW/HW follows Sellin [m].

    Args:
        r_at_trunk_base: radius at trunk base [m]
        dbh: diameter at breast height [cm]
        H: tree height [m]
        tree_age: age of tree [yr]
        rho_TS: sapwood density [kg m-3]
        rho_T: heartwood density [kg m-3]
        species: element of ["pine", "spruce"]
        custom_species_params: tree species parameters (for all species)

    Notes:
        Source: [6]_, [7]_
    """
    # TODO
    # currently, the spruce literature seems to be the best,
    # the pine (Vanninnen) does not work well
    species = "spruce"

    params = initialize_params(custom_species_params[species])

    #    if species == "pine":
    #        # compute Vanninnen prediction for tree sapwood biomass fraction
    #        constant = params["SW_constant"].magnitude
    #        c_H = params["SW_H"].magnitude
    #        c_a = params["SW_A"].magnitude
    #        Wss_frac = np.minimum(1.0, constant + c_H * H * 100 + c_a * tree_age)
    #
    #        def g(SW_m):  # m
    #            V_T = V_T_func(dbh, H, species, custom_species_params)
    #            V_TH = V_TH_func(SW_m, dbh, H, species, custom_species_params)
    #            V_TS = V_T - V_TH
    #
    #            return V_TS * rho_TS / (V_T * rho_T) - Wss_frac
    #
    #        # it seems like an adaptation only happens after the
    #        # initial value for the root search, 0.05 seems too late
    #        #        root_res = root(g, 0.05)
    #        root_res = root(g, 0.01)  # m
    #
    #        if not root_res.success:
    #            # possible creativity here
    #            pass
    #
    #        return float(root_res.x)
    #
    #    elif species == "spruce":

    if species == "spruce":
        r = r_at_trunk_base * 100  # from m to cm
        d = r * 2

        # Sellin 1994, Fig. 1 and Eq. [2]
        SW_Sellin = params["SW_a"] * d / (d + params["SW_b"])
        HW_Sellin = params["HW_slope"] * d  # slope from Fig. 1

        # use Sellin SW-HW proportions for scaling to current tree radius
        SW = r * SW_Sellin / (SW_Sellin + HW_Sellin)
        #        HW = r * HW_Sellin / (SW_Sellin + HW_Sellin)
        return (SW / 100).magnitude  # from cm to m

    else:
        raise ValueError(f"No sapwood width computation found for {species}")
