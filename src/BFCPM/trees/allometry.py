"""Marklund, Repola, and Lehtonen allometric functions for tree organ biomass.

Notes:
    .. [1] Marklund L.G. (1988), "Biomassafunktioner for tall, gran
        och björk i Sverige", SLU Rapport 45, 73p

    .. [2] Kärkkäinen L., 2005 Appendix 4.

    .. [3] Kellomäki et al. (2001). Atm. Env.

    .. [4] Repola (2008) Silva Fennica

    .. [5] Repola (2009) Silva Fenica

    .. [6] Repola (2014) Silva Fennica

    .. [7] Lehtonen (2005) Tree Phys.

    Authors:
        - Samuli Launiainen, 25.04.2014
        - Holger Metzler, 01.06.2021
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from ..type_aliases import SpeciesParams
from . import tree_utils
from .single_tree_params import initialize_params

params_allometry: Dict[str, Dict[str, Any]] = {
    "Marklund": {
        "leaves": {
            "pine": {"a": 7.7681, "b": 7, "c": 3.7983},
            "spruce": {"a": 7.8171, "b": 12, "c": 1.9602},
            "birch": {
                "a": 8.058,
                "b": 8,
                "c": 3.9823,
            },  # Kellomäki et al. 2001 Atm. Env.
        },
        "stemwood": {
            "pine": {"a": 11.4219, "b": 14, "c": 2.2184},
            "spruce": {"a": 11.4873, "b": 14, "c": 2.2471},
            "birch": {"a": 10.8109, "b": 11, "c": 2.3327},
        },
        "stembark": {
            "pine": {"a": 8.8489, "b": 16, "c": 2.9748},
            "spruce": {"a": 9.8364, "b": 15, "c": 3.3912},
            "birch": {"a": 10.3876, "b": 14, "c": 3.2518},
        },
        "livingbranches": {
            "pine": {"a": 9.1015, "b": 10, "c": 2.8604},  # including needles
            "spruce": {"a": 8.5242, "b": 13, "c": 1.2804},  # including needles
            "birch": {"a": 10.2806, "b": 10, "c": 3.3633},  # excluding leaves
        },
        "stump": {
            "pine": {"a": 11.0481, "b": 15, "c": 3.9657},
            "spruce": {"a": 10.6686, "b": 17, "c": 3.3645},
        },
        "roots": {
            "pine": {
                "a1": 13.2902,
                "b1": 9,
                "c1": 6.3413,
                "a2": 8.8795,
                "b2": 10,
                "c2": 3.8375,
            },
            "spruce": {
                "a1": 13.3703,
                "b1": 8,
                "c1": 6.3851,
                "a2": 7.6283,
                "b2": 12,
                "c2": 2.5706,
            },
        },
    },
    "Repola": {
        "dbh_mean": {"pine": 13.1, "spruce": 17.9},
        "dbh_sd": {"pine": 5.3, "spruce": 7.2},
        "leaves": {
            "pine": {
                "b0": -6.303,
                "b1": 14.472,
                "b2": -3.976,
                #                "alpha": 6,
                #                "beta": 1,
                "n": 6,
                "m": 1,
                "var_u": 0.109,
                "var_e": 0.118,
            },  # Repola (2009)
            "spruce": {
                "b0": -2.994,
                "b1": 12.251,
                "b2": -3.415,
                #                "alpha": 10,
                #                "beta": 1,
                "n": 10,
                "m": 1,
                "var_u": 0.107,
                "var_e": 0.089,
            },  # Repola (2009)
            "birch": {
                "b0": -29.556,
                "b1": 33.372,
                "b2": 0,
                "alpha": 2,
                "beta": 0,
                "var_u": 0.000,
                "var_e": 0.077,
            },  # Repola (2008)
        },
        "stemwood": {
            "pine": {
                "b0": -3.721,
                "b1": 8.103,
                "b2": 5.066,
                #                "alpha": 14,
                #                "beta": 12,
                "n": 14,
                "m": 12,
                "var_u": 0.002,
                "var_e": 0.009,
            },  # Repola (2009)
            "spruce": {
                "b0": -3.555,
                "b1": 8.042,
                #                "b2": 0.869,
                "b3": 0.869,
                #                "b3": 0.015,
                "b4": 0.015,
                #                "alpha": 14,
                "n": 14,
                "var_u": 0.009,
                "var_e": 0.009,
            },  # Repola (2009)
            "birch": {
                "b0": -4.879,
                "b1": 9.651,
                "b2": 1.012,
                "alpha": 12,
                "var_u": 0.00263,
                "var_e": 0.00544,
            },  # Repola (2008)
        },
        "stembark": {
            "pine": {
                "b0": -4.548,
                "b1": 7.997,
                #                "b2": 0.357,
                "b3": 0.357,
                #                "alpha": 12,
                "n": 12,
                "var_u": 0.015,
                "var_e": 0.061,
            },  # Repola (2009)
            "spruce": {
                "b0": -4.548,
                "b1": 9.448,
                #                "b2": 0.436,
                "b3": 0.436,
                #                "alpha": 18,
                "n": 18,
                "var_u": 0.023,
                "var_e": 0.041,
            },  # Repola (2009)
            "birch": {
                "b0": -5.401,
                "b1": 10.061,
                "b2": 2.651,
                "alpha": 12,
                "beta": 20,
                "var_u": 0.01043,
                "var_e": 0.04443,
            },  # Repola (2008)
        },
        "livingbranches": {
            "pine": {
                "b0": -6.162,
                "b1": 15.075,
                "b2": -2.618,
                #                "alpha": 12,
                #                "beta": 12,
                "n": 12,
                "m": 12,
                "var_u": 0.041,
                "var_e": 0.089,
            },  # Repola (2009)
            "spruce": {
                "b0": -4.214,
                "b1": 14.508,
                "b2": -3.277,
                #                "alpha": 13,
                #                "beta": 5,
                "n": 13,
                "m": 5,
                "var_u": 0.039,
                "var_e": 0.081,
            },  # Repola (2009)
            "birch": {
                "b0": -4.152,
                "b1": 15.874,
                "b2": -4.407,
                "alpha": 16,
                "beta": 10,
                "var_u": 0.02733,
                "var_e": 0.07662,
            },  # Repola (2008)
        },
        "stump": {
            "pine": {
                "b0": -6.753,
                "b1": 12.681,
                #                "alpha": 12,
                "n": 12,
                "var_u": 0.010,
                "var_e": 0.044,
            },  # Repola (2009)
            "spruce": {
                "b0": -3.964,
                "b1": 11.730,
                #                "alpha": 26,
                "n": 26,
                "var_u": 0.065,
                "var_e": 0.058,
            },  # Repola (2009)
            "birch": {
                "b0": -3.574,
                "b1": 11.304,
                "alpha": 26,
                "var_u": 0.02154,
                "var_e": 0.04542,
            },  # Repola (2008)
        },
        "roots": {
            "pine": {
                "b0": -5.550,
                "b1": 13.408,
                #                "alpha": 15,
                "n": 15,
                "var_u": 0.000,
                "var_e": 0.079,
            },  # Repola (2009)
            "spruce": {
                "b0": -2.294,
                "b1": 10.646,
                #                "alpha": 24,
                "n": 24,
                "var_u": 0.105,
                "var_e": 0.114,
            },  # Repola (2009)
            "birch": {
                "b0": -3.223,
                "b1": 6.497,
                "b2": 1.033,
                "alpha": 22,
                "var_u": 0.0480,
                "var_e": 0.02677,
            },  # Repola (2008)
        },
    },
}
"""Marklund ([1]_) and Repola 2009 ([5]_) allometry parameters."""


params_young_allometry = {
    "Repola": {
        # stembark, stump, roots are missing, taking from old function
        "leaves": {
            "pine": {
                "b0": 1.850,
                "b1": 12.504,
                "b2": -10.624,
                "b3": 0.0,
                "n": 5,
                "m": 1,
                "var_u": 0.023,
                "var_e": 0.105,
            },
            "spruce": {
                "b0": -2.924,
                "b1": 8.464,
                "b2": 0.0,
                "b3": 0.0,
                "n": 8,
                "m": 0,
                "var_u": 0.027,
                "var_e": 0.113,
            },
            "birch": {
                "b0": -5.551,
                "b1": 8.584,
                "b2": 0.0,
                "b3": 0.0,
                "n": 5,
                "m": 0,
                "var_u": 0.120,
                "var_e": 0.126,
            },
        },
        "stemwood": {
            "pine": {
                "b0": -3.010,
                "b1": 7.999,
                "b2": 2.959,
                "b3": 0.0,
                "n": 8,
                "m": 13,
                "var_u": 0.003,
                "var_e": 0.018,
            },
            "spruce": {
                "b0": -2.669,
                "b1": 10.486,
                "b2": 0.0,
                "b3": 0.0,
                "n": 10,
                "m": 0,
                "var_u": 0.010,
                "var_e": 0.020,
            },
            "birch": {
                "b0": -4.879,
                "b1": 8.037,
                "b2": 22.614,
                "b3": -4.003,
                "n": 2,
                "m": 12,
                "var_u": 0.001,
                "var_e": 0.016,
            },
        },
        "livingbranches": {
            "pine": {
                "b0": -2.949,
                "b1": 15.444,
                "b2": -6.350,
                "b3": 0.0,
                "n": 9,
                "m": 4,
                "var_u": 0.005,
                "var_e": 0.131,
            },
            "spruce": {
                "b0": -3.390,
                "b1": 12.228,
                "b2": -3.834,
                "b3": 0.0,
                "n": 8,
                "m": 8,
                "var_u": 0.059,
                "var_e": 0.093,
            },
            "birch": {
                "b0": -4.190,
                "b1": 11.826,
                "b2": -3.393,
                "b3": 0.0,
                "n": 6,
                "m": 10,
                "var_u": 0.011,
                "var_e": 0.105,
            },
        },
    }
}
"""Repola allometry parameters for young trees (Repola2014SF, [6]_)."""


def _mleaf_from_dbh_marklund(
    dbh: float, species: str, custom_species_params: SpeciesParams
) -> float:
    if dbh == 0:
        return 0

    d = dbh
    p = params_allometry["Marklund"]["leaves"][species]
    a, b, c = p["a"], p["b"], p["c"]

    return np.exp(a * d / (d + b) - c)


def _mstemwood_from_dbh_marklund(
    dbh: float, species: str, custom_species_params: SpeciesParams
) -> float:
    if dbh == 0:
        return 0

    d = dbh
    p = params_allometry["Marklund"]["stemwood"][species]
    a, b, c = p["a"], p["b"], p["c"]

    return np.exp(a * d / (d + b) - c)


def _mstembark_from_dbh_marklund(
    dbh: float, species: str, custom_species_params: SpeciesParams
) -> float:
    if dbh == 0:
        return 0

    d = dbh
    p = params_allometry["Marklund"]["stembark"][species]
    a, b, c = p["a"], p["b"], p["c"]

    return np.exp(a * d / (d + b) - c)


def _mlivingbranches_from_dbh_marklund(
    dbh: float, species: str, custom_species_params: SpeciesParams
) -> float:
    if dbh == 0:
        return 0

    d = dbh
    p = params_allometry["Marklund"]["livingbranches"][species]
    a, b, c = p["a"], p["b"], p["c"]

    if species in ["pine", "spruce"]:
        return np.exp(a * d / (d + b) - c) - _mleaf_from_dbh_marklund(
            dbh, species, custom_species_params
        )
    elif species == "birch":
        return np.exp(a * d / (d + b) - c)
    else:
        raise ValueError("Unknown species")


def _mstump_from_dbh_marklund(
    dbh: float, species: str, custom_species_params: SpeciesParams
) -> float:
    if dbh == 0:
        return 0

    d = dbh
    p = params_allometry["Marklund"]["stump"][species]
    a, b, c = p["a"], p["b"], p["c"]

    return np.exp(a * d / (d + b) - c)


def _mroots_from_dbh_marklund(
    dbh: float, species: str, custom_species_params: SpeciesParams
) -> float:
    if dbh == 0:
        return 0

    d = dbh
    p = params_allometry["Marklund"]["roots"][species]
    a1, b1, c1 = p["a1"], p["b1"], p["c1"]
    a2, b2, c2 = p["a2"], p["b2"], p["c2"]

    return np.exp(a1 * d / (d + b1) - c1) + np.exp(a2 * d / (d + b2) - c2)


def _mleaf_from_dbh_and_H_repola(
    dbh: float, H: float, species: str, custom_species_params: SpeciesParams
) -> float:
    """Compute leaf biomass from diameter at breast height and height

    Compute tree biomasses for Scots pine, Norway spruce or birch
    based on Repola (2009, birch: 2008).

    Args:
        dbh: diamter at breast height [cm]
        H: tree height [m]
        species: element of ["pine", "spruce", "birch"]
        custom_species_params: tree species parameters (for all species)

    Returns:
        leaf biomass [kg_dw]
    """
    if dbh == 0:
        return 0

    params = initialize_params(custom_species_params[species])
    #    S_L_per_yr = params["S_L"].to("1/yr").magnitude

    d = 2 + 1.25 * dbh
    p = params_allometry["Repola"]["leaves"][species]

    b0, b1, b2 = p["b0"], p["b1"], p["b2"]
    n, m = p["n"], p["m"]

    var_u, var_e = p["var_u"], p["var_e"]
    intercept = b0 + (var_u + var_e) / 2

    #    # correct for leaf measurements done in winter
    #    if species == "pine":
    #        q = 1 / (1 - S_L_per_yr)
    #    elif species == "spruce":
    #        q = 1 / (1 - S_L_per_yr)
    #
    #    return q * np.exp(intercept + b1 * d / (d + n) + b2 * H / (H + m))
    return np.exp(intercept + b1 * d / (d + n) + b2 * H / (H + m))


def _mstemwood_from_dbh_and_H_repola(
    dbh: float, H: float, species: str, custom_species_params: SpeciesParams
) -> float:
    if dbh == 0:
        return 0

    d = 2 + 1.25 * dbh
    p = params_allometry["Repola"]["stemwood"][species]

    var_u, var_e = p["var_u"], p["var_e"]
    intercept_correction = (var_u + var_e) / 2

    if species == "pine":
        b0, b1, b2 = p["b0"], p["b1"], p["b2"]
        #        alpha, beta = p["alpha"], p["beta"]
        n, m = p["n"], p["m"]
        intercept = b0 + intercept_correction
        #        return np.exp(intercept + b1 * d / (d + alpha) + b2 * H / (H + beta))
        return np.exp(intercept + b1 * d / (d + n) + b2 * H / (H + m))

    elif species == "spruce":
        #        b0, b1, b2, b3 = p["b0"], p["b1"], p["b2"], p["b3"]
        #        alpha = p["alpha"]
        b0, b1, b3, b4 = p["b0"], p["b1"], p["b3"], p["b4"]
        n = p["n"]
        intercept = b0 + intercept_correction
        #        return np.exp(intercept + b1 * d / (d + alpha) + b3 * H + b2 * np.log(H))
        return np.exp(intercept + b1 * d / (d + n) + b3 * np.log(H)) + b4 * H

    #    elif species == "birch":
    #        b0, b1, b2 = p["b0"], p["b1"], p["b2"]
    #        alpha = p["alpha"]
    #        intercept = b0 + intercept_correction
    #        return np.exp(intercept + b1 * d / (d + alpha) + b2 * np.log(H))

    else:
        raise ValueError("Unknwon species")


def _mstembark_from_dbh_and_H_repola(
    dbh: float, H: float, species: str, custom_species_params: SpeciesParams
) -> float:
    if dbh == 0:
        return 0

    d = 2 + 1.25 * dbh
    p = params_allometry["Repola"]["stembark"][species]

    var_u, var_e = p["var_u"], p["var_e"]
    intercept_correction = (var_u + var_e) / 2

    if species in ["pine", "spruce"]:
        #        b0, b1, b2 = p["b0"], p["b1"], p["b2"]
        b0, b1, b3 = p["b0"], p["b1"], p["b3"]
        #        alpha = p["alpha"]
        n = p["n"]
        intercept = b0 + intercept_correction
        #        return np.exp(intercept + b1 * d / (d + alpha) + b2 * np.log(H))
        return np.exp(intercept + b1 * d / (d + n) + b3 * np.log(H))

    #    elif species == "birch":
    #        b0, b1, b2 = p["b0"], p["b1"], p["b2"]
    #        alpha, beta = p["alpha"], p["beta"]
    #        intercept = b0 + intercept_correction
    #        return np.exp(intercept + b1 * d / (d + alpha) + b2 * H / (H + beta))

    else:
        raise ValueError("Unknwon species")


def _mlivingbranches_from_dbh_and_H_repola(
    dbh: float, H: float, species: str, custom_species_params: SpeciesParams
) -> float:
    if dbh == 0:
        return 0

    d = 2 + 1.25 * dbh

    p = params_allometry["Repola"]["livingbranches"][species]
    var_u, var_e = p["var_u"], p["var_e"]
    intercept_correction = (var_u + var_e) / 2

    if species in ["pine", "spruce"]:
        b0, b1, b2 = p["b0"], p["b1"], p["b2"]
        #        alpha, beta = p["alpha"], p["beta"]
        n, m = p["n"], p["m"]
        intercept = b0 + intercept_correction
        #        return np.exp(intercept + b1 * d / (d + alpha) + b2 * H / (H + beta))
        return np.exp(intercept + b1 * d / (d + n) + b2 * H / (H + m))

    #    elif species == "birch":
    #        b0, b1, b2 = p["b0"], p["b1"], p["b2"]
    #        alpha, beta = p["alpha"], p["beta"]
    #        intercept = b0 + intercept_correction
    #        return np.exp(intercept + b1 * d / (d + alpha) + b2 * H / (H + beta))

    else:
        raise ValueError("Unknwon species")


def _mstump_from_dbh_and_H_repola(
    dbh: float, H: float, species: str, custom_species_params: SpeciesParams
) -> float:
    if dbh == 0:
        return 0

    d = 2 + 1.25 * dbh

    p = params_allometry["Repola"]["stump"][species]
    var_u, var_e = p["var_u"], p["var_e"]
    intercept_correction = (var_u + var_e) / 2

    if species in ["pine", "spruce"]:
        b0, b1 = p["b0"], p["b1"]
        #        alpha = p["alpha"]
        n = p["n"]
        intercept = b0 + intercept_correction
        #        return np.exp(intercept + b1 * d / (d + alpha))
        return np.exp(intercept + b1 * d / (d + n))

    #    elif species == "birch":
    #        b0, b1 = p["b0"], p["b1"]
    #        alpha = p["alpha"]
    #        intercept = b0 + intercept_correction
    #        return np.exp(intercept + b1 * d / (d + alpha))

    else:
        raise ValueError("Unknwon species")


def _mroots_from_dbh_and_H_repola(
    dbh: float, H: float, species: str, custom_species_params: SpeciesParams
) -> float:
    if dbh == 0:
        return 0

    d = 2 + 1.25 * dbh

    p = params_allometry["Repola"]["roots"][species]
    var_u, var_e = p["var_u"], p["var_e"]
    intercept_correction = (var_u + var_e) / 2

    if species in ["pine", "spruce"]:
        b0, b1 = p["b0"], p["b1"]
        #        alpha = p["alpha"]
        n = p["n"]
        intercept = b0 + intercept_correction
        #        return np.exp(intercept + b1 * d / (d + alpha))
        return np.exp(intercept + b1 * d / (d + n))

    #    elif species == "birch":
    #        b0, b1, b2 = p["b0"], p["b1"], p["b2"]
    #        alpha = p["alpha"]
    #        intercept = b0 + intercept_correction
    #        return np.exp(intercept + b1 * d / (d + alpha) + b2 * np.log(H))

    else:
        raise ValueError("Unknown species")


def _mleaf_from_dbh_and_H_lehtonen(
    dbh: float, H: float, species: str, custom_species_params: SpeciesParams
) -> float:
    params = {
        "pine": {"a": 0.1179, "b": 2.1052, "c": -0.7931},
        "spruce": {"a": 0.1002, "b": 2.5947, "c": -0.8647},
    }

    p = params[species]
    a, b, c = p["a"], p["b"], p["c"]

    return a * dbh**b * H**c


def _young_allometry_repola(
    dbh: float,
    H: float,
    species: str,
    component: str,
    custom_species_params: SpeciesParams,
) -> float:
    # leaves allometry here superseded by Lehtonen
    if component == "leaves":
        return _mleaf_from_dbh_and_H_lehtonen(dbh, H, species, custom_species_params)

    # some components are not provided for young trees,
    # we use the "old" function then
    if component in ["stembark", "stump", "roots"]:
        return funcs_allometry["Repola"][component](
            dbh, H, species, custom_species_params
        )

    p = params_young_allometry["Repola"][component][species]
    b0, b1, b2, b3 = p["b0"], p["b1"], p["b2"], p["b3"]
    n, m = p["n"], p["m"]
    var_u, var_e = p["var_u"], p["var_e"]
    intercept_correction = (var_u + var_e) / 2
    intercept = b0 + intercept_correction

    f = np.exp(intercept + b1 * dbh / (dbh + n) + b2 * H / (H + m) + b3 * np.log(H))
    return f


funcs_allometry: Dict[str, Dict[str, Callable]] = {
    "Marklund": {
        "leaves": _mleaf_from_dbh_marklund,
        "stemwood": _mstemwood_from_dbh_marklund,
        "stembark": _mstembark_from_dbh_marklund,
        "livingbranches": _mlivingbranches_from_dbh_marklund,
        "stump": _mstump_from_dbh_marklund,
        "roots": _mroots_from_dbh_marklund,
    },
    "Repola": {
        "leaves": _mleaf_from_dbh_and_H_repola,
        "stemwood": _mstemwood_from_dbh_and_H_repola,
        "stembark": _mstembark_from_dbh_and_H_repola,
        "livingbranches": _mlivingbranches_from_dbh_and_H_repola,
        "stump": _mstump_from_dbh_and_H_repola,
        "roots": _mroots_from_dbh_and_H_repola,
    },
}


def allometries_marklund(
    dbh: float,
    H: float,
    species: str,
    components: Tuple[str],
    custom_species_params: SpeciesParams,
):
    """Compute tree component biomasses.

    Compute tree biomasses for Scots pine, Norway spruce or birch
    based on [1]_. Stump and root biomass is not available in [1]_
    and is taken from [4]_.

    Args:
        dbh: diamter at breast height [cm]
        H: tree height [m]
        species: element of ["pine", "spruce", "birch"]
        components: tuple with elements from
            ("leaves", "stemwood", "stembark", "livingbranches", "stump", "roots")
        custom_species_params: tree species parameters (for all species)

    Returns:
        summed biomass [kg_dw] over the components

    Notes:
        Source: [1]_, [2]_, [3]_, [4]_
    """
    mass_kg_dw = 0.0
    for c in components:
        # birch stump and roots missing in Marklund, take from Repola
        if (species == "birch") and (c in ["stump", "roots"]):
            mass_kg_dw += funcs_allometry["Repola"][c](
                dbh, H, species, custom_species_params
            )
        else:
            mass_kg_dw += funcs_allometry["Marklund"][c](
                dbh, species, custom_species_params
            )

    return mass_kg_dw


def allometries_repola_and_lehtonen(
    dbh: float,
    H: float,
    species: str,
    components: List[str],
    custom_species_params: SpeciesParams,
    x1: float = None,
):
    """Compute tree component biomasses following Repola and Lehtonen.

    Compute tree biomasses for Scots pine, Norway spruce
    ([6]_ if `dbh < x1` else [5]_).
    For leaves use [7]_ instead of [6]_.
    Normalize the young curve such that it meets the old curve.

    Args:
        dbh: diamter at breast height [cm]
        H: tree height [m]
        species: element of ["pine", "spruce", "birch"]
        components: tuple with elements from
            ("leaves", "stemwood", "stembark", "livingbranches", "stump", "roots")
        custom_species_params: tree species parameters (for all species)
        x1: dbh to switch from [6_] or [7]_ to [5]_, if ``None`` use param values: `dbh_mean - dbh_sd` ([5]_, Table 3)

    Returns:
        summed biomass [kg_dw] over the components

    Notes:
        Source: [5]_, [6]_, [7]_
    """
    if x1 is None:
        dbh_mean = params_allometry["Repola"]["dbh_mean"][species]
        dbh_sd = params_allometry["Repola"]["dbh_sd"][species]
        x1 = dbh_mean - dbh_sd

    H1 = tree_utils.H_func(x1, species)
    if dbh < x1:
        # young trees
        ys = list()
        for c in components:
            # normalize such that young and old curves meet
            y1 = funcs_allometry["Repola"][c](
                x1, H1, species, custom_species_params
            )  # call old branch (see below)
            y1_young = _young_allometry_repola(
                x1, H1, species, c, custom_species_params
            )
            y = _young_allometry_repola(dbh, H, species, c, custom_species_params)
            norm_factor = y1 / y1_young

            y_normalized = y * norm_factor

            y = y_normalized
            ys.append(y_normalized)

        return sum(ys)
    else:
        # older trees
        ys = list()
        for c in components:
            y = funcs_allometry["Repola"][c](dbh, H, species, custom_species_params)
            if x1 > 0:
                y1 = funcs_allometry["Repola"][c](
                    x1, H1, species, custom_species_params
                )
                y1_young = _young_allometry_repola(
                    x1, H1, species, c, custom_species_params
                )
                norm_factor = y1_young / y1
            else:
                norm_factor = 1.0

            y_normalized = y  # * norm_factor
            ys.append(y_normalized)

        return sum(ys)


def allometries_repola_2009(
    dbh: float,
    H: float,
    species: str,
    components: Tuple[str],
    custom_species_params: SpeciesParams,
):
    """Compute tree component biomasses following Repola (2009).

    Args:
        dbh: diamter at breast height [cm]
        H: tree height [m]
        species: element of ["pine", "spruce", "birch"]
        components: tuple with elements from
            ("leaves", "stemwood", "stembark", "livingbranches", "stump", "roots")
        custom_species_params: tree species parameters (for all species)

    Returns:
        summed biomass [kg_dw] over the components

    Notes:
        Source: [5]_
    """
    # older trees
    ys = [
        funcs_allometry["Repola"][c](dbh, H, species, custom_species_params)
        for c in components
    ]
    return sum(ys)


def allometries_repola_2014(
    dbh: float,
    H: float,
    species: str,
    components: Tuple[str],
    custom_species_params: SpeciesParams,
):
    """Compute tree component biomasses following Repola (2014).

    Args:
        dbh: diamter at breast height [cm]
        H: tree height [m]
        species: element of ["pine", "spruce", "birch"]
        components: tuple with elements from
            ("leaves", "stemwood", "stembark", "livingbranches", "stump", "roots")
        custom_species_params: tree species parameters (for all species)

    Returns:
        summed biomass [kg_dw] over the components

    Notes:
        Source: [6]_
    """
    # young trees
    ys = [
        _young_allometry_repola(dbh, H, species, c, custom_species_params)
        for c in components
    ]
    return sum(ys)
