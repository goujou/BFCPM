"""Overarching simulation parameters.

Also the interfaces between the model modules and the tree-cutting matrix.
"""
from __future__ import annotations

from typing import Dict, Tuple, Type

import numpy as np

from . import DATA_PATH, Q_
from .params import BREAST_HEIGHT, soil_p, water_p
from .productivity.vegetation import taper_curve
from .soil.dead_wood_classes.C_model import SoilCDeadWoodClasses
from .soil.simple_soil_model.C_model import SimpleSoilCModel
from .soil.soil_c_model_abc import SoilCModelABC
from .trees.single_tree_allocation import SingleTree
from .type_aliases import TreeSoilInterface, WoodProductInterface
from .wood_products.simple_wood_product_model.C_model import \
    SimpleWoodProductModel
from .wood_products.wood_product_model_abc import WoodProductModelABC

# a species dependency could be included here
tree_soil_interfaces: Dict[
    Tuple[Type[SingleTree], Type[SoilCModelABC]], TreeSoilInterface
] = {
    (SingleTree, SimpleSoilCModel): {
        # pool_from: {pool_to_1: proportion_1, pool_to_2: proportion_2, ...}
        # proportions musst add up to 1.0
        # pool_froms not mentioned have only external output
        "B_L": {"Litter": 1.0},  # leaf litter to Litter
        "B_R": {"Litter": 1.0},  # fine roots litter to Litter
        "B_OS": {"CWD": 1.0},  # sapwood litter to CWD
        "B_OH": {"CWD": 1.0},  # heartwood litter to CWD
    },
    (SingleTree, SoilCDeadWoodClasses): {
        # pool_from: {pool_to_1: proportion_1, pool_to_2: proportion_2, ...}
        # proportions musst add up to 1.0
        # pool_froms not mentioned have only external output
        "B_L": {"Litter": 1.0},  # leaf litter to Litter
        "B_R": {"Litter": 1.0},  # fine roots litter to Litter
        "B_OS": {"DWC_1": 1.0},  # sapwood litter to CWD
        "B_OH": {"DWC_1": 1.0},  # heartwood litter to CWD
    },
}
"""Interface between tree and soil."""
# Depending on the tree model and the soil model, tree pool contents
# are distributed to soil pools due to senescence, where fractions
# must sum to one for each listed tree pool. External outputs from tree
# pools not mentioned go completely to the atmosphere.
# """


def solve_volume_fractions(
    d: float,
    h: float,
    species: str,
    #    dlog: float = 18.0, dfibre: float = 8.0, llog: float = 4.0,
    dlog: float = 16.0,
    dfibre: float = 8.0,
    llog: float = 4.0,
    lfibre: float = 3.0,
    lstump: float = 0.2,
    dz: float = 0.01,
) -> Dict[str, float]:
    """
    Approximates volumetric partitioning of a tree to sawlogs, fibre and cutting residues.

    Args:
        d: diameter at breast height [cm]
        h: tree height [m]
        species: element of  ``['pine', 'spruce', 'birch']``
        dlog: minimum log diameter [cm]
        dfibre: minimim fibre diameter [cm]
        llog: minimim log length [m]
        lfibre: minimim fibre length [m]
        lstump: stump height [m]
        dz: resolution of taper curve [m]

    Returns:
        dictionary

        - V_stem: stem volume including stump [m3]
        - f_log, f_fibre, f_resid, f_stump: volume fractions [-]
    """

    def cross_sect_area(d):
        """Cross-sectional area (m2) at height ``d``."""
        return np.pi * (1e-2 * d / 2.0) ** 2

    hs, ds = taper_curve(
        h, d, species, L=BREAST_HEIGHT.to("m").magnitude, dz=dz
    )  # taper curve
    A = cross_sect_area(ds)
    Vtot = np.trapz(A, dx=dz)  # volume including stump
    #    print(d, hs[-1], ds[-1], A[-1], Vtot)

    # remove stump
    ixs = (np.abs(hs - lstump)).argmin()
    hs = hs[ixs:]
    ds = ds[ixs:]

    ixl = (np.abs(ds - dlog)).argmin()
    ixf = (np.abs(ds - dfibre)).argmin()

    A = cross_sect_area(ds)  # m2
    Vtot_no_stump = np.trapz(A, dx=dz)  # m3, volume without stump

    Vstump = Vtot - Vtot_no_stump

    #    print(ixl, hs[ixl], llog + lstump)
    if (ixl > 0) & (hs[ixl] > llog + lstump):
        Vlog = np.trapz(A[ixs:ixl], dx=dz)
    else:
        Vlog = 0.0
        llog = 0.0

    if (ixf > 0) & (hs[ixf] > lstump + llog + lfibre):
        Vfibre = np.trapz(A[ixl:ixf], dx=dz)
    else:
        Vfibre = 0.0

    Vres = Vtot_no_stump - Vlog - Vfibre

    vf_dict: Dict[str, float] = {
        "V_stem": Vtot,
        "log": Vlog / Vtot,
        "fibre": Vfibre / Vtot,
        "residue": Vres / Vtot,
        "stump": Vstump / Vtot,
    }

    return vf_dict


def trunk_to_dead_wood_classes_no_harvesting(
    d: float, h: float, species: str
) -> Dict[str, float]:
    """Return fractions of the trunk's C fate."""
    vf_dict = solve_volume_fractions(h, d, species)

    result = {
        "DWC_1": vf_dict["residue"],
        "DWC_2": vf_dict["fibre"],
        "DWC_3": 0,
        "DWC_4": 0,
        "DWC_5": 0,
        "DWC_6": 0,
    }

    bole_dbhs = np.array([13, 25, 50, 80, 150])
    idx = (np.abs(bole_dbhs - d)).argmin()
    result[f"DWC_{idx+2}"] += vf_dict["log"] + vf_dict["stump"]

    return result


def trunk_to_dead_wood_classes_default(
    d: float, h: float, species: str
) -> Dict[str, float]:
    """Return fractions of the trunk's C fate."""
    vf_dict = solve_volume_fractions(h, d, species)

    result = {
        "DWC_1": 0,
        "DWC_2": 0,
        "DWC_3": 0,
        "DWC_4": 0,
        "DWC_5": 0,
        "DWC_6": 0,
        # done then by the interface
        "residue": vf_dict["residue"],
        "fibre": vf_dict["fibre"],
        "log": vf_dict["log"],
    }

    # put the stump in the right DWC
    bole_dbhs = np.array([13, 25, 50, 80, 150])
    idx = (np.abs(bole_dbhs - d)).argmin()
    result[f"DWC_{idx+2}"] += vf_dict["stump"]

    return result


# a species dependency could be included here
wood_product_interfaces: Dict[
    Tuple[Type[SingleTree], Type[SoilCModelABC], Type[WoodProductModelABC]],
    Dict[str, WoodProductInterface],
] = {
    (SingleTree, SimpleSoilCModel, SimpleWoodProductModel): {
        "default": {
            # pool_from: {pool_to_1: proportion_1, pool_to_2: proportion_2, ...}
            # proportions musst add up to 1.0
            "B_L": {"Litter": 1.0},  # leaf litter to Litter
            "C_L": {"Litter": 1.0},
            "B_R": {"Litter": 1.0},  # fine roots litter to Litter
            "C_R": {"Litter": 1.0},
            # parts of C_S belonging to the stem to Wood-products
            # --> slightly different structure
            "C_S": {
                "other": {"CWD": 1.0},
                "trunk": {
                    "log": "WP_L",
                    "fibre": "WP_S",
                    "residue": "CWD",
                    "stump": "CWD",
                },
            },
            "B_OS": {"CWD": 1.0},  # sapwood litter to CWD
            "B_OH": {"CWD": 1.0},  # heartwood litter to CWD
            "B_TS": {"log": "WP_L", "fibre": "WP_S", "residue": "CWD", "stump": "CWD"},
            "B_TH": {"log": "WP_L", "fibre": "WP_S", "residue": "CWD", "stump": "CWD"},
            # what to do with it, it's a fictional pool
            # actually, it stores captured C artifically for one year
            "E": {"Litter": 1.0},
            # function to distribute trunk biomass to CWD and wood products
            "_trunk_fate_func": solve_volume_fractions,
        }
    },
    (SingleTree, SoilCDeadWoodClasses, SimpleWoodProductModel): {
        "default": {
            # pool_from: {pool_to_1: proportion_1, pool_to_2: proportion_2, ...}
            # proportions musst add up to 1.0
            "B_L": {"Litter": 1.0},  # leaf litter to Litter
            "C_L": {"Litter": 1.0},
            "B_R": {"Litter": 1.0},  # fine roots litter to Litter
            "C_R": {"Litter": 1.0},
            # parts of C_S belonging to the stem to Wood-products
            # --> slightly different structure
            "C_S": {
                "other": {"DWC_1": 1.0},
                "trunk": {
                    "DWC_1": "DWC_1",
                    "DWC_2": "DWC_2",
                    "DWC_3": "DWC_3",
                    "DWC_4": "DWC_4",
                    "DWC_5": "DWC_5",
                    "DWC_6": "DWC_6",
                    "residue": "DWC_1",
                    "fibre": "WP_S",
                    "log": "WP_L",
                },
            },
            "B_OS": {"DWC_1": 1.0},
            "B_OH": {"DWC_1": 1.0},
            "B_TS": {
                "DWC_1": "DWC_1",
                "DWC_2": "DWC_2",
                "DWC_3": "DWC_3",
                "DWC_4": "DWC_4",
                "DWC_5": "DWC_5",
                "DWC_6": "DWC_6",
                "residue": "DWC_1",
                "fibre": "WP_S",
                "log": "WP_L",
            },
            "B_TH": {
                "DWC_1": "DWC_1",
                "DWC_2": "DWC_2",
                "DWC_3": "DWC_3",
                "DWC_4": "DWC_4",
                "DWC_5": "DWC_5",
                "DWC_6": "DWC_6",
                "residue": "DWC_1",
                "fibre": "WP_S",
                "log": "WP_L",
            },
            # what to do with it, it's a fictional pool
            # actually, it stores captured C artifically for one year
            "E": {"Litter": 1.0},
            # function to distribute trunk biomass to CWD and wood products
            "_trunk_fate_func": trunk_to_dead_wood_classes_default,
        },
        "no_harvesting": {
            # pool_from: {pool_to_1: proportion_1, pool_to_2: proportion_2, ...}
            # proportions musst add up to 1.0
            "B_L": {"Litter": 1.0},  # leaf litter to Litter
            "C_L": {"Litter": 1.0},
            "B_R": {"Litter": 1.0},  # fine roots litter to Litter
            "C_R": {"Litter": 1.0},
            # parts of C_S belonging to the stem to Wood-products
            # --> slightly different structure
            "C_S": {
                "other": {"DWC_1": 1.0},
                "trunk": {
                    "DWC_1": "DWC_1",
                    "DWC_2": "DWC_2",
                    "DWC_3": "DWC_3",
                    "DWC_4": "DWC_4",
                    "DWC_5": "DWC_5",
                    "DWC_6": "DWC_6",
                },
            },
            "B_OS": {"DWC_1": 1.0},
            "B_OH": {"DWC_1": 1.0},
            "B_TS": {
                "DWC_1": "DWC_1",
                "DWC_2": "DWC_2",
                "DWC_3": "DWC_3",
                "DWC_4": "DWC_4",
                "DWC_5": "DWC_5",
                "DWC_6": "DWC_6",
            },
            "B_TH": {
                "DWC_1": "DWC_1",
                "DWC_2": "DWC_2",
                "DWC_3": "DWC_3",
                "DWC_4": "DWC_4",
                "DWC_5": "DWC_5",
                "DWC_6": "DWC_6",
            },
            # what to do with it, it's a fictional pool
            # actually, it stores captured C artifically for one year
            "E": {"Litter": 1.0},
            # function to distribute trunk biomass to CWD and wood products
            "_trunk_fate_func": trunk_to_dead_wood_classes_no_harvesting,
        },
    },
}
"""Interfaces between trees and soil + wood products.

This interface is used in case of thinning or cutting. All tree pools
need to be mentioned and their fractions must sum to 1.
"""


stand_params_library = {
    "default": {
        # data and forcing files
        #   INACTIVE "dpath": DATA_PATH.joinpath('forcing/FIHy_data_1997_2019.dat'),
        #   PART OF simulation_params "fpath": DATA_PATH.joinpath('forcing/FIHy_forcing_1997_2019.dat'),
        # stand archive data file
        "dbh_path": DATA_PATH.joinpath("forcing/hyde_treedata.txt"),
        # tree height layer grid
        "z": Q_(np.linspace(0, 30, 31), "m"),
        # stand location
        "loc": {"lat": 61.51, "lon": 24.0},
        # controls
        "ctr": {
            # 'Ags_model': 'MF', # Medlyn-Farquhar
            "phenology": True,  # include phenology
            "leaf_area": True,  # seasonal LAI cycle
            "water_stress": True,
        },
        "water_p": water_p,
        "soil_p": soil_p,
        # cumulative relative upper bounds of number of trees to
        # build dbh classes
        "dbh_quantiles": Q_([0.25, 0.5, 0.75, 1.0], ""),
        # single tree class to be used (there's only one)
        "SingleTreeClass": SingleTree,
        # soil C model to be used
        "soil_model": SimpleSoilCModel(),
        # wood product model to be used
        "wood_product_model": SimpleWoodProductModel(),
        "wood_product_interface_name": "default",
    }
}
"""Stand parameters."""

simulation_params = {
    # data and forcing files
    #   INACTIVE "dpath": DATA_PATH.joinpath('forcing/FIHy_data_1997_2019.dat'),
    "fpath": DATA_PATH.joinpath("forcing/FIHy_forcing_1999_2019.dat"),
}
"""Default simulation parameters."""
