"""
Helper functions to prepare a :class:`~stand.Stand`.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

from . import Q_
from .simulation_parameters import (tree_soil_interfaces,
                                    wood_product_interfaces)
from .soil.soil_c_model_abc import SoilCModelABC
from .type_aliases import TreeSoilInterface, WoodProductInterface
from .wood_products.wood_product_model_abc import WoodProductModelABC


def load_data_from_dbh_file(
    dbh_path: Path, species: str
) -> Tuple[Q_[np.ndarray], Q_[np.ndarray]]:
    """
    Load tree stand data from archive.

    Args:
        dbh_path: file path of stand archive
        species: element of ["pine", "spruce", "birch"]

    Returns:
        Tuple with tree stand data.

        - dbhs: tree diameters in breast height [cm]
        - N_dbhs: corrensponding number of trees per hectare [ha-1]
    """
    dat = np.loadtxt(dbh_path, skiprows=1)

    # use year 2008 data
    if species == "pine":
        data = dat[:, [0, 2]]
    elif species == "spruce":
        data = dat[:, [0, 4]]
    elif species == "birch":
        data = dat[:, [0, 6]]
    else:
        raise ValueError(f"Unknown species: '{species}'")

    # assign correct units
    dbhs = Q_(data[:, 0], "cm")
    N_dbhs = Q_(data[:, 1], "1/ha")

    return dbhs, N_dbhs


# @ureg.check("[length]", "1/[area]", "[length]", None, None)
# def get_crown_leaf_mass_profiles(
#    dbhs: Q_[np.ndarray],
#    N_dbhs: Q_[np.ndarray],
#    z: Q_[np.ndarray],
#    species: str,
#    dbh_quantiles: Q_[np.ndarray]
# ) -> dict[str, Q_[np.ndarray]]:
#    """
#    Collect stand archive data in dbh classes.
#
#    Args:
#        dbhs: tree diameters in breast height [cm]
#        N_dbhs: corrensponding number of trees per hectare [ha-1]
#        z: grid of tree height layer boundaries [m]
#        species: element of ["pine", "spruce", "birch"]
#        dbh_quantiles: cumulative frequency thresholds for grouping trees [-]
#
#    Returns:
#        Dictionary containing crown leaf biomass profiles.
#
#        - 'lad_normed_dbh_classes': normed lad over z per dbh class [m-1]
#        - 'avg_mleaf_dbh_classes': average leaf bioamass per tree in dbh class [kg_dw]
#        - 'N_dbh_classes': number of trees per hectare in dbh class [ha-1]
#    """
#    dz = z[1] - z[0]
#    nr_quantiles = len(dbh_quantiles)
#
#    # get crown leaf mass profiles
#    lad_normed_dbhs = Q_(np.zeros((len(z), len(dbhs))), "1/m")
#    mleaf_dbhs = Q_(np.zeros((len(dbhs), )), "kg_dw / ha")
#    for k, dbh in enumerate(dbhs):
#        lad_normed_dbh = Q_(
#            tree_utils.lad_normed_func(
#                dbh.to("cm").magnitude,
#                z.to("m").magnitude,
#                species
#            ),
#            "1/m"
#        )
#        lad_normed_dbhs[:, k] = lad_normed_dbh
#        mleaf_dbhs[k] = Q_(tree_utils.mleaf_from_dbh(
#            dbh.to("cm").magnitude,
#            species
#        ), "kg_dw") * N_dbhs[k]
#
#    lad_normed_dbh_classes = Q_(np.zeros((nr_quantiles, len(z))), "1/m")
#    N_dbh_classes = Q_(np.zeros((nr_quantiles, )), "1/ha")
#    mleaf_dbh_classes = Q_(np.zeros((nr_quantiles, )), "kg_dw / ha")
#
#    N_dbhs_cum = np.cumsum(N_dbhs) / sum(N_dbhs)  # relative frequency
#    m = 0.0
#    for k, q in enumerate(dbh_quantiles):
#        indices = np.where((N_dbhs_cum > m) & (N_dbhs_cum <= q))[0]
#        lad_dbh_class = np.sum(
#            lad_normed_dbhs[:, indices] * mleaf_dbhs[indices],
#            axis=1
#        )
#
#        if np.sum(lad_dbh_class * dz) > 0:
#            lad_normed_dbh_classes[k] =\
#                lad_dbh_class / np.sum(lad_dbh_class * dz)
#        N_dbh_classes[k] = np.sum(N_dbhs[indices])
#
#        mleaf_dbh_classes[k] = np.sum(mleaf_dbhs[indices])
#        m = dbh_quantiles[k]
#
#    # check that no tree is lost
#    assert np.allclose(np.sum(N_dbh_classes), np.sum(N_dbhs))
#
#    # check that no leaf biomass is lost
#    assert np.allclose(np.sum(mleaf_dbh_classes), np.sum(mleaf_dbhs))
#
#    # check that all lad's integrate to one for non-empty dbh classes
#    assert(
#        np.allclose(
#            np.sum(lad_normed_dbh_classes*dz, axis=1),
#            (N_dbh_classes > 0)
#        )
#    )
#
#    # compute average leaf biomass for dbh classes
#    indices = N_dbh_classes != 0
#    lad_normed_dbh_classes = lad_normed_dbh_classes[indices, :]
#    mleaf_dbh_classes = mleaf_dbh_classes[indices]
#    N_dbh_classes = N_dbh_classes[indices]
#    avg_mleaf_dbh_classes = mleaf_dbh_classes / N_dbh_classes
#
#    return {
#        'lad_normed_dbh_classes': lad_normed_dbh_classes,
#        'avg_mleaf_dbh_classes': avg_mleaf_dbh_classes,
#        'N_dbh_classes': N_dbh_classes
#    }


# def compute_mean_tree_data(
#    dbh_path: Path,
#    z: np.ndarray,
#    species: str,
#    dbh_quantiles: Q_[np.ndarray],
#    max_dbh: Q_ = Q_(np.inf, "cm")
# ) -> dict[str, Q_[np.ndarray]]:
#    """
#    Load stand archive data and collect it in dbh classes.
#
#    Args:
#        dbh_path: file path of stand archive
#        z: grid of tree height layer boundaries [m]
#        species: element of ["pine", "spruce", "birch"]
#        dbh_quantiles: cumulative frequency thresholds for grouping trees [-]
#        max_dbh: only trees with dhbs les than or equal to max_dbh
#            will be considered [cm]
#
#    Returns:
#        Dictionary with prepared mean tree data.
#
#        - 'lad_normalized_dbh_classes': normalized lad over z per dbh class [m-1]
#        - 'avg_mleaf_dbh_classes': average leaf bioamass per tree in dbh class [kg_dw]
#        - 'N_dbh_classes': number of trees per hectare in dbh class [ha-1]
#    """
#    dbhs, N_dbhs = load_data_from_dbh_file(dbh_path, species)
#    mean_tree_data = get_crown_leaf_mass_profiles(
#        dbhs[dbhs < max_dbh],
#        N_dbhs[dbhs < max_dbh],
#        z,
#        species,
#        dbh_quantiles
#    )
#
#    return mean_tree_data


def load_tree_soil_interface(
    tree_class: type, soil_model: SoilCModelABC
) -> TreeSoilInterface:
    """Load an interface for fluxes between trees and the soil.

    The interface dictionary is a sub-dictionary loaded from
    :obj:`~.simulation_parameters.tree_soil_interfaces`.

    Args:
        tree_class: class to which a tree belongs
        soil_model: a soil model instance

    Returns:
        dictionary with tree soil interface

            - "tree_pool_from": {"soil_pool_to_1": fraction, ...}
    """
    for (
        TreeClass,
        SoilCModelClass,
    ), tree_soil_interface in tree_soil_interfaces.items():
        if issubclass(tree_class, TreeClass) and isinstance(
            soil_model, SoilCModelClass
        ):
            #        if issubclass(tree_class, TreeClass) and issubclass(soil_model, SoilCModelClass):
            #            return tree_soil_interfaces[(TreeClass, SoilCModelClass)]
            return tree_soil_interface

    msg = f"Found no tree soil interface for {tree_class} and {soil_model}"
    raise ValueError(msg)


def load_wood_product_interface(
    name: str,
    tree_class: type,
    soil_model: SoilCModelABC,
    wood_product_model: WoodProductModelABC,
) -> WoodProductInterface:
    """Load an interface for fluxes between trees, the soil, and wood products.

    The interface dictionary is a sub-dictionary loaded from
    :obj:`~.simulation_parameters.wood_product_interfaces`.

    Args:
        name: name of the interface
        tree_class: class to which a tree belongs
        soil_model: a soil model instance
        wood_product_model: a wood product model instance

    Returns:
        dictionary with soil and wood-product interface

            - "tree_pool_from":
                - {"soil_pool_to_1" or "wood_product_pool_to_1": fraction, ...}
    """
    for (
        TreeClass,
        SoilCModelClass,
        WoodProductModelClass,
    ), wp_interface_dict in wood_product_interfaces.items():
        if (
            issubclass(tree_class, TreeClass)
            and isinstance(soil_model, SoilCModelClass)
            and isinstance(wood_product_model, WoodProductModelClass)
        ):
            return wp_interface_dict[name]

    msg = "Found no wood product interface for "
    msg += f"{tree_class}, {soil_model}, and {wood_product_model}."
    raise ValueError(msg)
