"""Collection of type aliases used in the project."""

from __future__ import annotations

from typing import (TYPE_CHECKING, Any, Dict, List, NewType, Optional, Tuple,
                    TypedDict, Union)

from . import Q_

if TYPE_CHECKING:
    from .management.management_strategy import ManagementStrategy


Quantity = NewType("Q_", Q_)  # type: ignore
"""Pint Quantity type. Displayed as ``Q_``."""


TreeSoilInterface = Dict[str, Dict[str, float]]
"""Interface between a tree and a soil model.

Shape: {tree_pool_from: {soil_pool_to_1: frac1}, ...}. See also values of
:attr:`~.simulation_parameters.tree_soil_interfaces`.
"""


WoodProductInterface = Dict[str, Any]
"""Interface between a tree and a wood product model.

See values of :attr:`~.simulation_parameters.wood_product_interfaces`.
"""


InputFluxes = Dict[str, Quantity]
"""Fluxes into pools. Key is pool name."""


OutputFluxesInt = Dict[int, Quantity]
"""Fluxes out of pools. Key is pool number not name."""


TreeExternalOutputFluxes = Dict[Tuple[str, Union[str, None]], Quantity]
"""External output fluxes from a tree.

External output fluxes from
:class:`.trees.single_tree_C_model.SingleTreeCModel`.
According to :obj:`~.type_aliases.TreeSoilInterface`,
:class:`~.stand.Stand` distributes them to the atmosphere or
to the soil or the atmosphere (pool_to=None).
"""


CuttingFluxes = Dict[Tuple[str, str], Quantity]
"""Distribution of tree material at cutting or thinning.

- (pool_from, pool_to): fraction, pool_to = None means output to atmosphere
"""


OneSpeciesParams = Dict[str, Any]
"""Parameter dictionary for one species.
Examples in :mod:`~trees.single_tree_params`.
"""


SpeciesParams = Dict[str, OneSpeciesParams]
"""Parameter dictionary for all involved tree species.

Example in :mod:`~.trees.single_tree_params`.
"""


MSData = Tuple[str, str]
"""The tuple is a (Trigger, Action) pair for a MeanTree to plant.

The strings must be found in :mod:`~.management.library.py`.
"""


MSDataList = List[MSData]
"""A list of management_strategy_data for a MeanTree to plant."""


SimulationProfile = List[Tuple[str, float, float, MSDataList, Optional[str]]]
"""Simulation described by a list of trees.

List[Tuple[tree_species, dbh_in_cm, N_in_m2-1, management_strategy_data, waiting or not]].
'waiting' means the MeanTree is not immediately planted at the beginning of the simulation.
So it needs some kind of (trigger, plant_action) in the management_strategy_data.
It is optional, if omitted, the tree will be planted right away.
"""


TreeSetting = Tuple[int, Q_, Q_, "ManagementStrategy", Optional[str]]
"""A setting of a MeanTree to be planted in the :class:`~.stand.Stand`.
[nr_of_the_tree, dbh_in_cm, N_in_m2-1, management_strategy, 'waiting' or omitted].
"""


SpeciesSettings = Dict[str, List[TreeSetting]]
"""The species setting of a new :class:`~.stand.Stand`.

The key is supposed to give the tree species.
"""
