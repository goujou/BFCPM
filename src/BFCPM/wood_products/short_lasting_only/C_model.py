"""
This module contains a very simple wood-product model.

One pool: short-lasting (:math:`\\mathrm{WP}_S`) material.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
from bgc_md2.models.BFCPMShortLastingOnlyWoodProductModel.source import (
    WP_S_input, srm)
from CompartmentalSystems.helpers_reservoir import \
    numerical_function_from_expression

from ... import Q_
from ..wood_product_model_abc import WoodProductModelABC
from .C_model_parameters import initialize_params


class ShortLastingOnlyWoodProductModel(WoodProductModelABC):
    """Very simple wood-product model with a fast pool only"""

    def __init__(self, *args, **kwargs):
        super().__init__(srm, initialize_params, *args, **kwargs)

    ###########################################################################

    # required by abstract base class
    @property
    def nr_pools(self):
        return self.srm.nr_pools

    # required by abstract base class
    @property
    def pool_names(self) -> List[str]:
        return [sv.name for sv in self.srm.state_vector]

    ###########################################################################

    # required by asbtract base class
    def _create_U_func(self) -> Callable:
        return numerical_function_from_expression(
            self.srm.external_inputs, (WP_S_input,), {}, {}
        )

    # required by asbtract base class
    def _create_U(self, input_fluxes: Dict[str, Q_[float]]) -> np.ndarray:
        nr_pools = self.srm.nr_pools

        flux_unit = self.flux_unit
        _WP_S_input = input_fluxes.get("WP_S", Q_(0, flux_unit)).to(flux_unit)

        # U, external inputs
        U = self._U_func(_WP_S_input.magnitude).reshape(nr_pools)

        return U
