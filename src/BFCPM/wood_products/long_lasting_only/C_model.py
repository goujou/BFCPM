"""
This module contains a very simple wood-product model.

One pool: long-lasting (:math:`\\mathrm{WP}_L`) material.
"""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np
from bgc_md2.models.BFCPMLongLastingOnlyWoodProductModel.source import (
    WP_L_input, srm)
from CompartmentalSystems.helpers_reservoir import \
    numerical_function_from_expression

from ... import Q_
from ..wood_product_model_abc import WoodProductModelABC
from .C_model_parameters import initialize_params


class LongLastingOnlyWoodProductModel(WoodProductModelABC):
    """Simple wood-product model with a fast and a slow pool."""

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
            self.srm.external_inputs, (WP_L_input,), {}, {}
        )

    # required by asbtract base class
    def _create_U(self, input_fluxes: Dict[str, Q_[float]]) -> np.ndarray:
        nr_pools = self.srm.nr_pools

        flux_unit = self.flux_unit
        _WP_L_input = input_fluxes.get("WP_L", Q_(0, flux_unit)).to(flux_unit)

        # U, external inputs
        U = self._U_func(_WP_L_input.magnitude).reshape(nr_pools)

        return U
