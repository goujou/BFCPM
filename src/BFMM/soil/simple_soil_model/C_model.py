"""
This module contains a simple soil model class with litter,
coarse woody debris, and soil organic carbon.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from bgc_md2.models.ACGCASoilModel2.source import srm, u_CWD, u_L
from CompartmentalSystems.helpers_reservoir import \
    numerical_function_from_expression

from ... import Q_
from ..soil_c_model_abc import SoilCModelABC
from .C_model_parameters import initialize_params


class SimpleSoilCModel(SoilCModelABC):
    """Simple soil model class with two litter and two soil pools."""

    def __init__(self, *args, **kwargs):
        super().__init__(srm, initialize_params, *args, **kwargs)

    ###########################################################################

    # required by asbtract base class
    def _create_U_func(self) -> Callable:
        return numerical_function_from_expression(
            self.srm.external_inputs, (u_L, u_CWD), {}, {}
        )

    # required by asbtract base class
    def _create_U(self, input_fluxes: dict[str, Q_[float]]) -> np.ndarray:
        nr_pools = self.srm.nr_pools

        flux_unit = self.flux_unit
        _u_L = input_fluxes.get("Litter", Q_(0, flux_unit)).to(flux_unit)
        _u_CWD = input_fluxes.get("CWD", Q_(0, flux_unit)).to(flux_unit)

        # U, external inputs
        U = self._U_func(_u_L.magnitude, _u_CWD.magnitude).reshape(nr_pools)

        return U
