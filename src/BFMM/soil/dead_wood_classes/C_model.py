"""
Soil C model with 8 pools: Litter, 6 dead-wood classes, SOC.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from bgc_md2.models.ACGCASoilModelDeadWoodClasses.source import (srm, u_1, u_2,
                                                                 u_3, u_4, u_5,
                                                                 u_6, u_Litter)
from CompartmentalSystems.helpers_reservoir import \
    numerical_function_from_expression

from ... import Q_
from ..soil_c_model_abc import SoilCModelABC
from .C_model_parameters import initialize_params


class SoilCDeadWoodClasses(SoilCModelABC):
    """Soil C model with Litter, 6 dead-wood classes, and SOC."""

    def __init__(self, *args, **kwargs):
        super().__init__(srm, initialize_params, *args, **kwargs)

    ###########################################################################

    # required by asbtract base class
    def _create_U_func(self) -> Callable:
        return numerical_function_from_expression(
            self.srm.external_inputs, (u_Litter, u_1, u_2, u_3, u_4, u_5, u_6), {}, {}
        )

    # required by asbtract base class
    def _create_U(self, input_fluxes: dict[str, Q_[float]]) -> np.ndarray:
        nr_pools = self.srm.nr_pools

        _u_Litter = input_fluxes.get("Litter", Q_(0, self.flux_unit)).to(self.flux_unit)
        _u_1 = input_fluxes.get("DWC_1", Q_(0, self.flux_unit)).to(self.flux_unit)
        _u_2 = input_fluxes.get("DWC_2", Q_(0, self.flux_unit)).to(self.flux_unit)
        _u_3 = input_fluxes.get("DWC_3", Q_(0, self.flux_unit)).to(self.flux_unit)
        _u_4 = input_fluxes.get("DWC_4", Q_(0, self.flux_unit)).to(self.flux_unit)
        _u_5 = input_fluxes.get("DWC_5", Q_(0, self.flux_unit)).to(self.flux_unit)
        _u_6 = input_fluxes.get("DWC_6", Q_(0, self.flux_unit)).to(self.flux_unit)

        # U, external inputs
        U = self._U_func(
            _u_Litter.magnitude,
            _u_1.magnitude,
            _u_2.magnitude,
            _u_3.magnitude,
            _u_4.magnitude,
            _u_5.magnitude,
            _u_6.magnitude,
        ).reshape(nr_pools)

        return U
