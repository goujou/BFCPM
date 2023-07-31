#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constants used in the model calculations.

Authors:
    - Samuli Launiainen
"""

import numpy as np

EPS = np.finfo(float).eps
"""machine epsilon"""

micromolCO2_TO_gC = 1.201e-5
"""[mol CO2 to g C]"""

LATENT_HEAT = 44100.0
"""[J mol\ :sup:`-1`\ ], latent heat of vaporization at 20\ :math:`^{\circ}`\ C"""

MOLAR_MASS_H2O = 18.015e-3
"""[kg mol\ :sup:`-1`\ ], molar mass of H\ :sub:`2`\ O"""

MOLAR_MASS_CO2 = 44.01e-3
"""[kg mol\ :sup:`-1`\ ], molar mass of CO\ :sub:`2`"""

MOLAR_MASS_C = 12.01e-3
"""[kg mol\ :sup:`-1`\ ], molar mass of C"""

MOLAR_MASS_AIR = 29.0e-3
"""[kg mol\ :sup:`-1`\ ], molar mass of air"""

SPECIFIC_HEAT_H2O = 4.18e3
"""[J kg\ :sup:`-1` K\ :sup:`-1`\ ], specific heat of H\ :sub:`2`\ O"""

SPECIFIC_HEAT_ORGANIC_MATTER = 1.92e3
"""[J kg\ :sup:`-1` K\ :sup:`-1`\ ], specific heat of organic matter"""

SPECIFIC_HEAT_AIR = 29.3
"""[J mol\ :sup:`-1` K\ :sup:`-1`\ ], heat capacity of air at constant pressure"""

STEFAN_BOLTZMANN = 5.6697e-8
"""[W m\ :sup:`-2` K\ :sup:`-4`\ ], Stefan-Boltzmann constant"""

VON_KARMAN = 0.41
"""[-], von Karman constant"""

DEG_TO_KELVIN = 273.15
"""[K], zero degrees celsius in Kelvin"""

NORMAL_TEMPERATURE = 273.15
"""[K], zero degrees celsius in Kelvin"""

AIR_DENSITY = 41.6
"""[mol m\ :sup:`-3`\ ], density of air at 20\ :math:`^{\circ}`\ C"""

AIR_VISCOSITY = 15.1e-6
"""[m\ :sup:`2` s\ :sup:`-1`\ ], kinematic viscosity of air at 20\ :math:`^{\circ}`\ C"""

THERMAL_DIFFUSIVITY_AIR = 21.4e-6
"""[m\ :sup:`2` s\ :sup:`-1`\ ], thermal diffusivity of air at 20\ :math:`^{\circ}`\ C"""

MOLECULAR_DIFFUSIVITY_CO2 = 15.7e-6
"""[m\ :sup:`2` s\ :sup:`-1`\ ], molecular diffusvity of CO\ :sub:`2` at 20\ :math:`^{\circ}`\ C"""

MOLECULAR_DIFFUSIVITY_H2O = 24.0e-6
"""[m\ :sup:`2` s\ :sup:`-1`\ ], molecular diffusvity of H\ :sub:`2` O at 20\ :math:`^{\circ}`\ C"""

GAS_CONSTANT = 8.314
"""[J mol\ :sup:`-1` K\ :sup:`-1`], universal gas constant"""

GRAVITY = 9.81
"""[kg m\ :sup:`2` s\ :sup:`-1`], standard gravity"""

WATER_DENSITY = 1.0e3
"""[kg m\ :sup:`-3`\ ], water density"""

PAR_TO_UMOL = 4.56
"""[umol m\ :sup:`2` s\ :sup:`-1`], conversion from watts to micromol"""

DEG_TO_RAD = 3.14159 / 180.0
"""[rad], conversion from deg to rad"""

O2_IN_AIR = 2.10e5
"""[umol m\ :sup:`-1`], O :sup`2` concentration in air"""

afact = 1.6
"""H :sup`2` O/CO :sup`2` diffusivity ratio"""
