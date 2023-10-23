"""Main module, set up unit registry, new units, unit conversions, unit helper functions."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pint
from pint import Quantity
from scipy.integrate import quad

# raise warnings for debugging purposes
# unfortunately prevents unittest from working...
# import warnings
# warnings.simplefilter("error")

BASE_PATH = Path(__file__).parent.parent.parent
"""Package base path."""

FIGS_PATH = BASE_PATH.joinpath("figs")
"""Path for figures."""
FIGS_PATH.mkdir(parents=False, exist_ok=True)

DATA_PATH = BASE_PATH.joinpath("data")
"""Package data folder path."""
DATA_PATH.mkdir(parents=False, exist_ok=True)

PRE_SPINUPS_PATH = DATA_PATH.joinpath("pre_spinups")
"""Pre-spinups folder path."""
PRE_SPINUPS_PATH.mkdir(parents=False, exist_ok=True)

SIMULATIONS_PATH = DATA_PATH.joinpath("simulations")
"""Simulation output data path."""
SIMULATIONS_PATH.mkdir(parents=False, exist_ok=True)

LOGS_PATH = DATA_PATH.joinpath("logfiles")
"""Path for general log files."""
LOGS_PATH.mkdir(parents=False, exist_ok=True)


# define new units and their relations
# ureg = pint.UnitRegistry()
ureg = pint.get_application_registry()
ureg.setup_matplotlib()
Q_ = ureg.Quantity
"""Pint Quantity with custom units and additional methods.

Custom units are ``gC``, ``g_gluc``, and ``g_dw``. Custom methods
are defined below in this module.
"""

try:
    ureg.define("g_carbon = [mass_carbon] = gC")
    ureg.define("g_gluc = [mass_glucose]")
    ureg.define("g_dw = [mass_dryweight]")
except pint.RedefinitionError:
    pass

zeta_gluc = Q_(6 * 12 / 180.15, "gC/g_gluc")
"""Grams of carbon per grams glucose."""
zeta_dw = Q_(0.5, "gC/g_dw")
"""Grams of carbon per grams dry weight."""

c = pint.Context("allocation_model")
c.add_transformation("[mass_carbon]", "[mass_glucose]", lambda ureg, x: x / zeta_gluc)
c.add_transformation("[mass_glucose]", "[mass_carbon]", lambda ureg, x: x * zeta_gluc)

c.add_transformation("[mass_carbon]", "[mass_dryweight]", lambda ureg, x: x / zeta_dw)
c.add_transformation("[mass_dryweight]", "[mass_carbon]", lambda ureg, x: x * zeta_dw)
ureg.enable_contexts(c)


def integrate_with_units(
    #    f: Callable, a: Quantity[float], b: Quantity[float]
    f: Callable,
    a: Quantity,
    b: Quantity,
) -> Quantity:
    """Integrate a function using units.

    Args:
        f: integrand
        a: lower integral boundary
        b: upper integral boundary

    Returns:
        ``quad(f, a, b)`` with units
    """
    units_in = a.units
    b = b.to(units_in)
    units_out = f(a).units

    def integrand(number):
        q = Q_(number, units_in)
        return f(q).magnitude

    res = quad(integrand, a.magnitude, b.magnitude)[0]
    return Q_(res, units_in * units_out)


def from_list(l: list, unit: str = None) -> Quantity:
    """Create quantity array from a list of quantities.

    Args:
        l: list of quantities with compatible units
        unit: optional, force a unit

    Returns:
        Q_(np.array([e for e in l])) with correct units
    """

    def figure_out_unit():
        i = 0
        while (i <= len(l)) and (not hasattr(l[i], "units")):
            i += 1

        if i == len(l):
            return ""

        return l[i].units

    if not unit:
        base_units = figure_out_unit()
    else:
        base_units = unit

    arr = np.nan * np.ones(len(l), dtype=np.float64)
    for k, q in enumerate(l):
        if hasattr(q, "to"):
            arr[k] = q.to(base_units).magnitude
        else:
            arr[k] = q

    q_arr = Q_(arr, base_units)

    return q_arr


setattr(Q_, "from_list", from_list)


def get_netcdf_unit(q: Quantity) -> str:
    """Get the netcdf unit from a Quantity.

    Args:
        q: The quantity to be translated.

    Returns:
        The netcdf translation."""
    u = f"{q.units:~}".replace(" a", " yr")
    u = u.replace("ha", "HA")
    u = u.replace("a", "yr")
    return u.replace("HA", "ha")


setattr(Q_, "get_netcdf_unit", get_netcdf_unit)
