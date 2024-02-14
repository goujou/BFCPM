"""
Parameter collection.

This collection is far from being exhaustive. Other parameters can be found
in (at least):

- trees/single_tree_params.py
- soil/simple_soil_model/
- wood_products/simple_wood_products_model/
"""
from typing import Any, Dict

from . import Q_
from .type_aliases import GlobalTreeParams

BREAST_HEIGHT = Q_("1.3 m")
"""Height in which diameter in breast height is measured."""


water_p = {
    "interception": {"LAI": 4.0, "wmax": 0.2, "alpha": 1.28},
    "snowpack": {
        "Tm": 0.0,
        "Km": 2.9e-5,
        "Kf": 5.8e-6,
        "R": 0.05,
        "Tmax": 0.5,
        "Tmin": -0.5,
        "Sliq": 0.0,
        "Sice": 0.0,
    },
    "organic_layer": {
        "DM": 0.1,
        "Wmax": 10.0,
        "Wmin": 0.1,
        "Wcrit": 4.0,
        "alpha": 1.28,
        "Wliq": 10.0,
    },
}
"""
Water model parameters.
"""


# bucket
soil_p = {
    "depth": 0.5,
    "Ksat": 1e-5,
    "pF": {
        "ThetaS": 0.50,
        "ThetaR": 0.03,
        "alpha": 0.06,
        "n": 1.35,
    },  # Hyytiala A-horizon pF curve
    "MaxPond": 0.0,
    "Wliq": 0.4,
}
"""
Soil model parameters (non-carbon).
"""

global_tree_params: GlobalTreeParams = {
    "pine": {
        "leaf": {
            "length": 0.01,
            "PARalbedo": 0.1,
            "NIRalbedo": 0.5,
            "emissivity": 0.98,
        },
        # A-gs model
        "photop": {
            "name": "conifer",
            "Vcmax": 60.0,
            "Jmax": 114.0,  # 1.97*Vcmax (Kattge and Knorr, 2007)
            "Rd": 0.5,  # 0.023*Vcmax
            "tresp": {  # temperature response parameters (Kattge and Knorr, 2007)
                "Vcmax": [78.0, 200.0, 649.0],
                "Jmax": [56.0, 200.0, 646.0],
                "Rd": [33.0],
            },
            "alpha": 0.3,  # quantum efficiency parameter -
            "theta": 0.7,  # curvature parameter
            "g1": 2.6,  # stomatal slope kPa^(0.5)
            "g0": 1.0e-3,  # residual conductance mol m-2 s-1
            "La": 1e-4,  # Katul optimal model parameter
            #            "kn": 1.0,  # nitrogen attenuation coefficient -
            #            "kn": 0.0,  # nitrogen attenuation coefficient -
            "beta": 0.95,  # co-limitation parameter -
            "drp": [0.39, 0.83, 0.31, 3.0],  # Rew-based drought response
        },
        # cycle of photosynthetic activity
        "phenop": {
            "Xo": 0.0,
            "fmin": 0.1,
            "Tbase": -4.67,  # Kolari 2007
            "tau": 8.33,  # Kolari 2007
            "smax": 15.0,  # Kolari 2014
        },
        # cycle of LAI
        "laip": {
            "lai_min": 0.8,
            "lai_ini": None,
            "DDsum0": 0.0,
            "Tbase": 5.0,
            "ddo": 45.0,
            "ddmat": 250.0,
            "sdl": 12.0,
            "sdur": 30.0,
        },
    },
    "spruce": {
        "leaf": {
            "length": 0.01,
            "PARalbedo": 0.1,
            "NIRalbedo": 0.5,
            "emissivity": 0.98,
        },
        # A-gs model
        "photop": {
            "name": "conifer",
            #            "Vcmax": 60.0,
            "Vcmax": 50.0,
            "Jmax": 114.0,  # 1.97*Vcmax (Kattge and Knorr, 2007)
            "Rd": 0.5,  # 0.023*Vcmax
            "tresp": {  # temperature response parameters (Kattge and Knorr, 2007)
                "Vcmax": [78.0, 200.0, 649.0],
                "Jmax": [56.0, 200.0, 646.0],
                "Rd": [33.0],
            },
            "alpha": 0.3,  # quantum efficiency parameter -
            "theta": 0.7,  # curvature parameter
            "g1": 2.6,  # stomatal slope kPa^(0.5)
            "g0": 1.0e-3,  # residual conductance mol m-2 s-1
            "La": 1e-4,  # Katul optimal model parameter
            #            "kn": 1.0,  # nitrogen attenuation coefficient -
            #            "kn": 0.20,  # nitrogen attenuation coefficient -
            "beta": 0.95,  # co-limitation parameter -
            "drp": [0.39, 0.83, 0.31, 3.0],  # Rew-based drought response
        },
        # cycle of photosynthetic activity
        "phenop": {
            "Xo": 0.0,
            "fmin": 0.1,
            "Tbase": -4.67,  # Kolari 2007
            "tau": 8.33,  # Kolari 2007
            "smax": 15.0,  # Kolari 2014
        },
        # cycle of LAI
        "laip": {
            "lai_min": 0.8,
            "lai_ini": None,
            "DDsum0": 0.0,
            "Tbase": 5.0,
            "ddo": 45.0,
            "ddmat": 250.0,
            "sdl": 12.0,
            "sdur": 30.0,
        },
    },
    "birch": {
        #        "SLA": Q_(12.0, "m^2/kg_dw"), # marklund code
        "leaf": {
            "length": 0.05,
            "PARalbedo": 0.1,
            "NIRalbedo": 0.5,
            "emissivity": 0.98,
        },
        # A-gs model
        "photop": {
            "name": "decid",
            "Vcmax": 60.0,
            "Jmax": 114.0,  # 1.97*Vcmax (Kattge and Knorr, 2007)
            "Rd": 0.5,  # 0.023*Vcmax
            "tresp": {  # temperature response parameters (Kattge and Knorr, 2007)
                "Vcmax": [78.0, 200.0, 649.0],
                "Jmax": [56.0, 200.0, 646.0],
                "Rd": [33.0],
            },
            "alpha": 0.3,  # quantum efficiency parameter -
            "theta": 0.7,  # curvature parameter
            "g1": 5.0,  # stomatal slope kPa^(0.5)
            "g0": 1.0e-3,  # residual conductance mol m-2 s-1
            "La": 1e-4,  # Katul optimal model parameter
            #            "kn": 0.5,  # nitrogen attenuation coefficient -
            "beta": 0.95,  # co-limitation parameter -
            "drp": [0.39, 0.83, 0.31, 3.0],  # Rew-based drought response
        },
        # cycle of photosynthetic activity
        "phenop": {
            "Xo": 0.0,
            "fmin": 0.01,
            "Tbase": -4.67,  # Kolari 2007
            "tau": 8.33,  # Kolari 2007
            "smax": 15.0,  # Kolari 2014
        },
        # annual cycle of LAI
        "laip": {
            "lai_min": 0.1,  # relative to LAImax
            "lai_ini": None,
            "DDsum0": 0.0,
            "Tbase": 5.0,
            "ddo": 45.0,
            "ddmat": 250.0,
            "sdl": 12.0,
            "sdur": 30.0,
        },
    },
}
"""
Global tree parameters.
"""


###############################################################################


if __name__ == "__main__":
    pass
