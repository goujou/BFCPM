"""
This module contains all species-specific tree parameters.
"""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
from sympy import Symbol, latex

from .. import Q_, zeta_dw, zeta_gluc
from ..productivity.constants import micromolCO2_TO_gC
from ..type_aliases import OneSpeciesParams, SpeciesParams
from ..utils import latexify_unit


def _compute_construction_costs_from_growth_respiration_rate(r, delta):
    # here I assume that NPP has RM AND RG removed, possibly
    # in Ryan 97 they assume here that NPP has ONLY RM removed
    # maybe even delta should be ignored
    #
    # find r_L, r_W in Ryan 1997, p.878
    # (1) GR_L = r_L * NPP_L,  NPP_L = GPP_L - RA_L, GPP_L = f_L*E + MR_L, RA_L = MR_L + GR_L
    # (2) GR_L = f_L*E * C_gL/(C_gL+delta_L) * (1 - zeta_dw/(zeta_gluc*C_gL))
    # solved for C_gL
    #    return (1+r) * zeta_dw/zeta_gluc + r * delta

    # (1) NPP = GPP - RM
    return ((zeta_dw / zeta_gluc + r * delta) / (1 - r)).magnitude


def _compute_growth_respiration_rate_from_construction_costs(c, delta):
    return (1 - zeta_dw / (zeta_gluc * c)) * c / (c + delta)


# species_params: dict[str, dict[str, Any]] = dict()
species_params: SpeciesParams = dict()
"""All species-specific parameters."""


species_params["pine"] = {
    # photosynthesis
    "alpha": {
        "value": 1.0,
        "unit": "",
        "info": {
            "descr": "additional limitation factor for carbon uptake",
            "source": "empirical, for balanced mixed-species simulation",
        },
    },
    # specific leaf area
    "SLA": {
        "value": 6.162,
        "unit": "m^2/kg_dw",
        "info": {"descr": "specific leaf area", "source": r"\citet{Goude2019EJFR}"},
    },
    # respiration parameters
    "R_mL": {
        "value": 0.95,
        "unit": "g_gluc/g_dw/yr",
        "info": {
            "descr": "maintenance respiration rate of leaves",
            "source": r"\citet[Table~2]{Ogle2009TP}",
        },
    },
    "R_mR": {
        "value": 0.75,
        #        "value": 1.125, # Ryan 1997?
        "unit": "g_gluc/g_dw/yr",
        "info": {
            "descr": "maintenance respiration rate of fine roots",
            "source": r"\citet[Table~2]{Ogle2009TP}",
        },
    },
    "R_mS": {
        #        "value": 0.025,
        "value": (
            (
                Q_(1, "ngC/gC/s").to("gC/gC/yr")
                / zeta_gluc
                * zeta_dw
                * np.array([0.89, 2.31])
            ).magnitude
        ).mean(),  # 0.063
        "unit": "g_gluc/g_dw/yr",
        "info": {
            "descr": "maintenance respiration rate of sapwood",
            #            "source": "Ogle2009TP, Table~2"
            "source": r"\citet[Table~5, northern]{Lavigne1997TP}",
        },
    },
    # senescence parameters
    "S_L": {
        #        "value": 0.3,
        "value": 0.2,
        "unit": "1/yr",
        "info": {
            "descr": "senescence rate of leaves",
            #            "source": r"\citet[Table~2]{Pukkala2014FPE}",
            "source": r"\citet[Table~3]{Muukkonen2005Trees}",
        },
    },
    "S_R": {
        "value": 0.811,
        "unit": "1/yr",
        "info": {
            "descr": "senescence rate of fine roots",
            "source": r"\citet[Table~2]{Pukkala2014FPE}",
        },
    },
    "S_O": {
        #        "value": 0.0125,
        "value": 0.04,
        "unit": "1/yr",
        "info": {
            "descr": "senescence rate of coarse roots and branches",
            #            "source": "Pukkala2014FPE, Table~2"
            "source": (
                r"\citet[Table~1]{Vanninen2005TP}; also following simulations "
                r"for coarse roots, Eq.~(10) leads to $0.06$ for branches, "
                "we took one of the two"
            ),
        },
    },
    # root biomass to leaf biomass ratio
    "rho_RL": {
        # seems to heavily depend on stand fertility (high=0.66, low=0.36)
        # Vanninnen2005TP, Table 1
        # see also Helmisaari2007TP, Fig. 6
        "value": 0.67,
        "unit": "",
        "info": {
            "descr": "fine root to foliage biomass ratio",
            "source": r"\citet[Table~2]{Pukkala2014FPE}",
        },
    },
    # trunk shape parameters
    "eta_B": {
        "value": 0.045,
        "unit": "",
        "info": {
            "descr": (
                "relative height at which trunk transitions from a neiloid "
                "to a paraboloid"
            ),
            "source": r"\citet[Table~2]{Ogle2009TP}",
        },
    },
    "eta_C": {
        "value": 0.71,
        "unit": "",
        "info": {
            "descr": (
                "relative height at which trunk transitions from a paraboloid "
                "to a cone"
            ),
            "source": r"\citet[Table~2, called $\eta$]{Ogle2009TP}",
        },
    },
    # sapwood texture parameters
    "gamma_X": {
        "value": 0.62,
        "unit": "",
        "info": {
            "descr": "xylem conducting area to sapwood area ratio",
            "source": r"\citet[Table~2]{Ogle2009TP}",
        },
    },
    "gamma_C": {
        "value": 265_000,
        "unit": "g_gluc/m^3",
        "info": {
            "descr": "maximum storage capacity of living sapwood cells",
            "source": r"\citet[Table~2]{Ogle2009TP}",
        },
    },
    "gamma_W": {
        "value": 6.67e-07,
        "unit": "m^3/g_dw",
        "info": {
            "descr": "(inverse) density of sapwood structural tissue",
            "source": r"\citet[Table~2]{Ogle2009TP}",
        },
    },
    # sapwood depth/width paramters
    # currently SW is computed as in spruce
    #    "SW_constant": {
    #        "value": 1.049,
    #        "unit": "",
    #        "info": {
    #            "descr": "intercept for linear sapwood width model",
    #            "source": r"\citet[Table~3]{Vanninen2005TP}",
    #        },
    #    },
    #    "SW_H": {
    #        "value": -8.58e-05,
    #        "unit": "",
    #        "info": {
    #            "descr": "height parameter for linear sapwood width model",
    #            "source": r"\citet[Table~3]{Vanninen2005TP}",
    #        },
    #    },
    #    "SW_A": {
    #        "value": -1.83e-03,
    #        "unit": "",
    #        "info": {
    #            "descr": "tree age parameter linear sapwood width model",
    #            "source": r"\citet[Table~3]{Vanninen2005TP}",
    #        },
    #    },
    # spruce parameters #TODO
    # entries here just for manuscript table; in computation values are taken
    # directly from spruce parameter set
    "SW_a": {
        "value": 18.8,
        "unit": "",
        "info": {
            "descr": "numerator parameter for sapwood width model",
            "source": r"\citet[Eq.~\(2\)]{Sellin1994CJFE}",
        },
    },
    "SW_b": {
        "value": 60.0,
        "unit": "",
        "info": {
            "descr": "denominator parameter for sapwood width model",
            "source": r"\citet[Eq.~\(2\)]{Sellin1994CJFE}",
        },
    },
    "HW_slope": {
        "value": 24.0 / 50.0,
        "unit": "",
        "info": {
            "descr": "slope value for heartwood width line",
            "source": r"\citet[Fig.~1]{Sellin1994CJFE}",
        },
    },
    #    # wood density parameters
    #    "rho_W0": {
    #        "value": 460_000,
    #        "unit": "g_dw/m^3",
    #        "info": {
    #            "descr": r"initial wood density, used for $\dbh<\dbh_M$",
    #            "source": r"\citet[Table~1]{Pukkala2014FPE}",
    #        },
    #    },
    "rho_Wmax": {
        "value": 550_000,
        "unit": "g_dw/m^3",
        "info": {
            "descr": r"maximum density of newly produced sapwood",
            "source": r"computed to keep $\delta_W$ positive",
        },
    },
    "rho_Wmin": {
        "value": 280_000,
        "unit": "g_dw/m^3",
        "info": {
            "descr": "minimum wood density",
            "source": "empirical parameter after some testing",
        },
    },
    "dbh_M": {
        "value": 4.0,
        "unit": "cm",
        "info": {
            "descr": (
                r"for $\dbh<\dbh_M$ the allometrically derived "
                r"wood density is assumed to be useless"
            ),
            "source": "empirical parameter after some testing",
        },
    },
    # labile carbon storage parameters
    "delta_L": {
        "value": 0.11,
        "unit": "g_gluc/g_dw",
        "info": {
            "descr": "labile carbon storage capacity of leaves",
            "source": r"\citet[Table~2]{Ogle2009TP}",
        },
    },
    "delta_R": {
        "value": 0.08,
        "unit": "g_gluc/g_dw",
        "info": {
            "descr": "labile carbon storage capacity of fine roots",
            "source": r"\citet[Table~2]{Ogle2009TP}",
        },
    },
}

delta_L = species_params["pine"]["delta_L"]
delta_L = Q_(delta_L["value"], delta_L["unit"])
delta_R = species_params["pine"]["delta_R"]
delta_R = Q_(delta_R["value"], delta_R["unit"])
delta_W = delta_L  # from simulation experience

species_params["pine"].update(
    {
        # tissue construction parameters
        "C_gL": {
            #        "value": 1.51,
            "value": _compute_construction_costs_from_growth_respiration_rate(
                28.0 / 15 * 0.25, delta_L
            ),  # 2.44
            "unit": "g_gluc/g_dw",
            "info": {
                "descr": "construction costs of producing leaves",
                #            "source": "Ogle2009TP, Table~2"
                "source": (
                    r"\citet[p.878]{Ryan1997JGR} states that leaf construction "
                    r"costs were $28/15 \cdot 0.25$ (of leaf NPP)"
                ),
            },
        },
        "C_gR": {
            #        "value": 1.30,
            # Ryan 1997, Table 4
            # the desired root respiration rate is 3 times as high as in
            # current simulations, mR=0.75 g_gluc/g_dw/yr = 0.6/yr
            # could be increased, but
            # the construction respiration rate is 0.035/yr and so there is
            # more potential. Since maintenance
            # respired carbon and f_R_times_E are almost equal in size, we
            # multiply the current growth respiration rate by 6 and compute
            # the associated construction costs
            "value": _compute_construction_costs_from_growth_respiration_rate(
                Q_((3.6 + 5.6) / 2 * micromolCO2_TO_gC, "gC/kgC/s").to("kgC/kgC/yr")
                * Q_(1, "yr")
                / 0.6
                * 2
                * _compute_growth_respiration_rate_from_construction_costs(
                    Q_(1.30, "g_gluc/g_dw"), delta_R
                ),
                delta_R,
            ),  # 1.6
            "unit": "g_gluc/g_dw",
            "info": {
                "descr": "construction costs of producing fine roots",
                #            "source": "Ogle2009TP, Table~2"
                "source": r"\citet[Table~4]{Ryan1997JGR} and some empirical adaptation",
            },
        },
        "C_gHW": {
            "value": 1.0,  # keep at 1!!!, we only need it to correct for units
            #        "value": 1.251, # ratio between g_dw and g_gluc, otherwise 1 g_gluc
            # produces 1 g_dw in C, but there is more carbon in g_dw
            "unit": "g_gluc/g_dw",
            "info": {
                "descr": (
                    "construction costs of converting heartwood from "
                    "labile sapwood (actually: no costs)"
                ),
                "source": r"missing in \citet{Ogle2009TP} (causing a unit mismatch)",
            },
        },
        "C_gW": {
            #        "value": 1.47,
            "value": (
                Q_(1 + (0.24 + 0.25) / 2, "gC/gC") / zeta_gluc * zeta_dw
            ).magnitude,  # 1.557
            "unit": "g_gluc/g_dw",
            "info": {
                "descr": "construction costs of producing sapwood",
                #            "source": "Ogle2009TP, Table~2"
                "source": (
                    r"\citet[Table~5, northern]{Lavigne1997TP}, we add $1.0$ "
                    r"because for us growth is not part of the factor to multiply with"
                ),
            },
        },
    }
)


species_params["spruce"] = {
    # photosynthesis
    "alpha": {
        "value": 1.0,
        "unit": "",
        "info": {
            "descr": "additional limitation factor for carbon uptake",
            "source": "empirical, for balanced mixed-species simulation",
        },
    },
    # specific leaf area
    "SLA": {
        "value": 5.02,
        "unit": "m^2/kg_dw",
        "info": {"descr": "specific leaf area", "source": r"\citet{Goude2019EJFR}"},
    },
    # respiration parameters
    "R_mL": {
        "value": 0.95,
        "unit": "g_gluc/g_dw/yr",
        "info": {
            "descr": "maintenance respiration rate of leaves",
            "source": r"\citet[Table~2]{Ogle2009TP}",
        },
    },
    "R_mR": {
        "value": 0.75,
        #        "value": 1.125, # Ryan 1997?
        "unit": "g_gluc/g_dw/yr",
        "info": {
            "descr": "maintenance respiration rate of fine roots",
            "source": r"\citet[Table~2]{Ogle2009TP}",
        },
    },
    "R_mS": {
        #        "value": 0.025,
        "value": (
            (Q_(1, "ngC/gC/s").to("gC/gC/yr") / zeta_gluc * zeta_dw * 1.96).magnitude
        ),  # 0.077
        "unit": "g_gluc/g_dw/yr",
        "info": {
            "descr": "maintenance respiration rate of sapwood",
            #            "source": "Ogle2009TP, Table~2"
            "source": r"\citet[Table~5, northern]{Lavigne1997TP}",
        },
    },
    # senescence parameters
    "S_L": {
        #        "value": 0.2,
        "value": 0.1,
        "unit": "1/yr",
        "info": {
            "descr": "senescence rate of leaves",
            #            "source": r"\citet[Table~2]{Pukkala2014FPE}",
            "source": r"\citet{Muukkonen2004CJFR}",
        },
    },
    "S_R": {
        "value": 0.868,
        "unit": "1/yr",
        "info": {
            "descr": "senescence rate of fine roots",
            "source": r"\citet[Table~2]{Pukkala2014FPE}",
        },
    },
    "S_O": {
        #        "value": 0.0125,
        #        "value": 0.04,  # taken from pine
        "value": 0.0125,  # taken from pine
        "unit": "1/yr",
        "info": {
            "descr": "senescence rate of coarse roots and branches",
            #            "source": "Pukkala2014FPE, Table~2"
            #            "source": (
            #                r"\citet[Table~1]{Vanninen2005TP}; also following "
            #                r"simulations for coarse roots, Eq.~(10) leads to $0.06$ "
            #                r"for branches, we took one of the two"
            #            ),
            "source": r"\citet{Muukkonen2004CJFR}",
        },
    },
    # fine root lo leaf biomass ratio
    "rho_RL": {
        # see also Helmisaari2007TP, Fig. 6
        "value": 0.25,
        "unit": "",
        "info": {
            "descr": "fine root to foliage biomass ratio",
            "source": r"\citet[Table~2]{Pukkala2014FPE}",
        },
    },
    # trunk shape parameters
    "eta_B": {
        "value": 0.045,
        "unit": "",
        "info": {
            "descr": (
                "relative height at which trunk transitions from a neiloid "
                "to a paraboloid"
            ),
            "source": r"\citet[Table~2]{Ogle2009TP} (pine parameter)",
        },
    },
    "eta_C": {
        "value": 0.71,
        "unit": "",
        "info": {
            "descr": (
                "relative height at which trunk transitions from a paraboloid "
                "to a cone"
            ),
            "source": r"\citet[Table~2, called $\eta$]{Ogle2009TP} (pine parameter)",
        },
    },
    # sapwood texture parameters
    "gamma_X": {
        "value": 0.62,
        "unit": "",
        "info": {
            "descr": "xylem conducting area to sapwood area ratio",
            "source": r"\citet[Table~2]{Ogle2009TP} (pine parameter)",
        },
    },
    "gamma_C": {
        "value": 265_000,
        "unit": "g_gluc/m^3",
        "info": {
            "descr": "maximum storage capacity of living sapwood cells",
            "source": r"\citet[Table~2]{Ogle2009TP} (pine parameter)",
        },
    },
    "gamma_W": {
        "value": 6.67e-07,
        "unit": "m^3/g_dw",
        "info": {
            "descr": "(inverse) density of sapwood structural tissue",
            "source": r"\citet[Table~2]{Ogle2009TP} (pine parameter)",
        },
    },
    # sapwood depth/width paramters
    "SW_a": {
        "value": 18.8,
        "unit": "",
        "info": {
            "descr": "numerator parameter for sapwood width model",
            "source": r"\citet[Eq.~\(2\)]{Sellin1994CJFE}",
        },
    },
    "SW_b": {
        "value": 60.0,
        "unit": "",
        "info": {
            "descr": "denominator parameter for sapwood width model",
            "source": r"\citet[Eq.~\(2\)]{Sellin1994CJFE}",
        },
    },
    "HW_slope": {
        "value": 24.0 / 50.0,
        "unit": "",
        "info": {
            "descr": "slope value for heartwood width line",
            "source": r"\citet[Fig.~1]{Sellin1994CJFE}",
        },
    },
    # wood density parameters
    #    "rho_W0": {
    #        "value": 600_000,
    #        "unit": "g_dw/m^3",
    #        "info": {
    #            "descr": r"initial wood density, used for $\dbh<\dbh_M$",
    #            "source": r"\citet[Table~1]{Pukkala2014FPE}",
    #        },
    #    },
    "rho_Wmax": {
        "value": 550_000,
        "unit": "g_dw/m^3",
        "info": {
            "descr": r"maximum density of newly produced sapwood",
            "source": r"computed to keep $\delta_W$ positive",
        },
    },
    "rho_Wmin": {
        "value": 280_000,
        "unit": "g_dw/m^3",
        "info": {
            "descr": "minimum wood density",
            "source": "empirical parameter after some testing",
        },
    },
    "dbh_M": {
        "value": 4.0,
        "unit": "cm",
        "info": {
            "descr": (
                r"for $\dbh<\dbh_M$ the allometrically derived wood density "
                r"is assumed to be useless"
            ),
            "source": "empirical parameter after some testing",
        },
    },
    # labile carbon storage parameters
    "delta_L": {
        "value": 0.11,
        "unit": "g_gluc/g_dw",
        "info": {
            "descr": "labile carbon storage capacity of leaves",
            "source": r"\citet[Table~2]{Ogle2009TP} (pine parameter)",
        },
    },
    "delta_R": {
        "value": 0.08,
        "unit": "g_gluc/g_dw",
        "info": {
            "descr": "labile carbon storage capacity of fine roots",
            "source": r"\citet[Table~2]{Ogle2009TP} (pine parameter)",
        },
    },
}

delta_L = species_params["spruce"]["delta_L"]
delta_L = Q_(delta_L["value"], delta_L["unit"])
delta_R = species_params["spruce"]["delta_R"]
delta_R = Q_(delta_R["value"], delta_R["unit"])
delta_W = delta_L  # from simulation experience

species_params["spruce"].update(
    {
        # tissue construction parameters
        "C_gL": {
            #        "value": 1.51,
            "value": _compute_construction_costs_from_growth_respiration_rate(
                28.0 / 15 * 0.25, delta_L
            ),  # 2.44
            "unit": "g_gluc/g_dw",
            "info": {
                "descr": "construction costs of producing leaves",
                #            "source": "Ogle2009TP, Table~2"
                "source": (
                    r"\citet[p.878]{Ryan1997JGR} states that leaf construction "
                    r"costs were $28/15 \cdot 0.25$ (of leaf NPP)"
                ),
            },
        },
        "C_gR": {
            #        "value": 1.30,
            # Ryan 1997, Table 4
            # the desired root respiration rate is 3 times as high as in
            # current simulations, mR=0.75 g_gluc/g_dw/yr = 0.6/yr
            # could be increased, but
            # the construction respiration rate is 0.035/yr and so there is
            # more potential. Since maintenance
            # respired carbon and f_R_times_E are almost equal in size, we
            # multiply the current growth respiration rate by 6 and compute
            # the associated construction costs
            "value": _compute_construction_costs_from_growth_respiration_rate(
                Q_((3.9 + 5.4) / 2 * micromolCO2_TO_gC, "gC/kgC/s").to("kgC/kgC/yr")
                * Q_(1, "yr")
                / 0.6
                * 2
                * _compute_growth_respiration_rate_from_construction_costs(
                    Q_(1.30, "g_gluc/g_dw"), delta_R
                ),
                delta_R,
            ),  # 1.6
            "unit": "g_gluc/g_dw",
            "info": {
                "descr": "construction costs of producing fine roots",
                #            "source": "Ogle2009TP, Table~2"
                "source": r"\citet[Table~4]{Ryan1997JGR} and some empirical adaptation",
            },
        },
        "C_gHW": {
            "value": 1.0,  # keep at 1!!!, we only need it to correct for units
            #        "value": 1.251, # ratio between g_dw and g_gluc, otherwise 1 g_gluc
            # produces 1 g_dw in C, but there is more carbon in g_dw
            "unit": "g_gluc/g_dw",
            "info": {
                "descr": (
                    "construction costs of converting heartwood from "
                    "labile sapwood (actually: no costs)"
                ),
                "source": r"missing in \citet{Ogle2009TP} (causing a unit mismatch)",
            },
        },
        "C_gW": {
            #        "value": 1.47,
            "value": (Q_(1 + 0.76, "gC/gC") / zeta_gluc * zeta_dw).magnitude,  # 2.2018
            #            "value": (
            #                Q_(1 + (0.24 + 0.25) / 2, "gC/gC") / zeta_gluc * zeta_dw
            #            ).magnitude,  # 1.557
            #            "value": 2.2018,
            "unit": "g_gluc/g_dw",
            "info": {
                "descr": "construction costs of producing sapwood",
                #            "source": "Ogle2009TP, Table~2"
                #            "source": ("Lavigne1997TP, Table~5, northern values; we add 1 because "
                #                       "for us growth is not part of the factor to multiply with")
                #                "source": (
                #                    "We use the pine value because the spruce value seems "
                #                    "ridiculously high and makes growth respiration explode."
                #                ),
                #                "source": r"\citet{Lavigne1997TP}",
                #            "source": "Ogle2009TP, Table~2"
                "source": (
                    r"\citet[Table~5, northern]{Lavigne1997TP}, we add $1.0$ "
                    r"because for us growth is not part of the factor to multiply with"
                ),
            },
        },
    }
)

## spruce leaf respriation correction for SLA with respect to pine
# q = species_params["pine"]["SLA"]["value"] / species_params["spruce"]["SLA"]["value"]
# species_params["spruce"]["R_mL"]["value"] *= q
# species_params["spruce"]["C_gL"]["value"] *= q


# species_params["birch"]: OnespeciesParams = {
#    # respiration parameters
#    "R_mL":
#    {
#        "value": 0.95,
#        "unit": "g_gluc/g_dw/yr",
#        "info": {
#            "descr": "maintenance respiration rate of leaves",
#            "source": r"\citet[Table 2]{Ogle2009TP}"
#        }
#    },
#
#    "R_mR":
#    {
#        "value": 0.75,
#        "unit": "g_gluc/g_dw/yr",
#        "info": {
#            "descr": "maintenance respiration rate of fine roots",
#            "source": r"\citet[Table 2]{Ogle2009TP}"
#        }
#    },
#
#   "R_mS":
#    {
#        "value": 0.025,
#        "unit": "g_gluc/g_dw/yr",
#        "info": {
#            "descr": "maintenance respiration rate of sapwood",
#            "source": "Ogle2009TP, Table 2"
#        }
#    },
#
#    # senescence parameters
#    "S_L":
#    {
#        "value": 1.0,
#        "unit": "1/yr",
#        "info": {
#            "descr": "senescence rate of leaves",
#            "source": "Pukkala2014FPE, Table 2"
#        }
#    },
#
#    "S_R":
#    {
#        "value": 1.0,
#        "unit": "1/yr",
#        "info": {
#            "descr": "senescence rate of fine roots",
#            "source": "Pukkala2014FPE, Table 2"
#        }
#    },
#
#    "S_O":
#    {
#        "value": 0.0135,
#        "unit": "1/yr",
#        "info": {
#            "descr": "senescence rate of coarse roots and branches",
#            "source": "Pukkala2014FPE, Table 2"
#        }
#    },
#
#    # root shape and allometric parameters
##    "rho_R":
##    {
##        "value": 200_000,
##        "unit": "g_dw/m^3",
##        "info": {
##            "descr": "tissue density of fine roots",
##            "source": "Ogle2009TP, Table 2"
##        }
##    },
##
##    "r_R":
##    {
##        "value": 0.00027,
##        "unit": "m",
##        "info": {
##            "descr": "average fine root radius",
##            "source": "Ogle2009TP, Table 2"
##        }
##    },
##
##    "f_1":
##    {
##        "value": 4.0,
##        "unit": "",
##        "info": {
##            "descr": "fine root to leaf area ratio",
##            "source": "Ogle2009TP, Table 2"
##        }
##    },
#
#    # fine root lo leaf biomass ratio
#    "rho_RL":
#    {
#        "value": 0.67,
#        "unit": "",
#        "info": {
#            "descr": "fine root to foliage biomass ratio",
#            "source": "Pukkala2014FPE, Table 2"
#        }
#    },
#
#    # trunk shape parameters
#    "eta_B":
#    {
#        "value": 0.045,
#        "unit": "",
#        "info": {
#            "descr": (
#                "relative height at which trunk transitions from a neiloid "
#                "to a paraboloid"
#            ),
#            "source": "Ogle2009TP, Table 2 (pine parameter)"
#        }
#    },
#
#    "eta_C":
#    {
#        "value": 0.71,
#        "unit": "",
#        "info": {
#            "descr": (
#                "relative height at which trunk transitions from a paraboloid "
#                "to a cone"
#            ),
#            "source": "Ogle2009TP, Table 2 (called eta) (pine parameter)"
#        }
#    },
#
#    # sapwood texture parameters
#    "gamma_X":
#    {
#        "value": 0.62,
#        "unit": "",
#        "info": {
#            "descr": "xylem conducting area to sapwood area ratio",
#            "source": "Ogle2009TP, Table 2"
#        }
#    },
#
#    "gamma_C":
#    {
#        "value": 265_000,
#        "unit": "g_gluc/m^3",
#        "info": {
#            "descr": "maximum storage capacity of living sapwood cells",
#            "source": "Ogle2009TP, Table 2"
#        }
#    },
#
#    "gamma_W":
#    {
#        "value": 6.67e-07,
#        "unit": "m^3/g_dw",
#        "info": {
#            "descr": "(inverse) density of sapwood structural tissue",
#            "source": "Ogle2009TP, Table 2"
#        }
#    },
#
##    "SW_max":
##    {
##        "value": 0.06,
##        "unit": "m",
##        "info": {
##            "descr": "maximum sapwood width",
##            "source": "Ogle2009TP, Table 2"
##        }
##    },
#
#    # sapwood depth paramters
#    "SW_constant":
#    {
#        "value": 1.049,
#        "unit": "",
#        "info": {
#            "descr": "intercept for linear sapwood width model (pine parameter)",
#            "source": "Vanninen2005TP, Table 3"
#        }
#    },
#
#
#    "SW_H":
#    {
#        "value": -8.58e-05,
#        "unit": "",
#        "info": {
#            "descr": "height parameter for linear sapwood width model (pine parameter)",
#            "source": "Vanninen2005TP, Table 3"
#        }
#    },
#
#    "SW_A":
#    {
#        "value": -1.83e-03,
#        "unit": "",
#        "info": {
#            "descr": "tree age parameter linear sapwood width model (pine parameter)",
#            "source": "Vanninen2005TP, Table 3"
#        }
#    },
#
#    # wood density parameters
#    "rho_W0":
#    {
##        "value": 580_000, # leads to negative B_S_star at init, #TODO
#        "value": 565_000,
#        "unit": "g_dw/m^3",
#        "info": {
#            "descr": r"initial wood density, used for $\dbh<\dbh_M$",
#            "source": "Pukkala2014FPE, Table 1"
#        }
#    },
#
#    "rho_Wmin":
#    {
#        "value": 180_000,
#        "unit": "g_dw/m^3",
#        "info": {
#            "descr": "minimum wood density",
#            "source": "fiction XXX"
#        }
#    },
#
#    "dbh_M":
#    {
#        "value": 4.0,
#        "unit": "cm",
#        "info": {
#            "descr": (r"for $\dbh<\dbh_M$ the allometrically derived "
#                      r"wood density is assumed to be useless"),
#            "source": "made up by looking at graphs" # TODO
#        }
#    },
#
#    # labile carbon storage parameters
#    "delta_L":
#    {
#        "value": 0.11,
#        "unit": "g_gluc/g_dw",
#        "info": {
#            "descr": "labile carbon storage capacity of leaves",
#            "source": "Ogle2009TP, Table 2"
#        }
#    },
#
#    "delta_R":
#    {
#        "value": 0.08,
#        "unit": "g_gluc/g_dw",
#        "info": {
#            "descr": "labile carbon storage capacity of fine roots",
#            "source": "Ogle2009TP, Table 2"
#        }
#    },
#
#    # tissue construction parameters
#    "C_gL":
#    {
#        "value": 1.51,
#        "unit": "g_gluc/g_dw",
#        "info": {
#            "descr": "construction costs of producing leaves",
#            "source": "Ogle2009TP, Table 2"
#        }
#    },
#
#    "C_gR":
#    {
#        "value": 1.30,
#        "unit": "g_gluc/g_dw",
#        "info": {
#            "descr": "construction costs of producing fine roots",
#            "source": "Ogle2009TP, Table 2"
#        }
#    },
#
#    "C_gHW":
#    {
#        "value": 1.0, # keep at 1!!!, we only need it to correct for units
##        "value": 1.251, # ratio between g_dw and g_gluc, otherwise 1 g_gluc
#                        # produces 1 g_dw in C, but there is more carbon in g_dw
#        "unit": "g_gluc/g_dw",
#        "info": {
#            "descr": (
#                "construction costs of converting heartwood from "
#                "labile sapwood (actually: no costs)"
#            ),
#            "source": "missing in Ogle2009TP (unit mismatch)"
#        }
#    },
#
#    "C_gW":
#    {
#        "value": 1.47,
#        "unit": "g_gluc/g_dw",
#        "info": {
#            "descr": "construction costs of producing sapwood",
#            "source": "Ogle2009TP, Table 2"
#        }
#    },
# }


# TODO: find species-specific parameters for birch
# species_params["birch"] = species_params["pine"]


def initialize_params(one_species_params: OneSpeciesParams) -> Dict[str, Any]:
    """Initialize the parameter dictionary.

    Args:
        one_species_params: tree species parameters (for one species)

    Returns:
        Dictionary:

        - name: value
    """
    params: Dict[str, Any] = dict()
    for name, d in one_species_params.items():
        params[name] = Q_(d["value"], d["unit"])

    return params


def species_params_to_latex_table(
    species_name: str, custom_species_params: SpeciesParams = None
) -> str:
    """Create a LaTeX table from the parameters of one species.

    Args:
        species_name: element of ``custom_species_params.keys()``
        custom_species_params: tree species parameters (for all species)

    Returns:
        LaTeX parameter table
    """
    ignore_list: List[str] = ["alpha"]

    species_str = {"pine": "Scots pine", "spruce": "Norway spruce", "birch": "Birch"}

    table_head = [
        r"\begin{table}[H]",
        r"\tiny",
        r"\begin{tabular}{lllp{4cm}p{4cm}}",
        r"\multicolumn{2}{l}{\textbf{" + f"{species_str[species_name]}" + r"}}\\",
        r"\thead{ }\\",
        r"\thead{Symbol} & \thead{Value} & \thead{Unit} & \thead{Description} & \thead{Source}\\",
    ]

    if custom_species_params is None:
        custom_species_params = species_params

    rows = []
    for name, d in custom_species_params[species_name].items():
        if name in ignore_list:
            continue

        name = ["$"] + [latex(Symbol(name))] + ["$"]  # type: ignore
        name = "".join(name)

        latex_dict = {
            "epsilon": "varepsilon",
            "SW": r"\text{SW}",
            "HW": r"\text{HW}",
            "_{slope}": r"_{\text{slope}}",
            "_{max}": r"_{\text{max}}",
            "SLA": r"\mathrm{SLA}",
            #            "BH": r"\mathrm{BH}"
            "_{mL}": r"_{\text{mL}}",
            "_{mR}": r"_{\text{mR}}",
            "_{mS}": r"_{\text{mS}}",
            "_{RL}": r"_{\text{RL}}",
            "_{constant}": r"_{\text{constant}}",
            "_{W0}": r"_{W_0}",
            "_{Wmax}": r"_{W_\text{max}}",
            "_{Wmin}": r"_{W_\text{min}}",
            "_{gL}": r"_{\text{gL}}",
            "_{gR}": r"_{\text{gR}}",
            "_{gHW}": r"_{\text{gHW}}",
            "_{gW}": r"_{\text{gW}}",
            "dbh": r"\dbh",
        }
        for old, new in latex_dict.items():
            name = name.replace(old, new)

        unit = latexify_unit(d["unit"])

        val = d["value"]
        if "e" in str(val) or (val < 1e-02) or (val > 1e05):
            val_str = f"{val:0.3e}"
        elif val == int(val):
            val_str = str(val)
        else:
            val_str = f"${d['value']:0.3f}$"

        row = (
            " & ".join([name, val_str, unit, d["info"]["descr"], d["info"]["source"]])
            + "\\\\"
        )
        rows.append(row)

    s = table_head + rows + [r"\end{tabular}"]
    s += [r"\caption{" + f"{species_str[species_name]} parameters." + r"}"]
    s += [r"\label{table:" + f"{species_name}_params" + r"}"]
    s += [r"\end{table}"]
    return "\n".join(s)


def params_to_latex_table(
    speciess: List[str], custom_species_params: SpeciesParams = None
) -> str:
    """Create a LaTeX tables from the parameters of a list of species.

    Args:
        speciess: elements of ``custom_species_params.keys()``
        custom_species_params: tree species parameters (for all species)

    Returns:
        LaTeX parameter table
    """
    s: List[str] = []
    for species in speciess:
        s += [species_params_to_latex_table(species, custom_species_params)]

    return "\n\n".join(s)


###############################################################################


if __name__ == "__main__":
    print(params_to_latex_table(["pine", "spruce"]))
