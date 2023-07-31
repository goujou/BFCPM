"""
This module contains the relevant variables of the :class:`.single_tree_allocation.SingleTree` class.
"""
from typing import Dict, List, Union

import pint
from sympy import Symbol, latex

from .. import Q_
from ..utils import latexify_unit

var_infos: Dict[str, Dict[str, Union[str, Dict[str, str]]]] = {
    "r": {
        "target_unit": "m",
        "info": {
            "descr": "tree radius at trunk base",
            "source": r"Section~\ref{appendix:tree_allometries}",
        },
    },
    "Delta_r": {
        "target_unit": "m",
        "info": {
            "descr": "change of tree radius at trunk base",
            "source": "dynamically solved for",
        },
    },
    "r_BH": {
        "target_unit": "m",
        "info": {
            "descr": "radius at breast height",
            "source": r"\citet[SI, Eq.~(24)]{Ogle2009TP}",
        },
    },
    "dbh": {
        "target_unit": "cm",
        "info": {
            "descr": "tree radius at breast height",
            "source": r"Section~\ref{appendix:tree_allometries}",
        },
    },
    "H": {
        "target_unit": "m",
        "info": {
            "descr": "tree height",
            "source": r"Eq.~\eqref{eqn:H}, \citet{naslund1947funktioner, siipilehto2015naslundin}",
        },
    },
    "GPP": {
        "target_unit": "g_gluc/yr",
        "manuscript_unit": "gC/yr",
        "info": {
            "descr": r"carbon uptake by photosynthesis",
            "source": "-",
        },
    },
    "C_alloc": {
        "target_unit": "g_gluc/yr",
        "manuscript_unit": "gC/yr",
        "info": {
            "descr": "available gC/yr for allocation to tree organs",
            "source": r"$E/\Delta t-R_M$",
        },
    },
    "R_M": {
        "target_unit": "g_gluc/yr",
        "manuscript_unit": "gC/yr",
        "info": {
            "descr": "whole plant maintenance respiration",
            #            "source": r"\citet[SI, Eq.~(28)]{Ogle2009TP}",
            "source": r"$M_L+M_R+M_S$",
        },
    },
    "M_L": {
        "target_unit": "g_gluc/yr",
        "manuscript_unit": "gC/yr",
        "info": {
            "descr": "maintenance respiration leaves",
            #            "source": r"\citet[SI, Eq.~(28)]{Ogle2009TP}",
            "source": r"Eq.~\eqref{eqn:M_L}",
        },
    },
    "M_R": {
        "target_unit": "g_gluc/yr",
        "manuscript_unit": "gC/yr",
        "info": {
            "descr": "maintenance respiration fine roots",
            #            "source": r"\citet[SI, Eq.~(28)]{Ogle2009TP}",
            "source": r"analogous to $M_L$",
        },
    },
    "M_S": {
        "target_unit": "g_gluc/yr",
        "manuscript_unit": "gC/yr",
        "info": {
            "descr": "maintenance respiration sapwood",
            #            "source": r"\citet[SI, Eq.~(28)]{Ogle2009TP}",
            "source": r"Eq.~\eqref{eqn:M_S}",
        },
    },
    "G_L": {
        "target_unit": "g_gluc/yr",
        "manuscript_unit": "gC/yr",
        "info": {
            "descr": "growth respiration leaves",
            "source": r"Eq.~\eqref{eqn:G_L}",
        },
    },
    "G_R": {
        "target_unit": "g_gluc/yr",
        "manuscript_unit": "gC/yr",
        "info": {
            "descr": "growth respiration fine roots",
            "source": r"analogous to $G_L$",
        },
    },
    "G_OS_E": {
        "target_unit": "g_gluc/yr",
        "manuscript_unit": "gC/yr",
        "info": {
            "descr": "growth respiration sapwood from transient C",
            "source": r"Eq.~\eqref{eqn:G_OS_E}",
        },
    },
    "G_OS_CS": {
        "target_unit": "g_gluc/yr",
        "manuscript_unit": "gC/yr",
        "info": {
            "descr": "growth respiration sapwood from labile storage C",
            "source": r"Eq.~\eqref{eqn:G_OS_CS}",
        },
    },
    "eta_L": {
        "target_unit": "",
        "info": {
            "descr": "CUE during leaf tissue growth",
            "source": r"Eq.~\eqref{eqn:eta_L}",
        },
    },
    "eta_R": {
        "target_unit": "",
        "info": {
            "descr": "CUE during fine root tissue growth",
            "source": r"analogous to $\eta_L$",
        },
    },
    "eta_W": {
        "target_unit": "",
        "info": {
            "descr": "CUE during sapwood production",
            "source": r"Eq.~\eqref{eqn:eta_W}",
        },
    },
    "eta_HW": {
        "target_unit": "",
        "info": {
            "descr": "CUE during heartwood production",
            "source": r"fixed to $1$",
        },
    },
    "H_TH": {
        "target_unit": "m",
        "info": {
            "descr": "height of trunk heartwood section",
            "source": r"\citet[SI, Eq.~(9)]{Ogle2009TP}, corrected and introduced capturing of equalities",
        },
    },
    "h_B": {
        "target_unit": "m",
        "info": {
            "descr": (
                "height at which trunk transitions from a neiloid to a " "paraboloid"
            ),
            "source": r"\citet[SI, p.10]{Ogle2009TP}",
        },
    },
    "h_C": {
        "target_unit": "m",
        "info": {
            "descr": (
                "height at which trunk transitions from a paraboloid to a " "cone"
            ),
            "source": r"\citet[SI, p.10]{Ogle2009TP}",
        },
    },
    "r_B": {
        "target_unit": "m",
        "info": {
            "descr": (
                "radius at which trunk transitions from a neiloid to a " "paraboloid"
            ),
            "source": r"\citet[SI, Eq.~(10a)]{Ogle2009TP}",
        },
    },
    "r_C": {
        "target_unit": "m",
        "info": {
            "descr": (
                "radius at which trunk transitions from a paraboloid to a " "cone"
            ),
            "source": r"\citet[SI, Eq.~(10b)]{Ogle2009TP}",
        },
    },
    #    "SA": {
    #        "target_unit": "m^2",
    #        "descr": "cross-sectional area of sapwood ad trunk base",
    #        "source": "Ogle2009TP_SI, Eq.~(16)",
    #    },
    #    "XA": {
    #        "target_unit": "m^2",
    #        "descr": "total cross-sectional area of xylem conduit lumen at base",
    #        "source": "Ogle2009TP_SI, Eq.~(17)",
    #    },
    "LA": {
        "target_unit": "m^2",
        "info": {
            "descr": "total leaf area",
            "source": r"$\text{SLA}\,B_L$",
        },
    },
    #    "H_crown": {
    #        "target_unit": "m",
    #        "descr": "Height of crown base over ground",
    #        "source": "",
    #    },
    #    "R_C_base": {
    #        "target_unit": "m",
    #        "descr": "crown radius at base of crown",
    #        "source": "Ogle2009TP_SI, Eq.~(22)",
    #    },
    #    "R_C_max": {
    #        "target_unit": "m",
    #        "descr": ("maximum potential crown radius at distance m*H from " "top of tree"),
    #        "source": "Ogle2009TP_SI, Eq.~(23)",
    #    },
    #    "LAI": {
    #        "target_unit": "m^2/m^2",
    #        "descr": "leaf area index",
    #        "source": "Ogle2009TP_SI, adapted from Eq.~(26)",
    #    },
    "B_L": {
        "target_unit": "g_dw",
        "manuscript_unit": "gC",
        "info": {
            "descr": "biomass of leaves",
            "source": "-",
        },
    },
    #    "RA": {
    #        "target_unit": "m^2",
    #        "descr": "total fine root area",
    #        "source": "Ogle2009TP_SI, Eq.~(18A)",
    #    },
    "B_R": {
        "target_unit": "g_dw",
        "manuscript_unit": "gC",
        "info": {
            "descr": "biomass of fine roots",
            "source": "-",
        },
    },
    "V_neiloid": {
        "target_unit": "m^3",
        "info": {
            "descr": "trunk volume (base)",
            "source": r"\citet[SI, Eq.~(9, line 1)]{Ogle2009TP}",
        },
    },
    "V_paraboloid": {
        "target_unit": "m^3",
        "info": {
            "descr": "trunk volume (middle)",
            "source": r"\citet[SI, Eq.~(9, line 2)]{Ogle2009TP}",
        },
    },
    "V_cone": {
        "target_unit": "m^3",
        "info": {
            "descr": "trunk volume (top)",
            "source": r"\citet[SI, Eq.~(9, line 3)]{Ogle2009TP}",
        },
    },
    "V_T": {
        "target_unit": "m^3",
        "info": {
            "descr": "trunk volume",
            "source": r"\citet[SI, Eq.~(9)]{Ogle2009TP}",
        },
    },
    "V_TH": {
        "target_unit": "m^3",
        "info": {
            "descr": "volume of trunk heartwood section",
            "source": r"\citet[SI, Eq.~(14)]{Ogle2009TP}, introduced capturing of equalities",
        },
    },
    "V_TS": {
        "target_unit": "m^3",
        "info": {
            "descr": "volume of trunk sapwood",
            "source": r"\citet[SI, Eq.~(15)]{Ogle2009TP}",
        },
    },
    "SW": {
        "target_unit": "m",
        "info": {
            "descr": "width (or depth) of sapwood at trunk base",
            "source": r"Section~\ref{appendix:f_T}, \citet{Helmisaari2007TP, Sellin1994CJFE}",
        },
    },
    "B_TS": {
        "target_unit": "g_dw",
        "manuscript_unit": "gC",
        "info": {
            "descr": "biomass of trunk sapwood",
            "source": r"adapted from \citet[SI, Eq.~(1C)]{Ogle2009TP}",
        },
    },
    "B_OS": {
        "target_unit": "g_dw",
        "manuscript_unit": "gC",
        "info": {
            "descr": r"biomass of ``other'' sapwood",
            "source": r"\citet[SI, Eq.~(20A)]{Ogle2009}",
        },
    },
    "C_S_star": {
        "target_unit": "g_gluc",
        "info": {
            "descr": "maximum amount of labile carbon stored in sapwood",
            "source": r"\citet[SI, Eq.~(5)]{Ogle2009TP}",
        },
    },
    "B_S_star": {
        "target_unit": "g_dw",
        "info": {
            "descr": "biomass of 'living' sapwood",
            "source": r"\citet[SI, Eq.~(29)]{Ogle2009TP}",
        },
    },
    "B_S": {
        "target_unit": "g_dw",
        "manuscript_unit": "gC",
        "info": {
            "descr": "biomass of bulk sapwood",
            "source": r"$B_\text{OS}+B_\text{TS}$",
        },
    },
    "delta_S": {
        "target_unit": "g_gluc/g_dw",
        "info": {
            "descr": "concentration of labile carbon storage of bulk sapwood",
            "source": r"Eq.~\eqref{eqn:delta_S}, \citet[SI, Eq.~(7)]{Ogle2009TP}",
        },
    },
    "rho_W": {
        "target_unit": "g_dw/m^3",
        "info": {
            "descr": "density of newly produced sapwood",
            "source": r"Eq.~\eqref{eqn:rho_W}",
        },
    },
    "delta_W": {
        "target_unit": "g_gluc/g_dw",
        "info": {
            "descr": (
                "maximum labile carbon storage capacity of newly produced sapwood"
            ),
            "source": r"Eq.~\eqref{eqn:delta_W}, \citet[SI, Eq.~(6)]{Ogle2009TP}",
        },
    },
    #    "B_O":
    #    {
    #        "target_unit": "g_dw",
    #        "descr": r"biomass of ``other'' wood (coarse roots, branches)",
    #        "source": "Ogle2009TP_SI, Eq.~(20A)"
    #    },
    "B_OH": {
        "target_unit": "g_dw",
        "info": {
            "descr": r"biomass of ``other'' heartwood",
            "source": r"\citet[SI, Eq.~(20B)]{Ogle2009TP}",
        },
    },
    "B_TH": {
        "target_unit": "g_dw",
        "manuscript_unit": "gC",
        "info": {
            "descr": "biomass of trunk heartwood",
            "source": r"adapted from \citet[SI, Eq.~(1D)]{Ogle2009TP}",
        },
    },
    "B_T": {
        "target_unit": "g_dw",
        "manuscript_unit": "gC",
        "info": {
            "descr": "biomass of trunk",
            "source": r"$B_{\text{TH}} + B_{\text{TS}} + \frac{B_{\text{TS}}}{B_S}\,C_S$",
        },
    },
    "m_X": {
        "target_unit": "g_dw",
        "info": {
            "descr": "allometrically derived biomass of tree organ $X$",
            "source": r"based on Eq.~\eqref{eqn:allometric_biomass}",
        },
    },
    "lambda_S": {
        "target_unit": "",
        "info": {
            "descr": r"ratio of ``other'' sapwood to trunk sapwood",
            "source": r"Eq.~\eqref{eqn:lambda_S}",
        },
    },
    "lambda_H": {
        "target_unit": "",
        "info": {
            "descr": r"ratio of ``other'' heartwood to trunk heartwood",
            "source": r"Eq.~\eqref{eqn:lambda_S}",
        },
    },
    "C_L": {
        "target_unit": "g_gluc",
        "manuscript_unit": "gC",
        "info": {
            "descr": "labile carbon in leaves",
            "source": r"adapted from \citet[SI, Eq.~(4)]{Ogle2009TP}",
        },
    },
    "C_R": {
        "target_unit": "g_gluc",
        "manuscript_unit": "gC",
        "info": {
            "descr": "labile carbon in roots",
            "source": r"adapted from \citet[SI, Eq.~(4)]{Ogle2009TP}",
        },
    },
    "C_S": {
        "target_unit": "g_gluc",
        "manuscript_unit": "gC",
        "info": {
            "descr": "labile carbon stored in bulk sapwood",
            "source": r"\citet[SI, Eq.~(3)]{Ogle2009TP}",
        },
    },
    "v_T": {
        "target_unit": "1/yr",
        "info": {
            "descr": "sapwood to heartwood conversion rate of trunk",
            "source": r"Eq.~\eqref{eqn:v_T}, \citet[SI, Eq.~(2)]{Ogle2009TP}",
        },
    },
    "v_O": {
        "target_unit": "1/yr",
        "info": {
            "descr": (
                "sapwood to heartwood conversion rate of " "coarse roots and branches"
            ),
            "source": r"Eq.~\eqref{eqn:v_O}, \citet[SI, Eq.~(1F)]{Ogle2009TP}",
        },
    },
    "f_L_times_E": {
        "target_unit": "g_gluc/yr",
        "manuscript_unit": "gC/yr",
        "info": {
            "descr": "flux rate from transient pool to leaves",
            "source": r"\citet[SI, Eq.~(1A)]{Ogle2009TP}",
        },
    },
    "f_L": {
        "target_unit": "",
        "info": {
            "descr": "partitioning from transient pool to leaves",
            "source": r"Section~\ref{appendix:f_L}, \citet[SI, Eq.~(1A)]{Ogle2009TP}",
        },
    },
    "f_R_times_E": {
        "target_unit": "g_gluc/yr",
        "manuscript_unit": "gC/yr",
        "info": {
            "descr": "flux rate from transient pool to fine roots",
            "source": r"\citet[SI, Eq.~(1B)]{Ogle2009TP}",
        },
    },
    "f_R": {
        "target_unit": "",
        "info": {
            "descr": "partitioning from transient pool to fine roots",
            "source": r"Section~\ref{appendix:f_R}, \citet[SI, Eq.~(1B)]{Ogle2009TP}",
        },
    },
    "f_T_times_E": {
        "target_unit": "g_gluc/yr",
        "manuscript_unit": "gC/yr",
        "info": {
            "descr": "flux rate from transient pool to trunk",
            "source": r"\citet[SI, Eq.~(31C)]{Ogle2009TP}",
        },
    },
    "f_T": {
        "target_unit": "",
        "info": {
            "descr": "partitioning from transient pool to trunk",
            "source": r"Section~\ref{appendix:f_T}, \citet[SI, Eq.~(31C)]{Ogle2009TP}",
        },
    },
    "f_O_times_E": {
        "target_unit": "g_gluc/yr",
        "manuscript_unit": "gC/yr",
        "info": {
            "descr": "flux rate from transient pool to coarse roots and branches",
            "source": r"\citet[SI, Eq.~(1E)]{Ogle2009TP}",
        },
    },
    "f_O": {
        "target_unit": "",
        "info": {
            "descr": "partitioning from transient pool to coarse roots and branches",
            "source": r"Section~\ref{appendix:f_O}, \citet[SI, Eq.~(1E)]{Ogle2009TP}",
        },
    },
    "E": {
        "target_unit": "g_gluc",
        "manuscript_unit": "gC",
        "info": {
            "descr": "transient carbon as glucose",
            "source": "-",
        },
    },
    "f_CS_times_CS": {
        "target_unit": "g_gluc/yr",
        "manuscript_unit": "gC/yr",
        "info": {
            "descr": r"flux from $C_S$ pool to $B_{\text{OS}}$ in static and shrinking state",
            "source": "-",
        },
    },
    "f_CS": {
        "target_unit": "",
        "info": {
            "descr": r"fraction of $C_S$ used to regrow ``other'' sapwood",
            "source": r"Eq.~\eqref{eqn:f_C_S}",
        },
    },
}
"""Information on tree variables."""


def assign(name: str, v: Q_, target_unit: str = None) -> Q_:
    """Assign a value to a variable, use the correct unit.

    The unit is taken from :obj:`~.single_tree_vars.var_infos`.

    Args:
        name: variable name (must be contained in ``var_infos``)
        v: value to assign to the variable
        target_unit: if provided, forced on the return value,\
            otherwise unit is taken from ``var_infos``.

    Returns:
        variable with correct unit
    """
    if target_unit is None:
        target_unit = var_infos[name]["target_unit"]  # type: ignore

    if not isinstance(v, pint.Quantity):
        v = Q_(v, target_unit)

    v = v.to(target_unit)
    return v


def vars_to_latex_table() -> str:
    """Create a LaTeX table from the variable information."""
    ignore_list: List[str] = [
        # ignore state variables
        "E",
        "B_L",
        "C_L",
        "B_R",
        "C_R",
        "C_S",
        "B_OS",
        "B_OH",
        "B_TS",
        "B_TH",
    ]

    # other variables not used in the manuscript
    ignore_list += [
        "f_L_times_E",
        "f_R_times_E",
        "f_O_times_E",
        "f_T_times_E",
        "f_CS_times_CS",
        "V_neiloid",
        "V_paraboloid",
        "V_cone",
        "h_B",
        "h_C",
        "r_B",
        "r_C",
    ]

    table_head = [
        r"\begin{table}[H]",
        r"\tiny",
        r"\begin{tabular}{llp{5cm}p{6cm}}",
        r"\thead{ }\\",
        r"\thead{Symbol} & \thead{Unit} & \thead{Description} & \thead{Source}\\",
    ]

    rows = []
    for name, d in var_infos.items():
        if name in ignore_list:
            continue

        name = ["$"] + [latex(Symbol(name))] + ["$"]  # type: ignore
        name = "".join(name)

        latex_dict = {
            "G_{OS E}": r"G_{\text{OS},E}",
            "G_{OS CS}": r"G_{\text{OS},C_S}",
            "f_{CS}": "f_{C_S}",
            "_{r}": " r",
            "_{OS}": r"_{\text{OS}}",
            "_{OH}": r"_{\text{OH}}",
            "_{TS}": r"_{\text{TS}}",
            "_{TH}": r"_{\text{TH}}",
            "_{CS}": r"_{\text{CS}}",
            "SW": r"\text{SW}",
            "HW": r"\text{HW}",
            "dbh": r"\dbh",
            "LA": r"\text{LA}",
            "neiloid": r"\text{neiloid}",
            "paraboloid": r"\text{paraboloid}",
            "cone": r"\text{cone}",
            "GPP": r"\GPP",
            "C_{S star}": r"C^\ast_S",
            "B_{S star}": r"B^\ast_S",
            "_{BH}": r"_\text{BH}",
            "C_{alloc}": r"\Calloc",
        }
        for old, new in latex_dict.items():
            name = name.replace(old, new)

        unit = latexify_unit(d.get("manuscript_unit", d["target_unit"]))  # type: ignore
        row = " & ".join([name, unit, d["info"]["descr"], d["info"]["source"]]) + "\\\\"  # type: ignore
        rows.append(row)

    s = table_head + rows + [r"\end{tabular}"]
    s += [r"\caption{" + "Tree module variables. Units are per single tree." + r"}"]
    s += [r"\label{table:" + "tree_module_vars" + r"}"]
    s += [r"\end{table}"]
    return "\n".join(s)


###############################################################################


if __name__ == "__main__":
    print(vars_to_latex_table())
