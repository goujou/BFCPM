"""
Parameters for :class:`~.C_model.SoilCDeadWoodClasses`.

Furthermore, there is a function to make a LaTeX table for the parameters.
"""
from __future__ import annotations

from typing import Any, Dict

from sympy import Symbol, latex

from ... import Q_
from ...utils import latexify_unit

params = {
    "k_Litter": {
        "value": 0.438,
        "unit": "1/yr",
        "info": {
            "descr": "total Litter turnover rate",
            "source": r"\citet[Table~2]{Hyvonen2001CJFR}",
        },
    },
    "f_Litter": {
        "value": 0.5,
        "unit": "1/yr",
        "info": {
            "descr": "Litter respiration fraction",
            "source": r"\citet[Fig.~2]{Koven2013BGS}",
        },
    },
    "k_1": {
        "value": (7 * 0.154 + 14 * 0.07) / (7 + 14),
        "unit": "1/yr",
        "info": {
            "descr": "branches turnover rate",
            "source": r"\citet[Table~1]{Hyvonen2001CJFR}, weighted branch average",
        },
    },
    "f_1": {
        "value": 0.5,
        "unit": "1/yr",
        "info": {
            "descr": "CWD respiration fraction",
            "source": r"\citet[Fig.~2]{Koven2013BGS}",
        },
    },
    "k_2": {
        "value": 0.083,
        "unit": "1/yr",
        "info": {
            "descr": "CWD class turnover rate",
            "source": r"\citet[Table~1]{Hyvonen2001CJFR}",
        },
    },
    "f_2": {
        "value": 0.5,
        "unit": "1/yr",
        "info": {
            "descr": "CWD class respiration fraction",
            "source": r"\citet[Fig.~2]{Koven2013BGS}",
        },
    },
    "k_3": {
        "value": 0.056,
        "unit": "1/yr",
        "info": {
            "descr": "CWD class turnover rate",
            "source": r"\citet[Table~1]{Hyvonen2001CJFR}",
        },
    },
    "f_3": {
        "value": 0.5,
        "unit": "1/yr",
        "info": {
            "descr": "CWD class respiration fraction",
            "source": r"\citet[Fig.~2]{Koven2013BGS}",
        },
    },
    "k_4": {
        "value": 0.025,
        "unit": "1/yr",
        "info": {
            "descr": "CWD class turnover rate",
            "source": r"\citet[Table~1]{Hyvonen2001CJFR}",
        },
    },
    "f_4": {
        "value": 0.5,
        "unit": "1/yr",
        "info": {
            "descr": "CWD class respiration fraction",
            "source": r"\citet[Fig.~2]{Koven2013BGS}",
        },
    },
    "k_5": {
        "value": 0.014,
        "unit": "1/yr",
        "info": {
            "descr": "CWD class turnover rate",
            "source": r"\citet[Table~1]{Hyvonen2001CJFR}",
        },
    },
    "f_5": {
        "value": 0.5,
        "unit": "1/yr",
        "info": {
            "descr": "CWD class respiration fraction",
            "source": r"\citet[Fig.~2]{Koven2013BGS}",
        },
    },
    "k_6": {
        "value": 0.009,
        "unit": "1/yr",
        "info": {
            "descr": "CWD class turnover rate",
            "source": r"\citet[Table~1]{Hyvonen2001CJFR}",
        },
    },
    "f_6": {
        "value": 0.5,
        "unit": "1/yr",
        "info": {
            "descr": "CWD class respiration fraction",
            "source": r"\citet[Fig.~2]{Koven2013BGS}",
        },
    },
    "k_SOC": {
        "value": 0.023,
        "unit": "1/yr",
        "info": {
            "descr": "respiration rate SOM",
            "source": r"defined to match SOM stocks in \citet[Table~5]{Peltoniemi2004GCB}",
        },
    },
}
"""Mode parameters."""


def initialize_params() -> Dict[str, Any]:
    """Initialize the parameter dictionary.

    Returns:
        Dictionary

        - name: value
    """
    initialized_params: Dict[str, Any] = dict()
    for name, d in params.items():
        initialized_params[name] = Q_(d["value"], d["unit"])

    return initialized_params


def params_to_latex_table():
    """Create a LaTeX table from the parameters and print it to the screen."""
    ignore_list: list[str] = ["Litter_0", "CWD_0", "Soil_0"]

    table_head = [
        r"\begin{table}",
        r"\tiny",
        r"\begin{tabular}{lllp{3cm}p{5cm}}",
        r"\thead{ }\\",
        r"\thead{Symbol} & \thead{Value} & \thead{Unit} & \thead{Description} & \thead{Source}\\",
    ]
    rows = []
    for name, d in params.items():
        if name in ignore_list:
            continue

        name = ["$"] + [latex(Symbol(name))] + ["$"]
        name = "".join(name)

        latex_dict = {
            "_{Litter}": r"_{\text{Litter}}",
            "_{CWD}": r"_{\text{CWD}}",
            "_{SOM}": r"_{\text{SOM}}",
        }
        for old, new in latex_dict.items():
            name = name.replace(old, new)

        unit = latexify_unit(d["unit"])

        val = d["value"]
        if "e" in str(val) or val < 1e-02:
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
    s += [r"\caption{" + "Soil C module parameters." + r"}"]
    s += [r"\label{table:" + f"soil_C_params" + r"}"]
    s += [r"\end{table}"]
    print("\n".join(s))


###############################################################################


if __name__ == "__main__":
    params_to_latex_table()
