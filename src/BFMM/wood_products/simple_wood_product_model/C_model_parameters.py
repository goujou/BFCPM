"""
Parameters for :class:`~.C_model.SimpleWoodProductModel`.

Furthermore, there is a function to make a LaTeX table for the parameters.
"""
from __future__ import annotations

from typing import Any, Dict

from sympy import Symbol, latex

from ... import Q_
from ...utils import latexify_unit

params = {
    "k_S": {
        "value": 0.3,
        "unit": "1/yr",
        "info": {
            "descr": "turnover time of short-term wood products",
            "source": r"\citet[Table 4]{Pukkala2014FPE}",
        },
    },
    "k_L": {
        "value": 1 / 50.0,
        "unit": "1/yr",
        "info": {
            "descr": "turnover time of long-term wood products",
            "source": r"\citet[Table 4]{Pukkala2014FPE}",
        },
    },
}
"""Model parameters."""


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
    ignore_list: list[str] = ["S_0", "L_0"]

    table_head = [
        r"\begin{table}",
        r"\tiny",
        r"\begin{tabular}{lllp{4cm}p{4cm}}",
        r"\thead{ }\\",
        r"\thead{Symbol} & \thead{Value} & \thead{Unit} & \thead{Description} & \thead{Source}\\",
    ]
    rows = []
    for name, d in params.items():
        if name in ignore_list:
            continue

        name = ["$"] + [latex(Symbol(name))] + ["$"]
        name = "".join(name)

        latex_dict = {}
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
    s += [r"\caption{" + "Wood product C module parameters." + r"}"]
    s += [r"\label{table:" + "wood_product_C_params" + r"}"]
    s += [r"\end{table}"]
    print("\n".join(s))


###############################################################################


if __name__ == "__main__":
    params_to_latex_table()
