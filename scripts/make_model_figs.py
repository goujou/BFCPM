"""
Module to create model figures.
"""
from pathlib import Path

import bgc_md2.models.ACGCA as tree_model
import bgc_md2.models.ACGCA.source  # pylint: disable=unused-import
import bgc_md2.models.ACGCAWoodProductModel.source  # pylint: disable=unused-import
import matplotlib.pyplot as plt
from BFMM import FIGS_PATH
from BFMM.soil.dead_wood_classes.C_model import srm as SoilCDeadWoodClasses_srm
from BFMM.soil.simple_soil_model.C_model import srm as SimpleSoilCModel_srm
from BFMM.wood_products.simple_wood_product_model.C_model import \
    srm as SimpleWoodProductModel_srm


def main(black_and_white: bool):  # pylint: disable=redefined-outer-name
    """Create model figures and store them in :data:`~BFMM.FIGS_PATH`.

    Args:
        black_and_white: if False then produce colored figures
    """
    data = {
        "leaves": {
            "srm": tree_model.leaves.source.srm,
            "fontsize": 16,
        },
        "roots": {
            "srm": tree_model.roots.source.srm,
            "fontsize": 16,
        },
        "other": {
            "srm": tree_model.other.source.srm,
            "fontsize": 16,
        },
        "trunk": {
            "srm": tree_model.trunk.source.srm,
            "fontsize": 16,
        },
        "tree": {
            "srm": tree_model.source.srm,
            "fontsize": 16,
        },
        "simple_soil_model": {
            "srm": SimpleSoilCModel_srm,
            "fontsize": 16,
        },
        "soil_dead_wood_classes": {
            "srm": SoilCDeadWoodClasses_srm,
            "fontsize": 10,
        },
        "simple_wood_product_model": {
            "srm": SimpleWoodProductModel_srm,
            "fontsize": 16,
        },
    }

    for name, d in data.items():
        fig, ax = plt.subplots()
        d["srm"].plot_pools_and_fluxes(
            ax,
            fontsize=d["fontsize"],
            legend=False,
            color_fluxes=False,
            black_and_white=black_and_white,
        )

        if not black_and_white:
            bw_str = ""
        else:
            bw_str = "_bw"

        for ext in [".pdf", ".png", ".jpg"]:
            filename = FIGS_PATH.joinpath(name + bw_str + ext)
            fig.savefig(filename, dpi=500)
            print(f"Saved {filename}")

        plt.close(fig)


###############################################################################


if __name__ == "__main__":
    black_and_white = True
    main(black_and_white)
