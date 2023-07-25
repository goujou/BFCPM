#!/bin/bash

# check files (to some degree) for correct type hinting

PKGDIR="../src/BFMM"

# scripts
#mypy make_model_figs.py
#mypy profiling.py

# package

#mypy $PKGDIR/__init__.py
#mypy $PKGDIR/params.py
#mypy $PKGDIR/prepare_stand.py
#mypy $PKGDIR/simulation_parameters.py
#mypy $PKGDIR/stand.py
#mypy $PKGDIR/type_aliases.py
#mypy $PKGDIR/utils.py

# TODO

## trees
SUBPKGDIR="trees"

#mypy $PKGDIR/$SUBPKGDIR/__init__.py
#mypy $PKGDIR/$SUBPKGDIR/allometry.py
#mypy $PKGDIR/$SUBPKGDIR/mean_tree.py
#mypy $PKGDIR/$SUBPKGDIR/single_tree_allocation.py
#mypy $PKGDIR/$SUBPKGDIR/single_tree_C_model.py
#mypy $PKGDIR/$SUBPKGDIR/single_tree_params.py
#mypy $PKGDIR/$SUBPKGDIR/single_tree_vars.py
#mypy $PKGDIR/$SUBPKGDIR/tree_utils.py


## management
SUBPKGDIR="management"

#mypy $PKGDIR/$SUBPKGDIR/__init__.py
#mypy $PKGDIR/$SUBPKGDIR/library.py
#mypy $PKGDIR/$SUBPKGDIR/management_strategy.py

## productivity
SUBPKGDIR="productivity"

#mypy $PKGDIR/$SUBPKGDIR/__init__.py
##mypy $PKGDIR/SUBPKGDIR/constants.py
##mypy $PKGDIR/SUBPKGDIR/phenology.py
##mypy $PKGDIR/SUBPKGDIR/photo.py
##mypy $PKGDIR/SUBPKGDIR/radiation.py
##mypy $PKGDIR/SUBPKGDIR/utils.py
##mypy $PKGDIR/SUBPKGDIR/vegetation.py
##mypy $PKGDIR/SUBPKGDIR/waterbalance.py

## simulation
SUBPKGDIR="simulation"

#mypy $PKGDIR/$SUBPKGDIR/__init__.py
#mypy $PKGDIR/$SUBPKGDIR/library.py
#mypy $PKGDIR/$SUBPKGDIR/recorded_simulation.py
#mypy $PKGDIR/$SUBPKGDIR/simulation.py
#mypy $PKGDIR/$SUBPKGDIR/utils.py

## soil
SUBPKGDIR="soil"

#mypy $PKGDIR/$SUBPKGDIR/__init__.py
#mypy $PKGDIR/$SUBPKGDIR/soil_c_model_abc.py

SUBPKGDIR="soil/simple_soil_model"

#mypy $PKGDIR/$SUBPKGDIR/__init__.py
#mypy $PKGDIR/$SUBPKGDIR/C_model.py
#mypy $PKGDIR/$SUBPKGDIR/C_model_parameters.py

SUBPKGDIR="soil/dead_wood_classes"

#mypy $PKGDIR/$SUBPKGDIR/__init__.py
#mypy $PKGDIR/$SUBPKGDIR/C_model.py
#mypy $PKGDIR/$SUBPKGDIR/C_model_parameters.py

## wood products
SUBPKGDIR="wood_products"

#mypy $PKGDIR/$SUBPKGDIR/__init__.py
#mypy $PKGDIR/$SUBPKGDIR/wood_product_model_abc.py

SUBPKGDIR="wood_products/simple_wood_product_model"

#mypy $PKGDIR/$SUBPKGDIR/__init__.py
#mypy $PKGDIR/$SUBPKGDIR/C_model.py
#mypy $PKGDIR/$SUBPKGDIR/C_model_parameters.py


