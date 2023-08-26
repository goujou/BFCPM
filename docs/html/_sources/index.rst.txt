.. BFCPM documentation master file, created by
   sphinx-quickstart on Wed Jul 19 17:29:59 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Boreal Forest Carbon Path Model (BFCPM)
=======================================

... by Holger Metzler, Samuli Launiainen, and Giulia Vico

.. image:: _static/total_model_v2.png


Top-level modules
-----------------

.. autosummary::
    :toctree:
    :recursive:
    :template: module.rst

    ~BFCPM.prepare_stand
    ~BFCPM.stand
    ~BFCPM.type_aliases
    ~BFCPM.utils


Global model components
-----------------------

.. autosummary::
    :toctree:
    :recursive:
    :template: module.rst

    ~BFCPM.productivity
    ~BFCPM.trees
    ~BFCPM.soil
    ~BFCPM.wood_products

    ~BFCPM.management


Run a simulation
================

.. autosummary::
    :toctree:
    :recursive:
    :template: module.rst

    ~BFCPM.simulation
    ~BFCPM.simulation_parameters


Notes
=====

    .. [1] Launiainen, Samuli, et al. "Coupling boreal forest CO\ :sub:`2`, H\ :sub:`2`\ O 
        and energy flows by a vertically structured forest canopy–soil 
        model with separate bryophyte layer." Ecological modelling 312 
        (2015): 385-405.


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
