# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.abspath("../../src/"))


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "BFMM"
copyright = "2023, Holger Metzler, Samuli Launiainen, Viulia Vico"
author = "Holger Metzler, Samuli Launiainen, Viulia Vico"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints",
    "sphinx.ext.inheritance_diagram",
    "nbsphinx",
]

# activate autosummary
autosummary_generate = True

# make type hints work in documentation
autodoc_typehints = "description"

# show functions in the same order as in the source code
autodoc_member_order = "bysource"

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

templates_path = ["_templates"]
exclude_patterns = ["BFMM.productivity.*.rst"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

html_title = "BFMM"
