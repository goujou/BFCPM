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

project = "BFCPM"
copyright = "2023, Holger Metzler, Samuli Launiainen, Giulia Vico"
author = "Holger Metzler, Samuli Launiainen, Giulia Vico"
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
    #    "sphinx_toolbox.more_autodoc.genericalias",
    #    "sphinx_toolbox.more_autodoc.typehints",
    #    "sphinx_toolbox.more_autodoc.typevars",
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
exclude_patterns = ["BFCPM.productivity.*.rst"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]

html_title = "BFCPM"


#### sphinx formatting (e.g., for dictionaries) ####


from importlib import import_module
from pprint import pformat

from docutils import nodes
from sphinx import addnodes
from sphinx.ext.autodoc.directive import (DocumenterBridge,
                                          process_documenter_options)
from sphinx.util.docutils import SphinxDirective
from sphinx.util.inspect import memory_address_re


class PrettyPrintDirective(SphinxDirective):
    """Render a constant using pprint.pformat and insert into the document"""

    required_arguments = 1

    def run(self):
        # parse docstring
        reporter = self.state.document.reporter
        source, lineno = reporter.get_source_and_line(self.lineno)
        print("92", source)

        objtype = self.name[4:]
        doccls = self.env.app.registry.documenters[objtype]

        documenter_options = process_documenter_options(
            doccls, self.config, self.options
        )

        params = DocumenterBridge(
            self.env, reporter, documenter_options, lineno, self.state
        )
        documenter = doccls(params, self.arguments[0])
        documenter.parse_name()

        docstrings = documenter.get_doc()
        if docstrings is None:
            pass
        else:
            if not docstrings:
                docstrings.append([])

            for i, line in enumerate(documenter.process_doc(docstrings)):
                documenter.add_line(line, documenter.get_sourcename(), i)

        node = nodes.paragraph()
        node.document = self.state.document
        content = params.result

        # data value
        self.state.nested_parse(content, 0, node)
        module_path, member_name = self.arguments[0].rsplit("::", 1)

        member_data = getattr(import_module(module_path), member_name)

        #        if isinstance(member_data, dict):
        #            member_data_ = {str(key): value for key, value in member_data.items()}
        #            member_data = member_data_

        code = pformat(member_data, 2, width=68)
        code = memory_address_re.sub("", code)

        #        code = code.replace("{", "{\n")
        #        code = code.replace("}", "\n}")

        literal = nodes.literal_block(code, code)
        literal["language"] = "javascript"  # looks better than "python"

        return [
            addnodes.desc_name(text=member_name),
            addnodes.desc_content("", literal),
            *node.children,
        ]


def setup(app):
    app.add_directive("autodata", PrettyPrintDirective)
