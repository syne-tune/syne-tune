# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

# This file has been taken from Ray. The reason for reusing the file is to be able to support the same API when
# defining search space while avoiding to have Ray as a required dependency. We may want to add functionality in the
# future.
import datetime
import os
import shutil
import sys

import syne_tune


sys.path.insert(0, os.path.abspath("."))


def run_apidoc(app):
    """Generate doc stubs using sphinx-apidoc."""
    module_dir = os.path.join(app.srcdir, "../../")
    output_dir = os.path.join(app.srcdir, "_apidoc")
    excludes = [
        "../../container*",
        "../../examples*",
        "../../githooks*",
        "../../tst*",
        "../../syne_tune.egg-info*",
        "../../benchmarking/nursery/benchmark_automl*",
    ]

    # Ensure that any stale apidoc files are cleaned up first.
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)

    cmd = [
        "--separate",
        "--module-first",
        "--doc-project=API Reference",
        "-o",
        output_dir,
        module_dir,
    ]
    cmd.extend(excludes)

    try:
        from sphinx.ext import apidoc  # Sphinx >= 1.7

        apidoc.main(cmd)
    except ImportError:
        from sphinx import apidoc  # Sphinx < 1.7

        cmd.insert(0, apidoc.__file__)
        apidoc.main(cmd)


def setup(app):
    """Register our sphinx-apidoc hook."""
    app.connect("builder-inited", run_apidoc)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Syne Tune"
version = syne_tune.__version__
release = syne_tune.__version__
copyright = f"{datetime.datetime.now().year}, Amazon"
author = (
    "Aaron Klein, David Salinas, Matthias Seeger, Martin Wistuba, Cedric Archambeau"
)

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

# Use https://github.com/tox-dev/sphinx-autodoc-typehints:
# This creates type information in docstrings from the Python `typing`
# annotations
extensions = [
    "sphinx.ext.duration",
    "sphinx.ext.doctest",
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinxcontrib.bibtex",
    "myst_parser",
    "sphinxcontrib.jquery",  # can be removed as soon as the theme no longer depends on jQuery
]

myst_heading_anchors = 2

bibtex_bibfiles = []

source_suffix = [".rst", ".md"]

master_doc = "index"

autoclass_content = "class"
autodoc_member_order = "bysource"
default_role = "py:obj"

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# From Renate: Use this?
# html_sidebars = {
#     "**": [
#         "searchbox.html",
#         "localtoc.html",
#         "globaltoc.html",
#         "relations.html",
#         "sourcelink.html",
#     ]
# }
