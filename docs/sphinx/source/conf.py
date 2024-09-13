# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'solarspatialtools'
copyright = '2024, Joe Ranalli'
author = 'Joe Ranalli'

release = '0.4'
version = '0.4.5'

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join('..', '..', '..', 'src')))

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.duration',
              'sphinx.ext.doctest',
              'sphinx.ext.autodoc',
              'sphinx.ext.autosummary',
              'sphinx.ext.intersphinx',
              'nbsphinx',
              'nbsphinx_link',
              ]

intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
}

templates_path = ['_templates']



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'sphinx_rtd_theme'
# html_static_path = ['_static']

# Control the display of class members
autodoc_member_order = 'bysource'

# -- Options for EPUB output
epub_show_urls = 'footnote'

# Hide warning per sphinx #12300, appears to come from nbsphinx or nbsphinx_link
suppress_warnings = ["config.cache"]

