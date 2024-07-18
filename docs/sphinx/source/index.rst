.. solartoolbox documentation master file, created by
   sphinx-quickstart on Mon Jul 15 15:04:14 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SolarToolbox documentation
==========================

``solartoolbox`` is a python package containing implementations of various spatial algorithms for solar energy data. While excellent open source solar energy packages already exist (e.g. `pvlib-python <https://github.com/pvlib/pvlib-python>`_ and `pvanalytics <https://github.com/pvlib/pvanalytics>`_), the complexity of some high-level analyses found in the academic literature makes them as a poor fit for the scope of existing packages. This package fills that gap by implementing techniques that we hope can facilitate common spatial tasks for solar energy researchers and provide a platform for consistency and efficiency improvements in these calculations.


Installation
------------
The package can be most easily installed via `PyPi <https://pypi.org/project/solartoolbox/>`_ with the following command::

        pip install solartoolbox




Add your content using ``reStructuredText`` syntax. See the
`reStructuredText <https://www.sphinx-doc.org/en/master/usage/restructuredtext/index.html>`_
documentation for details.

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   cmv
   field
   othermods

.. toctree::
   :maxdepth: 2
   :caption: Examples

   examples

