.. currentmodule:: solarspatialtools

Synthetic irradiance generation
----------------------------------
The `solarspatialtools.synthirrad` package contains tools for generating synthetic irradiance timeseries and performing downscaling of timeseries. The package implements the following approaches:

cloudfield
==========

Generate a simulated field of clouds from which spatially distributed timeseries of kt can be extracted. The field distributions are based on the properties of a time series of kt values. This is an implementation of the method described by Lave et al [1]. Some aspects of the implementation diverge slightly from the initial paper to follow a subsequent code implementation of the method shared by the original authors.

 [1] Matthew Lave, Matthew J. Reno, Robert J. Broderick, "Creation and Value of Synthetic High-Frequency Solar Inputs for Distribution System QSTS Simulations," 2017 IEEE 44th Photovoltaic Specialist Conference (PVSC), Washington, DC, USA, 2017, pp. 3031-3033, doi: https://dx.doi.org/10.1109/PVSC.2017.8366378.

.. automodule:: solarspatialtools.synthirrad.cloudfield


    .. rubric:: Functions

    .. autosummary::
        :toctree: generated/

            get_timeseries_stats
            cloudfield_timeseries


