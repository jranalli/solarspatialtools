Other Functions
===============

signalproc module
-----------------
This module contains functions for performing common digital signal processing tasks. Many are utilized by other modules within the package. In the context of the cmv and field modules, these functions are used primarily to calculate relationships between signals, specifically relative delays.

.. automodule:: solartoolbox.signalproc

    .. rubric:: Signal Processing Functions

    .. autosummary::
        :toctree: generated/

            averaged_psd
            averaged_tf
            correlation
            tf_delay
            xcorr_delay
            compute_delays
            interp_tf
            apply_delay


spatial module
--------------

.. automodule:: solartoolbox.spatial

    .. rubric:: Coordinates and Vector Projections

    These functions are utilized as parts of the CMV and field modules. The first two are used to transform between latitude and longitude coordinates and a cartesian frame more suitable for use with the CMV and field routines. The remaining functions are used to perform vector operations on the field of points within the cartesian coordinate frame.

    .. autosummary::
        :toctree: generated/

            latlon2utm
            utm2latlon
            project_vectors
            compute_vectors
            compute_intersection

    .. rubric:: Basic Vector Operations

    The following functions are simply implementations of common vector operations that are utilized by other sections of the code. They are only included here for completeness.

    .. autosummary::
        :toctree: generated/

            dot
            unit
            magnitude
            pol2rect
            rect2pol
            rotate_vector

stats module
------------

.. automodule:: solartoolbox.stats

    .. rubric:: Variability Metrics

    These functions represent metrics that are used to analyze variability.

    .. autosummary::
        :toctree: generated/

            variability_score
            variability_index
            darr
            calc_quantile

    .. rubric:: Basic Statistics

    These are just implementations of basic statistical error calculations and are only included as shortcuts for the user.

    .. autosummary::
        :toctree: generated/

            rmse
            mse
            squared_error
            mae
            absolute_error
            bias_error
            mbe

irradiance module
-----------------
This module solely serves as a helper, wrapping the `pvlib` function `pvlib.irradiance.clearsky_index`, which doesn't natively handle some of the data types used by this package.

.. automodule:: solartoolbox.irradiance

    .. rubric:: Functions

    .. autosummary::
        :toctree: generated/

            clearsky_index