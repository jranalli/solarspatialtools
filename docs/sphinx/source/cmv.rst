.. currentmodule:: solarspatialtools

Cloud Motion Vector Identification
----------------------------------
The `cmv` module contains tools for identifying the cloud motion vector (CMV) from a distributed network of measurement sensors. Two methods are implemented, that of Jamaly and Kleissl [1] and of Gagne et al. [2].

The tool also contains a function for identifying the optimum subset of CMV vectors from among a long list of CMV pairs. This functions serves as part of the automated CMV identification workflow that serves the :mod:`solarspatialtools.field` module.

See examples found in the :ref:`cmv-examples` section. `cmv_demo` highlights identification of the CMV, while `automate_cmv_demo` shows how a group of useful CMVs can be downselected from a long time series of data [3].

[1] M. Jamaly and J. Kleissl, "Robust cloud motion estimation by spatio-temporal correlation analysis of irradiance data," Solar Energy, vol. 159, pp. 306-317, Jan. 2018. https://www.sciencedirect.com/science/article/pii/S0038092X17309556

[2] A. Gagne, N. Ninad, J. Adeyemo, D. Turcotte, and S. Wong, "Directional Solar Variability Analysis," in 2018 IEEE Electrical Power and Energy Conference (EPEC) (2018) pp. 1-6, iSSN: 2381-2842 https://www.researchgate.net/publication/330877949_Directional_Solar_Variability_Analysis

[3] J. Ranalli and W.B. Hobbs, “Automating Methods for Validating PV Plant Equipment Labels,” 52nd IEEE PV Specialists Conference, 2024.

.. automodule:: solarspatialtools.cmv


    .. rubric:: Functions

    .. autosummary::
        :toctree: generated/

            compute_cmv
            optimum_subset


