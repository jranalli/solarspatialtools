.. currentmodule:: solarspatialtools

Plant Field Validation
----------------------

The `field` module contains functions for validating the positions of solar panels in a plant field. It relies on knowledge of the cloud motion vector (see :mod:`solarspatialtools.cmv`) and several signal processing routines in :mod:`solarspatialtools.signalproc`. Full details of the methodology are available in reference papers [1, 2].

For example usages, see the :ref:`field-examples` page, specifically `field_demo`, `field_demo_detailed` and `field_reassignment_demo`. Examples of the full workflow are available in python files in the `demos` directory, `field_demo_full_process.py` and `field_demo_full_process_multithread.py`.

[1] J. Ranalli and W. Hobbs, “PV Plant Equipment Labels and Layouts can be Validated by Analyzing Cloud Motion in Existing Plant Measurements,” IEEE Journal of Photovoltaics, Vol. 14, No. 3, pp. 538-548, 2024. DOI: https://doi.org/10.1109/JPHOTOV.2024.3366666

[2] J. Ranalli and W.B. Hobbs, “Automating Methods for Validating PV Plant Equipment Labels,” 52nd IEEE PV Specialists Conference, 2024.

.. automodule:: solarspatialtools.field


    .. rubric:: Functions

    .. autosummary::
        :toctree: generated/

            compute_predicted_position
            compute_delays
            assign_positions
            remap_positions
            cascade_remap