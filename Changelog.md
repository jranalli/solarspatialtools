Initial releases of the package were still based upon use as a library for 
personal use. As such, the package was not well documented and the structure
underwent some significant changes. The following is a summary of the changes
that have been made to the package beginning with the first PyPi release. 

Beginning with Version 0.3.1 the field analysis package was introduced, which 
may lead to broader interest from potential users. As such, I began to improve
documentation and testing with that release. 

Backwards compatibility has not been uniformly maintained by releases due to 
the primary impact being on my individual use. Beginning with version 0.4.0, 
the package will adopt a consistent approach and will strive to maintain 
backwards compatibility for releases within the same minor version (0.X.X). 

# Version 0.2
First public release of the package via PyPi
# Version 0.2.1
- Add wrapper for `pvlib.clearsky_index` to handle pandas type
# Version 0.2.2
- Change input to camfilter to handle references that don't coincide with the 
site itself. 
- ** Breaks backwards compatibility **
# Version 0.2.3
- Add methods for calculating delay between signals to `signalproc`
# Version 0.2.4
- Add some additional options to `cmv` for qc and filtering of the data
# Version 0.2.5
- Add location analysis in `field` package
# Version 0.3.1
- Expands `field` analysis capability
- Add comprehensive testing
- ** Breaks backwards compatibility **
# Version 0.3.2
- Major speed improvements via vectorization of CMV 
- ** Breaks backwards compatibility due to function signature changes **
# Version 0.3.3
- Major speed improvements to field analysis (vectorization of transfer function delays)
- ** Breaks backwards compatibility **
# Version 0.3.4
- Bug fix for handling and excluding NaN within `field`
# Version 0.3.5
- Add multi-column support to several functions in `stats`. Thanks to Scott Sheppard for the suggestion.
# Version 0.4.0
- Added some functions to `field` to support automated remapping.
    - See `field_reassignment_demo` for details.
- Added a function to `cmv` for identifying an optimum subset of CMV pairs. 
    - See `automate_cmv_demo` for details.
- Added additional statistics to the `jamaly` method's output within `cmv.compute_cmv`
- Added additional sample plant data for demonstration purposes. 
    - We now include two sample plants and sample data from the HOPE-Melpitz campaign at 1s and 10s resolution.
- Added additional demos, including Jupyter notebooks in some cases.
    - `automate_cmv_demo` - Automated CMV pre-processing with HOPE-Melpitz data, includes a Jupyter notebook.
    - `field_reassignment_demo` - Automated remapping of the field data, includes a Jupyter notebook.
    - `field_demo_full_process` - Overview of the full field processing workflow 
    - `field_demo_full_process_multithread` - Same as previous but including multithreading support for speed
- Added tests of these new functions for more completeness.
# Version 0.4.1
- Bug fix for `cmv` so that `jamaly` r_corr is always positive. 
- Demo for digitizing a plant equipment layout added by @williamhobbs
- Improve comments in `cmv_demo`