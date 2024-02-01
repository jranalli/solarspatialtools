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