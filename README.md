# SolarSpatialTools
Spatial analysis tools for solar energy research

`solarspatialtools` is a python package containing implementations of various spatial algorithms for solar energy data. While excellent open source solar energy packages already exist (e.g. [pvlib-python](https://github.com/pvlib/pvlib-python) and [pvanalytics](https://github.com/pvlib/pvanalytics)), the complexity of some high-level analyses found in the academic literature makes them as a poor fit for the scope of existing packages. This package fills that gap by implementing techniques that we hope can facilitate common spatial tasks for solar energy researchers and provide a platform for consistency and efficiency improvements in these calculations. 

Two examples of what are believed to be the most generally useful techniques this package implements are:
- Processing the cloud motion vector from a distributed sensor network. 
- Verifying the locations of field components (e.g. combiners) within a distributed network of measurements

# Installation
The package can be most easily installed via [PyPi](https://pypi.org/project/solartoolbox/) 
with the following command:
```bash
pip install solarspatialtools
```

# Getting Started
A number of example codes are available in the demos folder. These are meant to demonstrate what are believed to be the most useful functions in the package and applications of the functions to real sample data included with the library.


# Common Data Formatting

The algorithm implementations in this package primarily involve spatially distributed analyses of solar energy data, necessitating processing of multiple simultaneous time series. Most codes will assume that data is provided using a common format, based upon ```pandas``` DataFrame objects. The demo codes use the following convention for the two most important data variables. 
- ```pos``` or ```pos_utm``` - Data for the location of each sensor should be stored in a DataFrame with the sensor id as the index and the latitude and longitude (or UTM coordinates) as columns. 
- ```ts_data``` - Sensor time series data should be stored in a DataFrame with the timestamp as the index and the sensor id as the columns. 
The index of ```pos``` needs to match the columns of ```ts_data``` so that the correspondence between the locations and the time series can be maintained.


# Structure of the Library
The codes are organized into a few subpackages and several function libraries.
The subpackages are meant to contain tools that are related to a specific
task. 

### Packages
```dataio```  
A package with codes for accessing a couple of distributed irradiance datasets 
that I've worked with and for converting them to a common format for use with 
the other codes. Current datasets:
- [HOPE and HOPE-MELPITZ](https://www.cen.uni-hamburg.de/en/icdc/data/atmosphere/samd-st-datasets.html)
- [NRCAN High Resolution Datasets](https://www.nrcan.gc.ca/energy/renewable-electricity/solar-photovoltaic/18409)

Some of these tools are meant to be used via the command line and some via
function call. This area of the package may be in need of some cleanup to 
improve consistency.

```visualization```  
Tools for visualizing various types of data or constructing common plots that 
might be useful for these analyses. Right now this only contains a function 
for decorating the frequency axis of plots with common timescales. This is an 
area that could use some expansion in the future.

```demos```  
Data and demonstration codes (including as jupyter notebooks) that demonstrate 
the functionality of the package. An explanation for the included data is 
warranted. 

- Anonymized Plant Combiner Data  
  - Anonymized combiner time series data from ~20 MW (`sample_plant_1.h5`) and 
  ~30 MW (`sample_plant_2.h5`) solar plants in the United States.
  - Field `utm` actually contains the UTM-like (East/North) centroid 
  positions of individual combiners, anonymized with an arbitrary offset. 
  Columns are `E` and `N` and units are meters.
  - Fields `data_a`, `data_b` through `data_e` contain the combiner current 
  measurements for five hours of operation throughout the year with known high 
  variability. Sampling period is 10 seconds. The absolute time stamps are 
  arbitrary and do not correspond to any real time. Data are normalized for 
  anonymization.
  - Combiner ids are used as column names for the `data` time series and 
  correspond to the matching index of `utm`
  - See `cmv_demo.ipynb` and `field_demo.ipynb` for examples using this data.
- HOPE Melpitz Campaign Data
  - Subset of data from the HOPE-Melpitz campaign of time series from 50 
  distributed irradiance sensors. For details on this data, refer to: 
  [Macke et al. (2017)](https://doi.org/10.5194/acp-17-4887-2017) 
  and [Dataset Website](https://www.cen.uni-hamburg.de/en/icdc/data/atmosphere/samd-st-datasets.html)
    - `hope_melpitz_1s` contains data sampled at 1s time resolution. 
      - Covers a single hour of data (9:15 - 10:15 UTC on Sept 8, 2013).
    - `hope_melpitz_10s` contains data sampled at 10s time resolution, acquired
    by temporally averaging time series data from the original dataset. 
      - Covers 4 full days, from Sept 8 - Sept 11, 2013.
  - In both cases, data were first postprocessed using only removal of nulls 
  (-9999) and linear interpolation to fill gaps left by the nulls, with a 
  maximum interpolation window of 5s. See `dataio.hope_campaign` for details on
  those postprocessing steps.
  - Fields are `latlon`, `utm`, and `data`. 
  - Numerical sensor IDs match those from the original dataset, and original
  timestamps are preserved in the `data` field. All timestamps are UTC.
  - See `dataio\hope_campaign.py` for details on the original dataset.
  - See `signalproc_demo.py` for examples using this data. 

## Function libraries in solarspatialtools (root level)

```cmv```  
Functions for computing the cloud motion vector from a distributed irradiance 
dataset. Two methods from literature are available:
- [Jamaly and Kleissl (2018)](https://www.sciencedirect.com/science/article/pii/S0038092X17309556)
- [Gagne et al. (2018)](https://www.researchgate.net/publication/330877949_Directional_Solar_Variability_Analysis)

```signalproc```  
Functions for performing signal processing on time series. The two primary 
parts of this are computations of averaged transfer functions between an input
and output signal (e.g. calculation of coherence) and code for computing the 
[Cloud Advection Model (CAM)](https://aip.scitation.org/doi/10.1063/5.0050428).

```spatial```  
Functions for dealing with spatially distributed locations. This includes 
conversion between lat/lon and UTM coordinates, along with some vector
operations needed to deal with other parts of the analysis. Examples include
computing vectors between all locations in a distributed location set and 
projecting those vectors parallel/perpendicular to a cloud motion direction.

```stats```  
A set of functions for calculating various quantities on datasets.
- Common statistical error metrics (RMSE, MBE, MAE, etc)
- Lagging cross-correlation via ```correlate()```
- Variability metrics (Variability Score, Variability Index, DARR)
- Quantile summary (e.g. for synthesizing a clear day from the 90th percentile 
of each hour of the day over a 30 day window)

```field```
Functions for predicting the position of field components on the basis of cloud
motion. 


# Contributing
This is an open source project and appreciates participation, engagement and contribution from community members. Development on the library is active and the project seeks to provide a useful tool for the research community. The project is currently maintained by an individual researcher, and the process for contributions is not as formalized as it might be for larger projects.     

If you've found a bug or have an idea for a new feature, please open an [issue](https://github.com/jranalli/solartoolbox/issues) 
on GitHub. Questions can be asked in the GitHub [discussions](https://github.com/jranalli/solartoolbox/discussions).

Code contributions are also welcome! Please follow the instructions in GitHub's [getting started guide](https://docs.github.com/en/get-started/start-your-journey/hello-world) to open a pull request. 

Changes to the contribution guidelines and policies may be made in the future in response to growth of the project and community.

# License
This project is licensed under the BSD 3-Clause License - see the [LICENSE](https://github.com/jranalli/solartoolbox/blob/main/LICENSE) file for full details. 

# History
SolarSpatialTools began as a library of tools developed to support my own 
research activities on solar energy. Initial publication took place under the 
name of `solartoolbox`. It was always shared publicly to encourage use by 
interested parties and make a small contribution to open science and promoting
reproducibility in the field. As a result, most initial releases lacked full 
documentation and the structure of the library underwent some significant 
changes as they adapted to my own changing needs from project to project. Some 
artifacts of that history may still be present in the code and certainly are 
reflected by the commit history.

Beginning with Version 0.3.1 and the introduction of the field analysis 
package, I began to see the potential for broader interest in the tools
which may lead to a greater need to accommodate other users. As such, I began 
to improve documentation and testing with that release and hope to reach a more
stable and consistent structure for the library. The expectation is that the 
packages `cmv` and `field` will be the most broadly useful to the research 
community and have been the focus of additional testing, documentation and 
tutorial development.

See `Changelog.md` for more details.

# Relationship with other packages
This package is not meant to replace or compete with well established packages 
like [pvlib](https://github.com/pvlib/pvlib-python) or
[pvanalytics](https://github.com/pvlib/pvanalytics). Instead, the focus is to 
serve as a complement to those packages, especially in offering functionality 
that would otherwise be out of scope for their mission. When overlap occurs, 
functionality developed here will be contributed to those more mature packages
if they are deemed in scope or suitable by the maintainers of those packages.

For example, the `pvlib-python` port of the Wavelet Variability Model was 
initially developed as part of this package, but was later contributed to 
`pvlib-python` in the [scaling](https://pvlib-python.readthedocs.io/en/stable/reference/generated/pvlib.scaling.wvm.html) 
module thereof.

## Author
Joe Ranalli  
Associate Professor of Engineering  
Penn State Hazleton  
jar339@psu.edu  
https://jranalli.github.io/


