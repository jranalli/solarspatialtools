# SolarToolbox
`solartoolbox` is a package containing tools for dealing with analysis of solar 
energy data. Its specific focus is on signal processing approaches and 
addressing variability from a spatiotemporal perspective. Tools here might be 
useful for dealing with distributed data sets, or performing analyses that 
rely on a spatially distributed set of measurements.

# Installation
The package can be most easily installed via [PyPi](https://pypi.org/project/solartoolbox/) 
with the following command:
```bash
pip install solartoolbox
```

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
the functionality of the package. An explanation the included data is 
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

## Function libraries in solartoolbox (root level)

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

## Common format for H5 files used for Data Storage

I've tried to format the multisite time series measurements in a way that's 
conveinent for loading the files and working with the data. This came about 
from my initial work analyzing the HOPE Campaign, which used 100 individual 
point measurements of GHI scattered through a region near Jülich, Germany.

All data is collected into a single H5 file containing multiple fields. I use
```pandas``` and specifically ```pandas.read_hdf()``` for getting the data
into python. 

- ```latlon```: The latitude/longitude of the individual measurement sites
- ```utm```: The UTM coordinates of the individual measurement sites
- ```data```: Global Horizontal Irradiance
- ```data_tilt```: Global Tilted Irradiance (if available)

#### Location Data
Data about the location of each individual site is stored in the H5 file. Two
possible keys are used depending on the projection. Both are available when 
possible. The key ```latlon``` represents the site in a latitude coordinate
system. The key ```utm``` will contain the positions using UTM (or similar) 
projection that attempts to place the layout into a rectilinear coordinates. 
Upon use of ```pandas.read_hdf()``` the data will be brought into a DataFrame 
object.

- The index of the DataFrame is the site id. The HOPE datasets use an integer 
for the id, while NRCAN uses a string. 
- Columns are labelled ```lat``` and ```lon``` and contain the lat and lon in 
degrees for each of the distributed sensors (or ```E```, ```N``` in the case of 
```utm```).

#### Irradiance Data
Measurements consist of the individual sensor time series with a shared time 
index. Upon use of ```pandas.read_hdf()``` the data will be brought into a 
DataFrame object. Each individual sensor has its own column. 

- Index of the DataFrame is the timestamp referenced to a timezone
- Columns contain the time series for each individual sensor, and are keyed by
the site id (HOPE - integer, NRCAN - string).

# Contributing
This project is happy to accept contributions and hear from users. The best way 
to interact right now is to open an [issue](https://github.com/jranalli/solartoolbox/issues) 
on GitHub. This is the best way to ask a question, make a suggestion about a 
feature or describe a bug that you've encountered. 

# License
This project is licensed under the BSD 3-Clause License - see the 
[LICENSE](https://github.com/jranalli/solartoolbox/blob/main/LICENSE) file
for full details. 

# History
Solartoolbox began as a library of tools developed to support my own research 
activities on solar energy. It was always shared publicly to encourage use by 
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


