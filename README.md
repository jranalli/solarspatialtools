# solartoolbox
Solartoolbox is a collection of tools that are used for my research on solar 
energy and data analysis of solar variability. I offer apologies in advance, 
because I'm not a developer, but a solar energy researcher, so this isn't meant 
to be a perfect API and may not exhibit best practices for software development 
or programming. Rather, these tools are primarily published for my own use, but 
are shared publicly if they may be valuable to other investigators or those who
try to replicate my work. 

The primary features at present relate to working with multisite datasets for
variability analysis, including via frequency domain approaches.

## Structure of the Library
The codes are currently broken up in a way that made the most sense to me
### Packages
```dataio```  
A package with codes for accessing datasets that I've been working with and 
converting them to a common format for use with the other codes. Current 
datasets:
- [HOPE and HOPE-MELPITZ](https://www.cen.uni-hamburg.de/en/icdc/data/atmosphere/samd-st-datasets.html)
- [NRCAN High Resolution Datasets](https://www.nrcan.gc.ca/energy/renewable-electricity/solar-photovoltaic/18409)

Some of these tools are meant to be used via the command line and some via
code. There needs to be some cleanup done there to get things more universal, 
but for now the codes are able to get the job done.

```visualization```  
Tools for visualizing various types of data or constructing common plots that 
might be useful for these analyses.

```demos```  
Some demonstration codes and jupyter notebooks to demonstrate usage of the 
tools.

### Function libraries
```solartoolbox (root)```
General tools or wrappers for other functions.

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
- ```data```: Global Horizontal Irradiance
- ```data_tilt```: Global Tilted Irradiance (if available)

#### Location Data
Data about the location of each individual site is stored in the H5 file with 
the key ```latlon``` as stated above. Upon use of ```pandas.read_hdf()``` the 
data will be brought into a DataFrame object.

- The index of the DataFrame is the site id. The HOPE datasets use an integer 
for the id, while NRCAN uses a string. 
- Columns are labelled ```lat``` and ```lon``` and contain the lat and lon in 
degrees for each of the distributed sensors.


#### Irradiance Data
Measurements consist of the individual sensor time series with a shared time 
index. Upon use of ```pandas.read_hdf()``` the data will be brought into a 
DataFrame object. Each individual sensor has its own column. 

- Index of the DataFrame is the timestamp referenced to a timezone
- Columns contain the time series for each individual sensor, and are keyed by
the site id (HOPE - integer, NRCAN - string).



## Significant Changelog
##### Version 0.2
First public release
##### Version 0.2.1
Add wrapper for `pvlib.clearsky_index` to handle pandas type
##### Version 0.2.2
Change input to camfilter to handle references that don't coincide with the 
site itself. This change breaks code!
##### Version 0.2.3
Add methods for calculating delay between signals to `signalproc`
##### Version 0.2.4
Add some additional options to CMV code
##### Version 0.2.5
Add field analysis.
##### Version 0.3.1
A non-backwards-compatible major revision to incorporate field analysis and add more comprehensive testing.
##### Version 0.3.2
A non-backwards-compatible major revision to create a major speedup on the CMV.

## Author
Joe Ranalli  
Associate Professor of Engineering  
Penn State Hazleton  
jar339@psu.edu  
https://jranalli.github.io/


