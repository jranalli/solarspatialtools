import pandas as pd
import os

from solarspatialtools.dataio import iotools

"""
Tool for processing NRCAN spatially resolved data as provided on the NRCAN 
data site. 

High Resolution Solar Datasets
   https://www.nrcan.gc.ca/energy/renewable-electricity/solar-photovoltaic/18409

Input
===================
The data should be provided as an individual csv file for each measurement 
sensor within the campaign. Those data need to be manually downloaded and 
stored in an individual directory for that particular day. The files must 
retain the original names with the dataset in order to format the dates, e.g. 
20140717_VAR01.csv. No other CSV files may be located within the directory. 
For example:

    Directory listing for c: \ data \ 20140717_Varennes \ 
    
        - 20140717_VAR01.csv
        - 20140717_VAR02.csv
        - ...

Because info about the sensor locations is not stored in the original data, 
a separate latlon CSV file is required. I had to create these manually. The 
format should be three column with Site, Latitude and Longitude columns. The
column names matter! For example:

    Contents of c: \ data \ Varennes_latlon.csv
     
        Site	Latitude	Longitude
        AFN01	44.190159	-78.096701
     
Output 
===================
The structure of the h5 files will be pandas DataFrame objects each with a key 
assigned. The keys will be as follows:

"data" - DataFrame has an index of the time, localized to UTC and contains the
irradiance in W/m2. Columns are labelled by station id.

"data_tilt" - DataFrame has an index of the time, localized to UTC and contains 
the tilted irradiance in W/m2. Columns are labelled by station id.

"latlon" - DataFrame will have an index of each stationid and columns "lat" and
"lon" for each station, measured in degrees East/North.

Parameters
===================
The parameters for this file are:
source_dir - a directory where all the CSV files for the given day are stored
latlon_file - the full path name of the latlon CSV file
target_file - the full path name of the output h5 file

See the bottom of the file for an example of usage



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

"""


def process_files(source_dir, latlon_file, target_file):
    """
    Perform the file conversion on NRCAN data into an h5 file with consistent
    time step.

    Parameters
    ----------
    source_dir : string
        a directory where all the CSV files for the given day are stored
    latlon_file : string
        the full path name of the latlon CSV file
    target_file: string
        the full path name of the output h5 file

    Output
    -------
        an h5 file saved to disk with keys 'data','data_tilt' and 'latlon'
    """

    # All the files in the source directory
    files = iotools.getfiles(source_dir, ".csv")

    # process the lat lon data
    latlon = pd.read_csv(latlon_file)
    latlon = latlon.rename(columns={"Site": "id",
                                    "Latitude": "lat",
                                    "Longitude": "lon"})
    latlon = latlon.set_index('id')

    # pass through all the files
    for i, file in enumerate(files):
        if "latlon" in file and ".csv" in file:
            pass
        else:
            # File for a single sensor
            data = pd.read_csv(os.path.join(source_dir, file))

            # Generate the time stamps
            file_time = pd.to_datetime(data.iloc[:, 0])\
                        + pd.to_timedelta(data.iloc[:, 1])
            #  Uses GMT+5, rather than Eastern time with DST shift.
            file_time = pd.DatetimeIndex(file_time).tz_localize('Etc/GMT+5')

            # We are currently converting this sensor's data into a column
            col = os.path.splitext(file)[0].split('_')[1]

            # Extract the GHI
            ghi = pd.DataFrame(data["G1 (W/m2)"]).rename(columns=
                                                         {'G1 (W/m2)': col})
            ghi.index = file_time

            # Fill in blank data for even sampling
            ghi = ghi.resample("10ms").ffill()

            # Build the GHI dataframe
            if 'data_ghi' not in vars():
                data_ghi = pd.DataFrame(ghi, index=ghi.index, columns=[col])
            else:
                data_ghi = data_ghi.assign(tmp=ghi).rename(columns=
                                                           {'tmp': col})

            # Extract the Global Tilted Irradiance
            gti = pd.DataFrame(data["G2 (W/m2)"]).rename(columns=
                                                         {'G2 (W/m2)': col})
            gti.index = file_time

            # Fill in blank data
            gti = gti.resample("10ms").ffill()

            # Build the GTI dataframe
            if 'data_gti' not in vars():
                data_gti = pd.DataFrame(gti, index=gti.index, columns=[col])
            else:
                data_gti = data_gti.assign(tmp=gti).rename(columns=
                                                           {'tmp': col})

    # All the dataframes are built, save them to the h5 file
    latlon.to_hdf(target_file, key='latlon', mode='a')
    data_ghi.to_hdf(target_file, key='data', mode='a', append=True)
    data_gti.to_hdf(target_file, key='data_tilt', mode='a', append=True)


if __name__ == "__main__":

    # Build the paths for the individual datafiles
    rootdir = 'c:/path/to/data'

    filedir = '20151008_Alderville'
    site = "Alderville"
    target_fn = 'nrcan_ald_v.h5'

    # filedir = '20150812_Alderville'
    # site = "Alderville"
    # target_fn = 'nrcan_ald_hv.h5'

    # filedir = '20140717_Varennes'
    # site = "Varennes"
    # target_fn = 'nrcan_var_v.h5'

    # filedir = '20150226_Varennes'
    # site = "Varennes"
    # target_fn = 'nrcan_var_hv.h5'

    target_rootdir = 'c:/path/to/output/'

    source_dir = os.path.join(rootdir, filedir)
    latlon_file = os.path.join(rootdir, f"latlon_{site}.csv")
    target_file = os.path.join(target_rootdir, target_fn)

    process_files(source_dir, latlon_file, target_file)
