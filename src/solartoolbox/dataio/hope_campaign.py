import os.path
import argparse
import pandas as pd
import numpy as np
import netCDF4
from solartoolbox.dataio.iotools import ensurepath, getfiles
from solartoolbox.dataio.iotools import download, wget_fromurl

"""
Tool for downloading all of the HOPE and HOPE-Melpitz data from the University
of Hamburg data site. Replaces use of their provided wget files for a Windows
environment. 
Old Site: 
   https://icdc.cen.uni-hamburg.de/all-samd-data.html
   
   Files can be navigated to via:
     Short Term Observations > HOPE > Instrument Groups > ...
         Radiation & Imager > rsds    ---> wget_hope.sh

     Short Term Observations > HOPE Melpitz > Instrument Groups > ...
         Radiation & Imager > rsds   ---> wget_melpitz.sh
New Site:
   Data > Atmosphere > SAMD Data Sets - Short Term Observations > HOPE 
   Field for irradiance is "rsds"
   https://www.cen.uni-hamburg.de/en/icdc/data/atmosphere/samd-st-datasets.html

If generated, the structure of the h5 files will be pandas DataFrame objects 
each with a key assigned. The three keys will be as follows:

"data" - DataFrame has an index of the time, localized to UTC and contains the
irradiance in W/m2. Columns are labelled by station id.

"flag" -  DataFrame has an index of the time, localized to UTC and contains the
data flag. Columns are labelled by station id. According to the raw field info 
inside the raw files, the flag meanings are:
    1 - good_data
    2 - okey_but_sometimes_dubious_data
    3 - bad_data_ignore_completely
    4 - no_observation

"latlon" - DataFrame will have an index of each stationid and columns "lat" and
"lon" for each station, measured in degrees East/North.


example usage:
    python hope_campaign.py HOPE d:\data\hope -m -h5 -q interp
    
    
    
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

# Magic strings pointing to the files for each of the HOPE data sets
URLS = {"HOPE":
        {"wget": "https://icdc.cen.uni-hamburg.de/thredds/fileServer/"
                 "ftpthredds/samd/"
                 "data/pyrnet/00/rsds/l1/hope/trop/2013/wget.sh",
         "meta": "https://icdc.cen.uni-hamburg.de/thredds/fileServer/"
                 "ftpthredds/samd/"
                 "metadata/icdc/hope_trop_pyrnet00_l1_rsds_v00.xml"},
        "HOPE-MELPITZ":
        {"wget": "https://icdc.cen.uni-hamburg.de/thredds/fileServer/"
                 "ftpthredds/samd/"
                 "data/pyrnet/00/rsds/l1/hopm/trop/2013/wget.sh",
         "meta": "https://icdc.cen.uni-hamburg.de/thredds/fileServer/"
                 "ftpthredds/samd/"
                 "metadata/icdc/hopm_trop_pyrnet00_l1_rsds_v00.xml"},
        "LINDENBERG":
        {"wget": "https://icdc.cen.uni-hamburg.de/thredds/fileServer/"
                 "ftpthredds/samd/"
                 "data/pyrnet/00/rsds/l1/ioprao/trop/2014/wget.sh",
         "meta": "https://icdc.cen.uni-hamburg.de/thredds/fileServer/"
                 "ftpthredds/samd/"
                 "metadata/icdc/ioprao_trop_pyrnet00_l1_rsds_v00.xml"}}


def flag_qc(data, flag, level=2):
    """

    Parameters
    ----------
    data : pandas DataFrame
        A dataframe containing the data to interpolate

    flag : pandas DataFrame
        A dataframe containing the flags for the data

    level : int
        The lowest unacceptable flag level (flag >= level is set to nan)

    Returns
    -------
    datanew : pandas DataFrame
        A modified version of the data
    """
    datanew = data.copy()
    datanew[flag >= level] = np.nan
    return datanew


def null_qc(data):
    """
    Replace data with nan based on the dataset hidden values (-999)

    Parameters
    ----------
    data : pandas DataFrame
        A dataframe containing the data to limit

    Returns
    -------
    datanew : pandas DataFrame
        A modified version of the data
    """

    datanew = data.copy()
    datanew[datanew < -100] = np.nan
    return datanew


def interp_nan(data, limit=5):
    """
    Interpolate over nan values. Duration is limited by the limit parameter.
    Periods of nan values that span longer than the limit will be left as nan.

    Parameters
    ----------
    data : pandas DataFrame
        A dataframe containing the data to interpolate

    limit : int, default 5
        The limit to the number of timesteps to interpolate through

    Returns
    -------
    datanew : pandas DataFrame
        A modified version of the data
    """

    datanew = data.copy()

    for col in datanew.columns:
        datanew_i = datanew[col]
        # interpolate limiting things to 5s or shorter nan windows
        datanew_i = datanew_i.interpolate(limit=limit, limit_area='inside')

        # Find the cases where things were nan for more than limit, and add
        # some padding to ensure we don't interp there
        blanks = np.isnan(datanew_i).rolling(f'{limit}s').mean() > \
                 (limit-1)/limit
        blanks_front = blanks.iloc[:-2*limit]
        blanks_back = blanks.iloc[2*limit:]
        blanks_union = blanks_front.values | blanks_back.values
        blanks = pd.Series(np.pad(blanks_union, limit), index=blanks.index)
        datanew_i[blanks] = np.nan

        datanew[col] = datanew_i

    return datanew


def parse_file(filename):
    """
    Convert a datafile in the HOPE format to pandas DataFrame objects.

    data DataFrame has an index of the time, localized to UTC and contains the
    irradiance in W/m2. Columns are labelled by station id.

    flag DataFrame has an index of the time, localized to UTC and contains the
    data flag. Columns are labelled by station id. According to the data
    description PDF file, the flag meanings are:
        1 - good_data
        2 - okey_but_sometimes_dubious_data
        3 - bad_data_ignore_completely
        4 - no_observation

    The stationdat DataFrame will have an index of each stationid and columns
    "lat" and "lon" for each station, measured in degrees East/North.

    Parameters
    ----------
    filename : string
        the full path filename to the datafile

    Returns
    -------
    (data, flag, stationdat) : (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        data is a dataframe of the actual time series indexed by timestamp and
            with columns of each station id
        flag is a dataframe of the qc flags indexed by timestamp and with
            columns of each station id
        stationid is a dataframe with the location of each of the individual
            stations indexed by station id and with columns of "lat" and "lon"
            in degrees.
    """

    # Read file
    data = netCDF4.Dataset(filename, mode='r')
    data.set_auto_mask(False)

    # Station Identifiers
    ids = data['station_id'][:]

    # The station info Frame
    lat = data['lat'][:]
    lon = data['lon'][:]

    stationdat = pd.DataFrame(np.array([lat, lon]).transpose(),
                              index=ids.transpose(),
                              columns=['lat', 'lon'])

    # Get all times, convert to pandas time and localize to UTC
    time = data['time']
    timeindex = pd.to_datetime(time[:], unit="s")
    timeindex = timeindex.tz_localize('UTC')

    # Get Irradiance
    irrad = data['rsds'][:]
    irrad_flag = data['rsds_flag'][:]

    # Build Data and Flag Frames
    data = pd.DataFrame(irrad, columns=ids, index=timeindex)
    flag = pd.DataFrame(irrad_flag, columns=ids, index=timeindex)

    return data, flag, stationdat


def dataset_to_h5(directory, filename, overwrite=False, verbose=True):
    """
    Convert a dataset stored in a directory to h5 file. DataFrame objects each
    have a key assigned. The three keys will be as follows:

    "data" - DataFrame has an index of the time, localized to UTC and contains
    the irradiance in W/m2. Columns are labelled by station id.

    "flag" -  DataFrame has an index of the time, localized to UTC and contains
    the data flag. Columns are labelled by station id. According to the raw
    field info inside the raw files, the flag meanings are:
    1 - good_data
    2 - okey_but_sometimes_dubious_data
    3 - bad_data_ignore_completely
    4 - no_observation

    "latlon" - DataFrame will have an index of each stationid and columns "lat"
    and "lon" for each station, measured in degrees East/North.

    Parameters
    ----------
    directory : string
        directory holding the downloaded dataset files

    filename : string
        Full context filename of the target h5 file.

    overwrite : bool
        should the file be overwritten if it exists?

    verbose : bool
        print status updates?

    Returns
    -------
    void
    """

    files = getfiles(directory, ext=".nc", sort=True)

    if os.path.isfile(filename):
        if overwrite:
            os.remove(filename)
        else:
            raise ValueError("Target file exists. Please specify a new file.")

    for i, fn in enumerate(files):
        if verbose:
            print(fn)

        data, flag, latlon = parse_file(os.path.join(directory, fn))

        if i == 0:
            latlon.to_hdf(filename, key='latlon', mode="a")

        data.to_hdf(filename, key='data', mode='a', append=True)
        flag.to_hdf(filename, key='flag', mode='a', append=True)


def _parse_args():
    """
    parse the arguments specified in the program call.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Name of the dataset to get")
    parser.add_argument("target", help="Target directory to save data")
    parser.add_argument("-m", "--meta", help="Download metadata?",
                        action="store_true")
    parser.add_argument("-h5", "--makeh5", help="create h5 summary file",
                        action="store_true")
    parser.add_argument("-s", "--silent", help="Hide printouts?",
                        action="store_true")
    parser.add_argument("-q", "--qc", help="Quality control.")
    args = parser.parse_args()
    if args.dataset.upper() not in URLS.keys():
        raise ValueError("Dataset must be one of: " + str(URLS.keys()))
    return args


def _main():
    args = _parse_args()
    ensurepath(args.target)
    if args.meta:
        if not args.silent:
            print("Metadata")
        download(URLS[args.dataset.upper()]['meta'],
                 os.path.join(args.target, args.dataset + "_metadata.xml"))
    wget_fromurl(URLS[args.dataset.upper()]['wget'],
                 args.target, not args.silent)

    if args.makeh5:
        try:
            dataset_to_h5(args.target,
                          os.path.join(args.target,
                                       args.dataset.lower() + ".h5"),
                          overwrite=False,
                          verbose=(not args.silent))
        except ValueError:
            userinput = input("File exists. Overwrite (Y/N)?")
            if userinput.lower() == "y":
                dataset_to_h5(args.target,
                              os.path.join(args.target,
                                           args.dataset.lower() + ".h5"),
                              overwrite=True,
                              verbose=(not args.silent))
            else:
                print("Continuing")

    if args.qc is not None:
        print('qc')
        fn_src = os.path.join(args.target, args.dataset.lower() + ".h5")
        fn_tgt = os.path.join(args.target,
                     args.dataset.lower() + "-qc{}.h5".format(args.qc.lower()))
        data = pd.read_hdf(fn_src, mode="r", key="data")
        flag = pd.read_hdf(fn_src, mode="r", key="flag")
        latlon = pd.read_hdf(fn_src, mode="r", key="latlon")

        if args.qc == "interp":
            data = null_qc(data)
            data = interp_nan(data)
        elif args.qc == 'null':
            data = null_qc(data)
        elif args.qc == 'flag':
            data = flag_qc(data, flag, 2)
        else:
            raise ValueError('QC must be one of None, null, interp, flag.')
        if os.path.isfile(fn_tgt):
            userinput = input("File exists. Overwrite (Y/N)?")
            if userinput.lower() == "y":
                os.remove(fn_tgt)
            else:
                return
        data.to_hdf(fn_tgt, key='data', mode='a', append=False)
        flag.to_hdf(fn_tgt, key='flag', mode='a', append=False)
        latlon.to_hdf(fn_tgt, key='latlon', mode='a', append=False)


if __name__ == "__main__":
    _main()
