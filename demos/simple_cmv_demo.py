import numpy as np
import pandas as pd
from solartoolbox import spatial, cmv
import pvlib
import matplotlib.pyplot as plt

"""
This script will do a demonstration of computing the Cloud Motion Vector (CMV)
using the methods available in solartoolbox.  
"""

# ##### INPUTS ##### #
fn = "data/nrcan_ald_hv.h5"
win = pd.date_range('2015-08-12 08:00:00', '2015-08-12 16:00:00',
                    freq='10ms',
                    tz='Etc/GMT+5')  # ALD HV
pt_ref = 'AFN11'
resample = '250ms'


# Read in the file, convert plant to UTM. Define Center Position
pos = pd.read_hdf(fn, mode="r", key="latlon")
pos_utm = spatial.latlon2utm(pos['lat'], pos['lon'])
center_loc = pvlib.location.Location(pos['lat'][pt_ref], pos['lon'][pt_ref])

# Read, select the window, and resample the dataset
ts = pd.read_hdf(fn, mode="r", key="data")
ts = ts.loc[win]
ts = ts.resample(resample).mean()

# Compute the clearsky
cs_ghi = center_loc.get_clearsky(ts.index, model='simplified_solis')['ghi']
# Apply the clear sky to the data column by column from pvlib
kt = ts.apply(lambda x:
              pvlib.irradiance.clearsky_index(x, cs_ghi, 1000), axis=0)

# Compute the cloud motion vector using both methods
cld_spd_gag, cld_dir_gag, dat_gag = cmv.compute_cmv(kt, pos_utm,
                                                    reference_id=None,
                                                    method='gagne',
                                                    corr_scaling='coeff')
cld_spd_jam, cld_dir_jam, dat_jam = cmv.compute_cmv(kt, pos_utm,
                                                    reference_id=None,
                                                    method='jamaly',
                                                    corr_scaling='coeff')

# Print the answer
print("Method  Spd    Angle  N_good")
print("Gagne   {:0.2f}   {:0.2f}".format(cld_spd_gag, np.rad2deg(cld_dir_gag)),
      sum(dat_gag.pair_flag == cmv.Flag.GOOD))
print("Jamaly  {:0.2f}   {:0.2f}".format(cld_spd_jam, np.rad2deg(cld_dir_jam)),
      sum(dat_jam.pair_flag == cmv.Flag.GOOD))

# ##### PLOTS ##### #
plt.title('Result Data for Jamaly Method')
# Correlation scatter
plt.scatter(dat_jam.pair_lag, dat_jam.pair_dists, c=dat_jam.corr_lag,
            vmin=0, vmax=1)
# Label the points that came through as GOOD from the CMV routine
plt.plot(dat_jam.pair_lag[dat_jam.pair_flag == cmv.Flag.GOOD],
         dat_jam.pair_dists[dat_jam.pair_flag == cmv.Flag.GOOD], 'x')
# Best fit line
plt.plot([np.min(dat_jam.pair_lag), np.max(dat_jam.pair_lag)],
         cld_spd_jam *
         np.array([np.min(dat_jam.pair_lag),np.max(dat_jam.pair_lag)]), 'r:')
plt.xlabel('Lag (s)')
plt.ylabel('Distance (m)')
plt.legend(['Point Correlation', 'GOOD points', 'Speed Fit'])
plt.tight_layout()
plt.show()
