import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from solartoolbox import stats, spatial, cmv

from scipy.stats import linregress

# This is a demo of automatically detecting CMVs within a time series and
# filtering to high variability periods with diverse cloud motion directions.

# This approach is part of the automated version of the field analysis package
# that is currently under development. The basic problem we're solving is for
# time series of arbitrary length, there will be a mix of steady and variable
# periods. We want to identify the variable periods and then select the most
# diverse cloud motion vectors (CMVs) within those periods. Since CMV is an
# expensive operation, we use the varability score as an initial downselect.
# - Identify variable periods using Variability Score
# - Compute CMVs for each of those periods
# - Select the most diverse CMVs

# #############
# # LOAD DATA #
# #############

# Load 3 days of data from the HOPE campaign.
fn = 'data/hope_melpitz_10s.h5'
pos = pd.read_hdf(fn, mode="r", key="latlon")
pos_utm = spatial.latlon2utm(pos['lat'], pos['lon'])
ts = pd.read_hdf(fn, mode="r", key="data")

# #######################
# # FIND VARIABLE HOURS #
# #######################

# Compute the variability score for each hour. The formulation using lambda
# allows it to be computed in a vectorized mode.
avg_interval = '1h'
vs = ts.resample(avg_interval).apply(
                lambda x: stats.variability_score(x[ts.columns]))

# We want to select hours over the time series that are likely to produce good
# CMVs, since CMV is computationally expensive. Compute the median of the
# variability score for each hour and sort the values in descending order. Save
# the 20 with the highest Variability Score.
vs = vs.median(axis=1).sort_values(ascending=False)
vs = vs.iloc[0:20]
print(vs)

# ################
# # COMPUTE CMVs #
# ################

# Now we'll compute the CMVs for each of those 20 hours.

# Build a holder for the output data
cmvs = pd.DataFrame(columns=["cld_spd", "cld_dir_rad", "df_p95", "ngood", "rval", "stderr", "error_index"])
cmvs_flags = []

# Loop over hours
for date in vs.index:
    # Select the subset of data
    hour = pd.date_range(date, date + pd.to_timedelta('1h'), freq='10s')
    hour = ts.loc[hour]

    # Normalize it to yield something like kt (could actually calculate
    # clearsky if available. Here, we're just using quantiles)
    hourlymax = np.mean(hour.quantile(0.95))
    kt = hour / hourlymax

    # Compute the CMV using the Jamaly method
    cld_spd, cld_dir, dat = cmv.compute_cmv(kt, pos_utm, method='jamaly', options={'minvelocity': 1})

    # Store the global flag
    cmvs_flags.append(dat.flag)

    # Perform a linear regression on the pairs to see the quality of the match
    # between the lag and the distance (i.e. how consistent is velocity over
    # the plant?).
    s, i, r, p, se = linregress(dat.pair_lag[dat.pair_flag == cmv.Flag.GOOD], dat.pair_dists[dat.pair_flag == cmv.Flag.GOOD])

    # Store the outputs
    dat.ngood = np.sum(dat.pair_flag == cmv.Flag.GOOD)
    dat.rval = r
    dat.stderr = se
    cmvs.loc[date] = [cld_spd, cld_dir, hourlymax, dat.ngood, dat.rval, dat.stderr, np.abs(dat.method_data["error_index"])]

# Display the 20 CMVs we just acquired
pd.options.display.max_columns = None
pd.options.display.width = 800
pd.options.display.max_rows = None
print(cmvs)

# ########################
# # SELECT THE BEST CMVs #
# ########################

# First we'll do some filtering on the quality. These are a bit arbitrary and
# will very likely require tuning from dataset to dataset.
ngood_min = 200
rval_min = 0.85
bad_inds = []
for row in cmvs.itertuples():
    if row.ngood < ngood_min:
        bad_inds.append(row.Index)
    elif row.rval < rval_min:
        bad_inds.append(row.Index)
    elif cmvs_flags[row.index(row.Index)] is not None:
        bad_inds.append(row.Index)

# Drop the bad ones from the DF
print(cmvs.loc[bad_inds])
cmvs = cmvs.drop(index=bad_inds)

# This tool gets a diverse angled set of CMVs
# Future updates may be interested in biasing them based on the actual quality
# of the CMV.
vx, vy = spatial.pol2rect(cmvs.cld_spd, cmvs.cld_dir_rad)
indices = cmv.optimum_subset(vx, vy, n=5)
print(cmvs.iloc[indices])

# Plot the vectors visually
plt.figure()
for i, (dx, dy) in enumerate(zip(vx, vy)):
    mylabel = 'All CMVs' if i == 0 else '_nolegend_'
    plt.arrow(0, 0, dx, dy, head_width=1, head_length=1, fc='k', label=mylabel)
for i, (dx, dy) in enumerate(zip(vx.iloc[indices], vy.iloc[indices])):
    mylabel = 'Selected CMVs' if i == 0 else '_nolegend_'
    plt.arrow(0, 0, dx, dy, head_width=1, head_length=1, fc='r', ec='r', label=mylabel)
plt.xlabel('Eastward Velocity (m/s)')
plt.ylabel('Northward Velocity (m/s)')
plt.axis('equal')
plt.legend()
plt.tight_layout()
plt.show()
