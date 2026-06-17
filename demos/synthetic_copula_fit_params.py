import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import pvlib

from solarspatialtools import irradiance


datafile = "data/hope_melpitz_10s.h5"
pos_utm = pd.read_hdf(datafile, mode="r", key="utm")
pos_latlon = pd.read_hdf(datafile, mode="r", key="latlon")
ts_data = pd.read_hdf(datafile, mode="r", key="data")
ts_data = ts_data[ts_data[40]>10]

loc = pvlib.location.Location(np.mean(pos_latlon['lat'][40]), np.mean(pos_latlon['lon'][40]))
cs_ghi = loc.get_clearsky(ts_data.index, model='simplified_solis')['ghi']
kt_in = irradiance.clearsky_index(ts_data, cs_ghi, 2.0)

# plt.plot(cs_ghi)
# plt.plot(kt_in)
# plt.show()


#### GENERATED CODE

# 1) Hourly mean for each channel, aligned back to each original sample
hour_key = pd.DatetimeIndex(kt_in.index).floor("h")
hourly_mean_per_sample = kt_in.groupby(hour_key).transform("mean")

# 2) Define bins: 0.0-0.1, 0.1-0.2, ..., 1.2-1.3
bin_edges = np.round(np.arange(0.0, 1.31, 0.1), 1)
bin_labels = [f"{hi:.1f}" for hi in bin_edges[1:]]

# 3) Bin each sample according to the hourly mean of its channel/hour
kt_bins = hourly_mean_per_sample.apply(
    lambda s: pd.cut(
        s,
        bins=bin_edges,
        labels=bin_labels,
        include_lowest=True,
        right=True,
    )
)

# 4) Put values into long form so you can group them by bin
binned_kt = (
    kt_in.stack()
    .rename("kt")
    .to_frame()
    .assign(
        hourly_mean=hourly_mean_per_sample.stack().values,
        bin=kt_bins.stack().values,
    )
    .dropna(subset=["bin"])
)

# binned_kt now has one row per timestamp/channel sample
print(binned_kt.head())

# Bin labels are now "less than listed value"
binned = [binned_kt.loc[binned_kt["bin"] == label, "kt"].values for label in bin_labels]

# Histogram bin edges at 0.5 intervals across the plotted kt range
hist_bin_edges = np.arange(0.0, 1.5, 0.05)

for i in range(len(binned)):
    plt.hist(binned[i], bins=hist_bin_edges)
    plt.xlim([0.0, 1.5])
    plt.figure()
    plt.plot(binned[i])
    plt.show()