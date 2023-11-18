import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pvlib.location

from solartoolbox import spatial, signalproc, stats


"""
This demo will compute transfer functions for a plant within the HOPE Melpitz
dataset. The plant consists of a linear subset of points aligned in the N-S
direction, and during an hour-long window with known cloud motion vector. 

It will compute some variability statistics and plot the different models of 
the transfer function.
"""


def main():
    # ##### INPUTS ##### #

    # The file name
    fn = "data/hope-melpitz-qcinterp.h5"

    # This model has a known cloud speed and direction during this time
    twin = pd.date_range('2013-09-08 9:15:00', '2013-09-08 10:15:00',
                         freq='1s', tz='UTC')
    cloud_speed = 20  # m/s
    cloud_dir = (0, 1)  # as x-y vector

    # The reference site within the plant, and the plant subset (N-S)
    ref_id = 40
    plant_pts = [60, 79, 92, 43, 42, 40, 86, 7, 14, 70]
    
    # ##### DATA SETUP ##### #

    # Read Data from HDF file
    pos = pd.read_hdf(fn, mode="r", key="latlon")
    ts = pd.read_hdf(fn, mode="r", key="data")

    # Project them to UTM
    pos_utm = spatial.latlon2utm(pos['lat'], pos['lon'])
    # Select just the points in the plant
    pos_sub = pos.loc[plant_pts]
    pos_sub_utm = pos_utm.loc[plant_pts]

    # Downselect the time series to just the points in the plant, and just
    # the time window of interest.
    ts_sub = ts.loc[twin]

    # Select the Input (reference) time series, and compute the Output
    # (aggregate time series)
    ts_in = ts_sub[ref_id]
    ts_sub = ts_sub[plant_pts]
    ts_agg = ts_sub.sum(axis=1) / len(plant_pts)

    # Calculate the clear sky index for the time series (using PVLIB functions)
    loc = pvlib.location.Location(np.mean(pos_sub['lat']),
                                  np.mean(pos_sub['lon']))
    cs_ghi = loc.get_clearsky(ts_in.index, model='simplified_solis')['ghi']
    kt_in = pvlib.irradiance.clearsky_index(ts_in, cs_ghi, 2)
    kt_agg = pvlib.irradiance.clearsky_index(ts_agg, cs_ghi, 2)

    # ##### TRANSFER FUNCTIONS ##### #

    # Apply the filter from the Cloud Advection Model
    camfilt = signalproc.get_camfilter(pos_sub_utm, cloud_speed,
                                       cloud_dir, pos_utm.loc[ref_id])
    ts_cam = signalproc.apply_filter(ts_in, camfilt)
    kt_cam = pvlib.irradiance.clearsky_index(ts_cam, cs_ghi)

    # Calculate Marcos Model Filtered Signal
    marcosfilt = signalproc.get_marcosfilter(10, camfilt.index)
    ts_marcos = signalproc.apply_filter(ts_in, marcosfilt)
    kt_marcos = pvlib.irradiance.clearsky_index(ts_marcos, cs_ghi)

    # Calculate WVM filtered signal
    kt_wvm, _, _ = pvlib.scaling.wvm(kt_in, pos_sub_utm, cloud_speed)
    ts_wvm = kt_wvm * cs_ghi  # Convert from kt to ghi

    # Calculate Real Data TF
    tf_bas, coh_bas = signalproc.averaged_tf(ts_in, ts_agg, navgs=8,
                                             overlap=0.5, detrend=None)
    tf_wvm, coh_wvm = signalproc.averaged_tf(ts_in, ts_wvm, navgs=8,
                                             overlap=0.5, detrend=None)
    # CAM and Marcos are already nice filter computations.

    # Plot TFs
    # Make the frequencies not wrap back around
    signalproc.cleanfreq(camfilt)
    signalproc.cleanfreq(marcosfilt)

    # ##### OUTPUT STATISTICS ##### #

    # Set up a few time scales for the statistics
    dt = (ts_in.index[1] - ts_in.index[0]).total_seconds()
    tau5 = 5/dt
    tau30 = 30/dt
    tau60 = 60/dt

    # Compute and print statistics
    # RMSE is how well the model-smoothing matches the real curve.
    # Others are comparisons of various varibility metrics
    print("     Metric      \tSing    \tAgg       \tCAM     \tWVM     \tMarcos")
    print("     RMSE:     \t{:8.3f}\t{:8.3f}\t{:8.3f}\t{:8.3f}\t{:8.3f}".format(
            stats.rmse(kt_agg, kt_in),
            0.0,
            stats.rmse(kt_agg, kt_cam),
            stats.rmse(kt_agg, kt_wvm),
            stats.rmse(kt_agg, kt_marcos)))
    print("     VI:    \t{:8.1f}\t{:8.1f}\t{:8.1f}\t{:8.1f}\t{:8.1f}".format(
        stats.variability_index(ts_in, cs_ghi, norm=False),
        stats.variability_index(ts_agg, cs_ghi, norm=False),
        stats.variability_index(ts_cam, cs_ghi, norm=False),
        stats.variability_index(ts_wvm, cs_ghi, norm=False),
        stats.variability_index(ts_marcos, cs_ghi, norm=False)))
    print("     VS_5:    \t{:8.2f}\t{:8.2f}\t{:8.2f}\t{:8.2f}\t{:8.2f}".format(
        stats.variability_score(ts_in, tau=tau5, moving_avg=True, pct=True),
        stats.variability_score(ts_agg, tau=tau5, moving_avg=True, pct=True),
        stats.variability_score(ts_cam, tau=tau5, moving_avg=True, pct=True),
        stats.variability_score(ts_wvm, tau=tau5, moving_avg=True, pct=True),
        stats.variability_score(ts_marcos, tau=tau5, moving_avg=True, pct=True)))
    print("     VS30:    \t{:8.2f}\t{:8.2f}\t{:8.2f}\t{:8.2f}\t{:8.2f}".format(
        stats.variability_score(ts_in, tau=tau30, moving_avg=True, pct=True),
        stats.variability_score(ts_agg, tau=tau30, moving_avg=True, pct=True),
        stats.variability_score(ts_cam, tau=tau30, moving_avg=True, pct=True),
        stats.variability_score(ts_wvm, tau=tau30, moving_avg=True, pct=True),
        stats.variability_score(ts_marcos, tau=tau30, moving_avg=True, pct=True)))
    print("     VS60:    \t{:8.2f}\t{:8.2f}\t{:8.2f}\t{:8.2f}\t{:8.2f}".format(
        stats.variability_score(ts_in, tau=tau60, moving_avg=True, pct=True),
        stats.variability_score(ts_agg, tau=tau60, moving_avg=True, pct=True),
        stats.variability_score(ts_cam, tau=tau60, moving_avg=True, pct=True),
        stats.variability_score(ts_wvm, tau=tau60, moving_avg=True, pct=True),
        stats.variability_score(ts_marcos, tau=tau60, moving_avg=True, pct=True)))

    # ##### PLOTS ##### #

    # Plot Comparison of Timeseries outputs of models
    plt.figure(figsize=(4, 3))
    plt.plot(ts_in.index, ts_in, 'k',
             ts_agg.index, ts_agg,
             ts_cam.index, ts_cam,
             ts_wvm.index, ts_wvm,
             ts_marcos.index, ts_marcos)
    plt.legend(['Input', 'Output', 'CAM', 'WVM', 'Marcos'])
    plt.ylabel('GHI')
    plt.xlabel('Time')
    plt.tight_layout()

    # Plot comparison of transfer functions (3 part plot)
    plt.figure(figsize=[4, 6])
    ax1 = plt.subplot(311)
    plt.semilogx(tf_bas.index, np.abs(tf_bas),
                 camfilt.index, np.abs(camfilt),
                 tf_wvm.index, np.abs(tf_wvm),
                 marcosfilt.index, np.abs(marcosfilt))
    plt.legend(['Data', 'CAM', 'WVM', 'Marcos'])
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.xlim([1e-4, 0.5])
    plt.ylabel('TF Mag')

    ax2 = plt.subplot(312, sharex=ax1)
    plt.semilogx(tf_bas.index, np.angle(tf_bas),
                 camfilt.index, np.angle(camfilt),
                 tf_wvm.index, np.angle(tf_wvm),
                 marcosfilt.index, np.angle(marcosfilt))
    plt.setp(ax2.get_xticklabels(), visible=False)
    plt.ylabel('TF Phase (rad)')

    plt.subplot(313, sharex=ax1)
    plt.semilogx(tf_bas.index, coh_bas,
                 [0], [0],
                 tf_wvm.index, coh_wvm)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Coherence')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
