import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from solartoolbox import cmv, spatial

# #############
# # READ DATA #
# #############

# This is the name of the default datafile
# It contains the definition of a plant layout as well as two time periods
# of data with expected CMVs. These data consist of normalized combiner
# current time series from a utility scale PV plant.
datafile = "data/sample_plant_2.h5"
pos_utm = pd.read_hdf(datafile, mode="r", key="utm")
df = pd.read_hdf(datafile, mode="r", key="data_a")

# The CMV routines in principle would work on non-normalized data, but have
# some quality control parameters that are scaled to work with clearsky index.
# So we'll normalize the current data to resemble clearsky index here. The
# alternative would be to adjust the quality control parameters to work with
# the scale of the raw data.
hourlymax = np.mean(df.quantile(0.95))
kt = df / hourlymax

# ################
# # COMPUTE CMVs #
# ################

# The cmv.compute_cmv function performs the cross-correlation analysis to
# determine the best fit wind speed and direction for the data. The method
# parameter can be used to choose between the two methods.
cld_spd_gag, cld_dir_gag, dat_gag = cmv.compute_cmv(kt, pos_utm,
                                                    reference_id=None,
                                                    method='gagne')
cld_spd_jam, cld_dir_jam, dat_jam = cmv.compute_cmv(kt, pos_utm,
                                                    reference_id=None,
                                                    method='jamaly')

# Print out some results for these cases
print("Method     Speed  Angle:rad  Angle:°   N_good")
print(f"Gagne   {cld_spd_gag:8.2f} {cld_dir_gag:10.2f} {np.rad2deg(cld_dir_gag):8.2f} {sum(dat_gag.pair_flag == cmv.Flag.GOOD):8,}")
print(f"Jamaly  {cld_spd_jam:8.2f} {cld_dir_jam:10.2f} {np.rad2deg(cld_dir_jam):8.2f} {sum(dat_jam.pair_flag == cmv.Flag.GOOD):8,}")


# ################
# # PLOT RESULTS #
# ################

# This plot shows the relationship between the time lag computed from the
# cross-correlation analysis and the separation distance between each pair in
# the final CMV-wise direction. Ideally, the time lag should be linearly
# related to the separation distance, with the slope of the line being the
# cloud motion speed.
plt.title('Result Data for Jamaly Method')
plt.scatter(dat_jam.pair_lag, dat_jam.pair_dists, c=dat_jam.corr_lag, vmin=0, vmax=1, s=1)
plt.plot([-150, 200], cld_spd_jam * np.array([-150, 200]), 'r--', linewidth=2)  # Best Fit
plt.ylim([-1500, 1000])
plt.xlim([-150, 900])
plt.colorbar().set_label("Peak Correlation (-)")
plt.xlabel('Time Lag Between Signals (s)')
plt.ylabel('Windward Separation Distance (m)')
plt.legend(['Point Correlation', 'Speed Fit'])

# This plot shows the same data as above, but highlights the points that were
# flagged as GOOD by the quality control routine. These points are used to
# determine the actual best fit cloud motion speed and direction.
plt.figure()
plt.title('Highlighted GOOD Points for Jamaly Method')
plt.scatter(dat_jam.pair_lag, dat_jam.pair_dists, c=dat_jam.corr_lag, vmin=0, vmax=1, s=1)
plt.plot([-150, 200], cld_spd_jam * np.array([-150, 200]), 'r--', linewidth=2)  # Best Fit
plt.plot(dat_jam.pair_lag[dat_jam.pair_flag == cmv.Flag.GOOD],
         dat_jam.pair_dists[dat_jam.pair_flag == cmv.Flag.GOOD], 'mo', markersize=1)
plt.ylim([-1500, 1000])
plt.xlim([-150, 900])
plt.colorbar().set_label("Peak Correlation (-)")
plt.xlabel('Time Lag Between Signals (s)')
plt.ylabel('Windward Separation Distance (m)')
plt.legend(['Point Correlation', 'GOOD Points', 'Speed Fit'])
plt.show()

# Extract only the lags associated with a given reference sensor from within
# the plant. Correct the lag's sign depending on which was the first point in
# the pair to accommodate upwind/downwind differences.
ref = 'CMB-022-5'
points = []
lags = []
for pair, lag in zip(dat_jam.allpairs, dat_jam.pair_lag):
    if ref in pair:
        point = pair[1] if ref == pair[0] else pair[0]
        lag_i = lag if ref == pair[0] else -lag
        points.append(point)
        lags.append(lag_i)

# Insert the modified data back into the DataFrame for plotting
pos_utm['lag'] = np.zeros_like(pos_utm['E'])
pos_utm.loc[points, 'lag'] = lags
pos_utm.loc[ref, 'lag'] = 0


# Generate a plot showing the delays relative to the reference sensor
# throughout the entire plant. The vector indicates the CMV direction.
plt.scatter(pos_utm['E'], pos_utm['N'], c=-pos_utm['lag'], cmap='viridis')
vscale = 100
velvec = np.array(spatial.unit(spatial.pol2rect(cld_spd_jam, cld_dir_jam))) * vscale
plt.arrow(pos_utm['E'][ref], pos_utm['N'][ref], velvec[0], velvec[1], length_includes_head=True, width=7, head_width=20, color='red')
plt.clim(-60, 60)
plt.colorbar()
plt.xlabel('E')
plt.ylabel('N')
plt.title(f'Lag Relative to Ref (s)')
axes = plt.gca()
axes.xaxis.set_ticklabels([])
axes.yaxis.set_ticklabels([])
