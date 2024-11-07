import pvlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from solarspatialtools import irradiance
from solarspatialtools import cmv
from solarspatialtools import spatial

from solarspatialtools.synthirrad.cloudfield import get_timeseries_stats, cloudfield_timeseries


# This file demonstrates code available in `solarspatialtools.synthirrad.cloudfield`, which implements methods described
# by Lave et al. [1] for generation of synthetic cloud fields that can be used to simulate high frequency solar
# irradiance data. Some aspects of the implementation diverge slightly from the initial paper to follow a subsequent
# code implementation of the method shared by the original authors.

# [1] Matthew Lave, Matthew J. Reno, Robert J. Broderick, "Creation and Value of Synthetic High-Frequency Solar Inputs
# for Distribution System QSTS Simulations," 2017 IEEE 44th Photovoltaic Specialist Conference (PVSC), Washington, DC,
# USA, 2017, pp. 3031-3033, doi: https://dx.doi.org/10.1109/PVSC.2017.8366378.



# #### Load Timeseries Data

# The model attempts to create representative variability to match that observed from a reference time series. In this
# case, we'll process one of the 1-second resolution timeseries from the HOPE-Melpitz campign. We will load the data and
# convert it to clearsky index. We'll then use it to calculate the cloud speed.

datafn = "data/hope_melpitz_1s.h5"
twin = pd.date_range('2013-09-08 9:15:00', '2013-09-08 10:15:00', freq='1s', tz='UTC')
data = pd.read_hdf(datafn, mode="r", key="data")

# Load the sensor positions
pos = pd.read_hdf(datafn, mode="r", key="latlon")
pos_utm = pd.read_hdf(datafn, mode="r", key="utm")

# Compute clearsky ghi and clearsky index
loc = pvlib.location.Location(np.mean(pos['lat']), np.mean(pos['lon']))
cs_ghi = loc.get_clearsky(data.index, model='simplified_solis')['ghi']
cs_ghi = 1000/max(cs_ghi) * cs_ghi # Normalize to 1000 W/m^2
kt = irradiance.clearsky_index(data, cs_ghi, 2)

# Compute the cloud motion vector
cld_spd, cld_dir, _ = cmv.compute_cmv(kt, pos_utm, reference_id=None, method='jamaly')
cld_vec_rect = spatial.pol2rect(cld_spd, cld_dir)
print(f"Cld Speed  {cld_spd:8.2f}, Cld Dir {np.rad2deg(cld_dir):8.2f}°")

# We want to describe how the sensors are distributed in the cloud motion vector direction. So we'll rotate the
# positions of the entire field to align with the CMV in the +X direction. This will allow us to describe positions of
# sensors within the field with respect to the motion of clouds, which seldom aligns with the cardinal directions.

# Rotation by -cld_dir to make CMV align with X Axis
rot = spatial.rotate_vector((pos_utm['E'], pos_utm['N']), theta=-cld_dir)
pos_utm_rot = pd.DataFrame({'X': rot[0] - np.min(rot[0]), 'Y': rot[1] - np.min(rot[1])}, index=pos_utm.index)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].scatter(pos_utm['E'], pos_utm['N'])
axs[0].set_title('Original Field')
axs[0].quiver(pos_utm['E'][40], pos_utm['N'][40], 200 * cld_vec_rect[0], 200 * cld_vec_rect[1], scale=10, scale_units='xy')
axs[0].set_xlabel('East')
axs[0].set_ylabel('North')
axs[1].scatter(pos_utm_rot['X'], pos_utm_rot['Y'])
axs[1].quiver(pos_utm_rot['X'][40], pos_utm_rot['Y'][40], 200*cld_spd, 0, scale=10, scale_units='xy')
axs[1].set_title('Rotated Field')
axs[1].set_xlabel('CMV Direction')
axs[1].set_ylabel('CMV Dir + 90 deg')
axs[0].set_aspect('equal')
axs[1].set_aspect('equal')
plt.show()

# The scaling of the cloud field is based on variability as expressed through statistical properties of the time series.
# So we'll extract those in advance. We do so for a single sensor (number 40) that is centrally located in the field,
# though more detailed analysis could consider representing properties for the entire field.
# - `ktmean` - The mean clearsky index
# - `kt1pct` - The 1st percentile of clearsky index, used similar to a minimum
# - `ktmax` - The maximum clearsky index (shows cloud enhancement)
# - `frac_clear` - Fraction of clear sky conditions in time series (characterized as kt > 0.95)
# - `vs` - The variability score of the clearsky index
# - `weights` - The weights are calculated from the magnitude-squared of the various wavelet modes.

ktmean, kt1pct, ktmax, frac_clear, vs, weights, scales = get_timeseries_stats(kt[40], plot=False)
print(f"ktmean: {ktmean:8.2f}")
print(f"kt1pct: {kt1pct:8.2f}")
print(f"ktmax: {ktmax:8.2f}")
print(f"frac_clear: {frac_clear:8.2f}")
print(f"vs: {vs:8.2f}")

# Plot the wavelet scales
plt.plot(scales, weights)
plt.xlabel('Wavelet Mode Scale')
plt.ylabel('Assigned Weight')
plt.show()

# Since we rotated the sensor positions, we can now calculate the overall spatial size of the distribution along and
# perpendicular to the cloud motion vector. We'll also look at the dureation of the time series (in this case 1 hour)
# and its temporal resolution (1 second).

# Using the cloud speed we can relate these spatial dimensions to time dimensions. When we generate the cloud field, we
# will assume that each pixel in the field represents a 1-second step in time. So moving 1 pixel within the field along
# the X axis represents either a 1 second shift upwind or downwind in space, or a 1 second shift of the time axis at a
# fixed spatial position as clouds transit the field. Moving 1 pixel along the Y axis will always represent a 1 second
# spatial shift perpendicular to the cloud motion vector, since no motion occurs in that direction.

x_extent = np.abs(np.max(pos_utm_rot['X']) - np.min(pos_utm_rot['X']))
y_extent = np.abs(np.max(pos_utm_rot['Y']) - np.min(pos_utm_rot['Y']))
t_extent = (np.max(twin) - np.min(twin)).total_seconds()
dt = (twin[1] - twin[0]).total_seconds()

spatial_time_x = x_extent / cld_spd
spatial_time_y = y_extent / cld_spd

xt_size = int(np.ceil(spatial_time_x + t_extent))
yt_size = int(np.ceil(spatial_time_y))

print(f"X Extent: {x_extent:8.2f} m, Y Extent: {y_extent:8.2f} m")
print(f"Time Extent: {t_extent:8.2f} s, Time Resolution: {dt:8.2f} s")
print(f"Field Size: {xt_size}x{yt_size}")

# The function `cloudfield_timeseries` generates a cloud field from which time series can be sampled. The field is
# generated by creating a random field of cloudiness, then scaling it to match the clear sky condition properties of the
# reference time series. The output field's first axis (rows) represents the spatial dimension perpendicular to the
# cloud motion vector. The second axis (columns) represent the spatial dimension along the cloud motion vector, which
# coincides with time axis.

# Each pixel represents a time step of 1 second, either in literal time, or associated with a spatial shift of the
# equivalent of 1 second of cloud motion. In this case, where the cloud velocity is around 20 m/s, this implies that a
# shift along either axis corresponds to a 20 m spatial shift.

np.random.seed(42)  # Seed for repeatability

field_final = cloudfield_timeseries(weights, scales, (xt_size, yt_size), frac_clear, ktmean, ktmax, kt1pct)

plt.imshow(field_final.T, aspect='equal', cmap='viridis')
plt.xlabel('Time and X axis position')
plt.ylabel('Y axis position')
plt.show()

# We can extract the time series at points in the field. We need to first convert our spatial positions into a spatially
# based coordinate system. We can then choose that starting point as a location for a time series at that point. The
# time series will extend along the x-axis with each time series at a length of `t_extent` seconds.

# One consequence of this approach that is important to note is that any two points that are located precisely
# up/down-wind from each other will have identical time series, albeit with a delay associated with the cloud motion.
# This is visible in the results below in which the two sensors are nearly aligned with the cloud motion, but have an
# upwind/downwind separation.

# Convert space to time to extract time series
xpos = pos_utm_rot['X'] - np.min(pos_utm_rot['X'])
ypos = pos_utm_rot['Y'] - np.min(pos_utm_rot['Y'])
xpos_temporal = xpos / cld_spd
ypos_temporal = ypos / cld_spd

# Extract simulated time series at all sensor positions
sim_kt = pd.DataFrame(index=twin, columns=pos_utm_rot.index)
for sensor in pos_utm_rot.index:
    # Negate X positions for upwind/downwind positions, so that downwind shows later
    x = int(max(xpos_temporal) - xpos_temporal[sensor])
    y = int(ypos_temporal[sensor])

    # Select a time series of length t_extent starting at the x,y position
    sim_kt[sensor] = field_final[x:x + int(t_extent) + 1, y]

# Create two side by side figures
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(sim_kt[[60, 70]])
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Clearsky Index')
axs[0].legend(['Upwind', 'Downwind'])
import matplotlib.dates as mdates

myFmt = mdates.DateFormatter('%H:%M')
axs[0].xaxis.set_major_formatter(myFmt)

axs[1].scatter(pos_utm_rot['X'], pos_utm_rot['Y'])
axs[1].scatter(pos_utm_rot['X'][[60, 70]], pos_utm_rot['Y'][[60, 70]], color='r')
for i, txt in enumerate([60, 70]):
    axs[1].annotate(txt, (pos_utm_rot['X'][txt], pos_utm_rot['Y'][txt]))
    axs[1].quiver(pos_utm_rot['X'][40], pos_utm_rot['Y'][40], 200 * cld_spd, 0, scale=10, scale_units='xy')
axs[1].set_aspect('equal')
axs[1].set_xlabel('X Position')
axs[1].set_ylabel('Y Position')
plt.show()


# Two sensors that are separated in the perpendicular direction to the motion experience differences in the time series
# due to spatial position discrepancies, but would see large scale phenomena at the same time. In part, that's because
# the weights for this particular cloud field are biased towards large scales.
# Create two side by side figures
fig, axs = plt.subplots(1, 2, figsize=(12, 5))
axs[0].plot(sim_kt[[18, 38]])
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Clearsky Index')
axs[0].legend(['A', 'B'])
import matplotlib.dates as mdates
myFmt = mdates.DateFormatter('%H:%M')
axs[0].xaxis.set_major_formatter(myFmt)

axs[1].scatter(pos_utm_rot['X'], pos_utm_rot['Y'])
axs[1].scatter(pos_utm_rot['X'][[18,38]], pos_utm_rot['Y'][[18,38]], color='r')
for i, txt in enumerate([18, 38]):
    axs[1].annotate(txt, (pos_utm_rot['X'][txt], pos_utm_rot['Y'][txt]))
    axs[1].quiver(pos_utm_rot['X'][40], pos_utm_rot['Y'][40], 200*cld_spd, 0, scale=10, scale_units='xy')
axs[1].set_aspect('equal')
axs[1].set_xlabel('X Position')
axs[1].set_ylabel('Y Position')
plt.show()