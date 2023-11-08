import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from solartoolbox import cmv, spatial

datafile = "data/sample_plant_2.h5"
pos_utm = pd.read_hdf(datafile, mode="r", key="latlon")
df = pd.read_hdf(datafile, mode="r", key="data_a")

hourlymax = np.mean(df.quantile(0.95))
kt = df / hourlymax

cld_spd_gag, cld_dir_gag, dat_gag = cmv.compute_cmv(kt, pos_utm,
                                                    reference_id=None,
                                                    method='gagne',
                                                    corr_scaling='coeff')
cld_spd_jam, cld_dir_jam, dat_jam = cmv.compute_cmv(kt, pos_utm,
                                                    reference_id=None,
                                                    method='jamaly',
                                                    corr_scaling='coeff')

print("Method     Speed  Angle:rad  Angle:°   N_good")
print(f"Gagne   {cld_spd_gag:8.2f} {cld_dir_gag:10.2f} {np.rad2deg(cld_dir_gag):8.2f} {sum(dat_gag.pair_flag == cmv.Flag.GOOD):8,}")
print(f"Jamaly  {cld_spd_jam:8.2f} {cld_dir_jam:10.2f} {np.rad2deg(cld_dir_jam):8.2f} {sum(dat_jam.pair_flag == cmv.Flag.GOOD):8,}")


plt.title('Result Data for Jamaly Method')
plt.scatter(dat_jam.pair_lag, dat_jam.pair_dists, c=dat_jam.corr_lag, vmin=0, vmax=1, s=1)
plt.plot([-150, 200], cld_spd_jam * np.array([-150, 200]), 'r--', linewidth=2)  # Best Fit
plt.ylim([-1500, 1000])
plt.xlim([-150, 900])
plt.colorbar().set_label("Peak Correlation (-)")
plt.xlabel('Time Lag Between Signals (s)')
plt.ylabel('Windward Separation Distance (m)')
plt.legend(['Point Correlation', 'Speed Fit'])

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

# Extract only the lags associated with the reference. Correct the lag's sign depending on which was the first point in the pair.
ref = 'CMB-022-5'
points = []
lags = []
for pair, lag in zip(dat_jam.allpairs, dat_jam.pair_lag):
    if ref in pair:
        point = pair[1] if ref == pair[0] else pair[0]
        lag_i = lag if ref == pair[0] else -lag
        points.append(point)
        lags.append(lag_i)

# Insert data back into the DataFrame for plotting
pos_utm['lag'] = np.zeros_like(pos_utm['E'])
pos_utm['lag'][points] = lags
pos_utm['lag'][ref] = 0


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