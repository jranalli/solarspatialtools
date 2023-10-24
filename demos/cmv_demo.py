import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from solartoolbox import cmv, spatial

datafile = "data/sample_field.h5"
ref = 'CMB-022-5'


pos_utm = pd.read_hdf(datafile, mode="r", key="latlon")
df = pd.read_hdf(datafile, mode="r", key="data_a")


cld_spd_gag, cld_dir_gag, dat_gag = cmv.compute_cmv(df, pos_utm,
                                                    reference_id=None,
                                                    method='gagne',
                                                    corr_scaling='coeff')
cld_spd_jam, cld_dir_jam, dat_jam = cmv.compute_cmv(df, pos_utm,
                                                    reference_id=None,
                                                    method='jamaly',
                                                    corr_scaling='coeff',
                                                    options={'minvelocity': 1})
tgts = []
dels = []
for pair, delay in zip(dat_jam.allpairs, dat_jam.pair_lag):
    if ref in pair:
        if ref == pair[0]:
            tgt = pair[1]
            deli = delay
        else:
            tgt = pair[0]
            deli = -delay
        tgts.append(tgt)
        dels.append(deli)


pos_utm['delay'] = np.zeros_like(pos_utm['E'])
pos_utm['delay'][tgts] = dels
pos_utm['delay'][ref] = 0

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



plt.figure()
plt.scatter(pos_utm['E'], pos_utm['N'], c=-pos_utm['delay'])


vscale = 100
velvec = np.array(spatial.unit(spatial.pol2rect(cld_spd_jam, cld_dir_jam))) * vscale
plt.arrow(pos_utm['E'][ref], pos_utm['N'][ref],
          velvec[0], velvec[1],
          length_includes_head=True,
          width=7,
          head_width=20,
          color='green')
plt.clim(-100, 100)
plt.colorbar()
plt.xlabel('E')
plt.ylabel('N')
plt.title(f'All Predicted Positions')
axes = plt.gca()
axes.xaxis.set_ticklabels([])
axes.yaxis.set_ticklabels([])

plt.show()
