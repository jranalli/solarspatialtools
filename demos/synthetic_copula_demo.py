import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from solarspatialtools import spatial
from solarspatialtools.synthirrad.copula import downscale, downscale_multihour, DEFAULT_PARAMS

# This file demonstrates code available in `solarspatialtools.synthirrad.copula`,
# which implements methods described by Widén and Munkhammar [1] for
# downscaling of irradiance using spatiotemporally correlated Gaussian Copulas.
# The implementation is based upon Matlab code shared by the original authors.
#
# [1] Joakim Widén and Joakim Munkhammar, "Spatio-Temporal Downscaling of
# Hourly Solar irradiance Data Using Gaussian Copulas," 2019 IEEE 46th
# Photovoltaic Specialists Conference (PVSC), Chicago, IL, USA, 2019, pp.
# 3172-3178, doi: https://dx.doi.org/10.1109/PVSC40753.2019.8980922.

def demonstrate_downscaling():
    # Demonstrate the downscaling process for 3 individual sites. Two are along
    # the cloud motion vector and a third is perpendicular.

    # Specify the cloud speed and direction
    cloud_spd = 1
    cloud_dir = 0 * 2 * np.pi / 180

    # Hourly clearsky
    mean_csi = 0.52

    # The site positions in meters. Consult spatial.lla2flat or others for help
    # with coordinate transformations from latitude/longitude coordinates.
    e_pos = np.array([0, 450, 0])
    n_pos = np.array([0, 0, 450])

    # A vector of times at which to compute the irradiance. Changing the value
    # of dt changes the temporal resolution of the downscaled time series.
    times = pd.date_range(start='2024-01-01 00:00:00',
                          end='2024-01-01 00:59:59', freq='15s')

    # This function performs the operation, and yields a time series for each
    # site.
    c = downscale(times, e_pos, n_pos, cloud_spd, cloud_dir, mean_csi,
                  DEFAULT_PARAMS, seed=42, scale=True, noneg=True)

    # The first two sites are separated temporally but with perfect correlation
    # due to their alignment with the cloud motion vector. A delay of 450 s is
    # visible. The third site is decorrelated, but aligns temporally with the
    # upwind site.
    plt.plot((times - times[0]).total_seconds(), c, linewidth=1, alpha=0.8)
    plt.xlabel("Time (s)")
    plt.ylabel("Clear-sky index")
    plt.legend(['Upwind site', 'Downwind site', 'Northward Site']);
    plt.show()



def demonstrate_multiple_speeds():
    # This function demonstrates differences in the cloud field produced for
    # multiple cloud speeds and variable mean csi. Note that this methodology
    # produces its time series directly from the copula and does not require
    # a priori simulation of the cloud field. Nonetheless, it is instructive
    # for visualization purposes.

    n = 60  # Num pixels in each axis
    dt = 10  # Time step in seconds
    cloud_dir = 0  # Cloud direction (east)
    cloud_spd_vals = [10, 25, 40]  # cloud speed in m/x
    mean_csi_vals = [0.5, 0.7, 0.9]  # mean clear-sky index

    # Compute the time vector to maintain a size of n, with a spacing of dt
    end_time = (pd.to_timedelta(f'{dt}s') * n).total_seconds()
    times = pd.date_range(start='2024-01-01 00:00:00',
                          end=f'2024-01-01 {int(end_time // 3600):02d}:{int((end_time - (end_time // 3600) * 3600) // 60):02d}:{int(end_time - (end_time // 60) * 60):02d}',
                          freq=f'{dt}s')

    fig, axes = plt.subplots(3, 3, figsize=(12, 12),
                             constrained_layout=True)
    tick_idx = np.arange(len(times))[::max(1, n // 5)]

    # Do this for multiple conditions in multiple plots.
    for row, mean_csi in enumerate(mean_csi_vals):
        for col, cloud_spd in enumerate(cloud_spd_vals):

            # Compute the x & y position values such that we form a north-south
            # rake perpendicular to the eastward cloud motion. The vertical
            # spacing should be equal to the spatiotemporal spacing implied by
            # advection to ensure an evenly spaced grid.
            e_pos = np.zeros(len(times))
            n_pos = np.arange(len(times)) * cloud_spd * dt

            # Downscale each condition and plot.
            c = downscale(times, e_pos, n_pos, cloud_spd, cloud_dir, mean_csi,
                          DEFAULT_PARAMS, seed=42, scale=True, noneg=True)

            ax = axes[row, col]
            im = ax.imshow(c, vmin=0, vmax=1.3)
            ax.set_aspect('equal')
            ax.set_xticks(tick_idx)
            ax.set_yticks(tick_idx)
            ax.set_xticklabels(n_pos[tick_idx] / 1000)
            ax.set_yticklabels(n_pos[tick_idx] / 1000)
            ax.set_title(f'cloud_spd={cloud_spd} m/s, mean_csi={mean_csi}')

    for ax in axes[-1, :]:
        ax.set_xlabel('Distance (km)')

    for ax in axes[:, 0]:
        ax.set_ylabel('Distance (km)')

    fig.colorbar(im, ax=axes, label='Clear-sky index', shrink=0.85);
    plt.show()


def matlab_compare():
    # Compares with the demo results from the original authors' MATLAB code.
    # It also demonstrates the use of the multihour helper function.

    # Cloud speed and direction in radians, provided as arrays for multihour
    cs = np.array([5, 5, 5, 5, 5, 5])
    cd = np.array([0, 0, 0, 0, 0, 0]) * 2 * np.pi / 360

    # Hourly clearsky
    mean_csi = np.array([0.52, 0.71, 0.5, 0.84, 0.63, 0.11])

    # Site positions, projected to East-North coordinate plane
    lat = np.array([21.31236, 21.31303, 21.32357])
    lon = np.array([-158.08463, -158.08505, -158.08424])
    # Epos, Npos = spatial.latlon2lcs(lat, lon, lat[0], lon[0])
    Epos, Npos = spatial.lla2flat(lat, lon, lat[0], lon[0], method="tmerc")

    # The reference times
    times = pd.date_range(start='2024-01-01 00:00:00', end='2024-01-01 00:59:59', freq='15s')

    # Multihour accepts arrays of cs, cd and mean_csi to produce multiple
    # hourly outputs at once. The results are concatenated timeseries for each
    # individual site.
    noneg = True
    scale = True
    c = downscale_multihour(times, Epos, Npos, cs, cd, mean_csi,
                            DEFAULT_PARAMS, seed=42, scale=scale, noneg=noneg)

    # Helper to plot the mean
    n_per_hour = times.shape[0]
    hcsi_block = np.repeat(mean_csi, n_per_hour)

    plt.plot(c, alpha=0.8, linewidth=1)
    plt.step(np.arange(hcsi_block.size), hcsi_block, where='post', color='k',
             linewidth=1, linestyle='--')
    plt.legend()

    plt.show()


if __name__ == '__main__':
    demonstrate_downscaling()
    demonstrate_multiple_speeds()
    matlab_compare()
