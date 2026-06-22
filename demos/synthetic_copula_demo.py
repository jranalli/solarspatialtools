import numpy as np
import pandas as pd

from solarspatialtools import spatial
from solarspatialtools.synthirrad.copula import downscale, downscale_multihour, DEFAULT_PARAMS


def matlab_compare():

    # Cloud speed and direction in radians
    cs = np.array([5, 5, 5, 5, 5, 5])
    cd = np.array([0, 0, 0, 0, 0, 0]) * 2 * np.pi / 360

    # Hourly clearsky
    hcsi = np.array([0.52,0.71,0.5,.995,0.63,0.11])

    # lat
    lat = np.array([21.31236, 21.31303, 21.32357])
    lon = np.array([-158.08463, -158.08505, -158.08424])
    # Epos, Npos = spatial.latlon2lcs(lat, lon, lat[0], lon[0])
    Epos, Npos = spatial.lla2flat(lat, lon, lat[0], lon[0], method="tmerc")

    times = pd.date_range(start='2024-01-01 00:00:00', end='2024-01-01 00:59:59', freq='15s')

    noneg = True
    scale = True

    # # How to execute manually - wrapped by downscale_multihour
    #
    # c = []
    #
    # # Loop over all hours
    # for i in range(len(hcsi)):
    #
    #     # Compute for this hour
    #     cm = downscale(times, Epos, Npos, cs[i], cd[i], hcsi[i], param,
    #                    scale=scale, noneg=noneg)
    #     c.append(cm)
    # c = np.concatenate(c, axis=0)

    c = downscale_multihour(times, Epos, Npos, cs, cd, hcsi, DEFAULT_PARAMS, seed=42, scale=scale, noneg=noneg)


    import matplotlib.pyplot as plt

    # Build a blocky hourly reference aligned with the high-resolution output.
    n_per_hour = times.shape[0]
    hcsi_block = np.repeat(hcsi, n_per_hour)

    plt.plot(c, alpha=0.8)
    plt.step(np.arange(hcsi_block.size), hcsi_block, where='post', color='k', linewidth=2, label='hcsi (hourly step)')
    plt.legend()

    plt.show()




def spatial_compare():

    # Cloud speed and direction in radians
    cs = 1
    cd = 0

    # Hourly clearsky
    hcsi = 0.52

    Epos = np.array([ 0, 450,  0, 450, 0])
    Npos = np.array([ 0,   0, 30,  30, 225])

    times = pd.date_range(start='2024-01-01 00:00:00', end='2024-01-01 00:59:59', freq='15s')

    c = downscale(times, Epos, Npos, cs, cd, hcsi, DEFAULT_PARAMS, seed=42, scale=True,
                  noneg=True)

    import matplotlib.pyplot as plt
    plt.plot(times, c, alpha=0.8)
    plt.legend([1,2,3,4,5])
    plt.show()


def distance_confirm():
    import matplotlib.pyplot as plt

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

    fig, axes = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    tick_idx = np.arange(len(times))[::max(1, n // 5)]

    for row, mean_csi in enumerate(mean_csi_vals):
        for col, cloud_spd in enumerate(cloud_spd_vals):
            # Compute the x & y position values such that we form a north-south rake with perpendicular motion
            # The vertical spacing should be equal to the spatiotemporal spacing implied by advection.
            e_pos = np.zeros(len(times))
            n_pos = np.arange(len(times)) * cloud_spd * dt

            # Downscale
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

if __name__ == '__main__':
    distance_confirm()

    # matlab_compare()
    # spatial_compare()
