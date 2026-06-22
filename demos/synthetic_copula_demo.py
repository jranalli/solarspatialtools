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


if __name__ == '__main__':
    matlab_compare()
    spatial_compare()
