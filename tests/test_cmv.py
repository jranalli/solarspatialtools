import pytest
from pytest import approx

import numpy as np
import pandas as pd

from solartoolbox import cmv


@pytest.fixture(params=[0, 45, 60, -60, 90, 124.5, 340.5])
def theta_deg(request):
    return request.param


@pytest.fixture(params=[2, 5, 10, 30])
def velocity(request):
    return request.param


@pytest.fixture(params=['jamaly', 'gagne'])
def mode(request):
    return request.param


def test_cmv_artificial(theta_deg, velocity, mode):
    # Generate a dataframe with 9 points arranged in a 3x3 square grid
    # The points are arranged in a grid with 1m spacing
    # The origin is in the center of the grid
    # The points are numbered as follows:
    # 6 7 8
    # 3 4 5
    # 0 1 2
    # The reference point is 4
    # The CMV is 1m/s at 45°
    # The expected result is 1m/s at 45°
    pos_utm = pd.DataFrame(index=range(9), columns=['E', 'N'], dtype=float)
    spacing = 10.0
    pos_utm.loc[0] = [-spacing, -spacing]
    pos_utm.loc[1] = [0.0, -spacing]
    pos_utm.loc[2] = [spacing, -spacing]
    pos_utm.loc[3] = [-spacing, 0.0]
    pos_utm.loc[4] = [0.0, 0.0]
    pos_utm.loc[5] = [spacing, 0.0]
    pos_utm.loc[6] = [-spacing, spacing]
    pos_utm.loc[7] = [0.0, spacing]
    pos_utm.loc[8] = [spacing, spacing]

    # Velocity direction
    theta = theta_deg * np.pi/180.0

    # delays between points are deterministic
    delays = np.zeros(9)
    delays[4] = 0
    delays[5] = spacing*np.cos(theta)/velocity
    delays[7] = spacing*np.sin(theta)/velocity
    delays[3] = -spacing*np.cos(theta)/velocity
    delays[1] = -spacing*np.sin(theta)/velocity
    delays[8] = spacing*np.sqrt(2)*np.cos(theta-45*np.pi/180)/velocity
    delays[2] = -spacing*np.sqrt(2)*np.sin(theta-45*np.pi/180)/velocity
    delays[6] = spacing*np.sqrt(2)*np.sin(theta-45*np.pi/180)/velocity
    delays[0] = -spacing*np.sqrt(2)*np.cos(theta-45*np.pi/180)/velocity

    # Generate the base signal. High enough sample rate and T is required
    # to fully resolve the delays. We use noise to ensure broad frequency
    # content.
    fs = 500  # sample rate
    T = 500.0  # total seconds
    np.random.seed(2023)
    t = np.linspace(0, T, int(T*fs), endpoint=False)  # time variable
    noise = np.random.random(len(t))/5  # add broadband noise
    x = 0.5*np.sin(2*np.pi*2*t) + noise  # noisy signal

    # The individual signals are delayed versions of the original. Noise is
    # shifted rather than regenerated to ensure broadband coherence.
    signals = np.zeros((9, len(x)))
    signals[4, :] = x
    for i, delay in enumerate(delays):
        signals[i, :] = np.roll(x, int(delay*fs))
    df = pd.DataFrame(signals.T, index=pd.TimedeltaIndex(t, 's'))

    # Compute the CMV
    cld_spd, cld_dir, dat = cmv.compute_cmv(df, pos_utm,
                                            reference_id=4,
                                            method=mode)

    # Convert the dir to positive angle, because compute_cmv is always pos
    if theta < 0:
        theta += 2*np.pi
    if theta > 2*np.pi:
        theta -= 2*np.pi

    assert (cld_spd == approx(velocity, rel=0.01))

    # Directions sometimes get wonky near the 0/360 boundary
    assert (cld_dir == approx(theta, rel=0.01) or
            cld_dir == approx(theta+2*np.pi, rel=0.01))


def test_cmv_gagne_data():
    datafile = "../docs/sphinx/source/demos/data/sample_plant_1.h5"
    pos_utm = pd.read_hdf(datafile, mode="r", key="utm")
    df = pd.read_hdf(datafile, mode="r", key="data_a")

    hourlymax = np.mean(df.quantile(0.95))
    kt = df / hourlymax

    cld_spd_gag, cld_dir_gag, dat_gag = cmv.compute_cmv(kt, pos_utm,
                                                        reference_id=None,
                                                        method='gagne')

    # Known values calculated on 11/3/2023
    assert cld_spd_gag == approx(10.53, abs=0.01)
    assert cld_dir_gag == approx(3.68, abs=0.01)
    assert sum(dat_gag.pair_flag == cmv.Flag.GOOD) == 574


def test_cmv_jamaly_data():
    datafile = "../docs/sphinx/source/demos/data/sample_plant_1.h5"
    pos_utm = pd.read_hdf(datafile, mode="r", key="utm")
    df = pd.read_hdf(datafile, mode="r", key="data_a")

    hourlymax = np.mean(df.quantile(0.95))
    kt = df / hourlymax

    cld_spd_jam, cld_dir_jam, dat_jam = cmv.compute_cmv(kt, pos_utm,
                                                        reference_id=None,
                                                        method='jamaly')

    assert cld_spd_jam == approx(10.54, abs=0.01)
    assert cld_dir_jam == approx(3.29, abs=0.01)
    assert sum(dat_jam.pair_flag == cmv.Flag.GOOD) == 12709

def test_cmv_jamaly_data_ref():
    # A test using a reference when we calculate the CMV

    datafile = "../docs/sphinx/source/demos/data/sample_plant_1.h5"
    pos_utm = pd.read_hdf(datafile, mode="r", key="utm")
    df = pd.read_hdf(datafile, mode="r", key="data_a")

    hourlymax = np.mean(df.quantile(0.95))
    kt = df / hourlymax

    cld_spd_jam, cld_dir_jam, dat_jam = cmv.compute_cmv(kt, pos_utm,
                                                        reference_id=pos_utm.index[0],
                                                        method='jamaly')


    # Known values calculated on 11/3/2023
    assert cld_spd_jam == approx(9.3911, abs=0.01)
    assert cld_dir_jam == approx(3.665955, abs=0.01)
    assert sum(dat_jam.pair_flag == cmv.Flag.GOOD) == 64


def test_optimum_subset_base():
    angles = np.linspace(0, np.pi, 9)
    magnitudes = np.linspace(1, 6, 9)
    vx = magnitudes * np.cos(angles)
    vy = magnitudes * np.sin(angles)
    indices = cmv.optimum_subset(vx, vy, n=4)
    assert np.sort(np.rad2deg(angles[indices])) == approx(np.sort(np.array([0, 45, 90, 135])))


def test_optimum_subset_rotate():
    angles = np.linspace(0, np.pi, 9) + np.pi/4
    magnitudes = np.linspace(1, 6, 9)
    vx = magnitudes * np.cos(angles)
    vy = magnitudes * np.sin(angles)
    indices = cmv.optimum_subset(vx, vy, n=4)
    assert np.sort(np.rad2deg(angles[indices])) == approx(np.sort(np.array([45, 90, 135, 180])))
