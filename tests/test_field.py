import pytest
from pytest import approx

import numpy as np
import pandas as pd

from solartoolbox import field, spatial


@pytest.fixture(params=['coherence', 'global_coherence', 'distance', 'all'])
def mode(request):
    return request.param


@pytest.fixture(params=[0, 1, 2, 3, 4])
def refdat(request, mode):
    data = {
        "coherence": (
            ('CMB-001-1', (45.79, 1116.99)),
            ('CMB-025-7', (235.84, 531.57)),
            ('CMB-015-3', (368.11, 812.28)),
            ('CMB-003-1', (242.76, 1041.82)),
            ('CMB-008-5', (165.38, 876.23))
        ),
        "global_coherence": (
            ('CMB-001-1', (51.69, 1106.64)),
            ('CMB-025-7', (232.39, 528.94)),
            ('CMB-015-3', (369.86, 809.84)),
            ('CMB-003-1', (243.85, 1055.34)),
            ('CMB-008-5', (162.35, 874.83))
        ),
        "distance": (
            ('CMB-001-1', (-38.24, 1064.02)),
            ('CMB-025-7', (239.24, 540.23)),
            ('CMB-015-3', (371.59, 812.76)),
            ('CMB-003-1', (256.50, 1067.87)),
            ('CMB-008-5', (161.47, 868.25))
        ),
        'all': (
            ('CMB-001-1', (364.61, 611.05)),
            ('CMB-025-7', (293.98, 506.67)),
            ('CMB-015-3', (411.25, 586.00)),
            ('CMB-003-1', (474.20, 654.19)),
            ('CMB-008-5', (356.40, 613.11))
        ),
    }

    datai = data[mode]

    return mode, datai[request.param]


def test_compute_predicted_position_static(refdat):
    mode = refdat[0]
    ref = refdat[1][0]
    expect = refdat[1][1]

    datafile = "../demos/data/sample_plant_2.h5"
    cmv_a = spatial.pol2rect(9.52, 0.62)
    cmv_b = spatial.pol2rect(8.47, 2.17)

    pos_utm = pd.read_hdf(datafile, mode="r", key="latlon")

    df_a = pd.read_hdf(datafile, mode="r", key="data_a")
    df_b = pd.read_hdf(datafile, mode="r", key="data_b")

    pos, _ = field.compute_predicted_position(
        [df_a, df_b],  # The dataframes with the two one hour periods
        pos_utm,  # the dataframe specifying the combiner positions
        ref,  # the position within pos_utm to calculate about
        [cmv_a, cmv_b],  # The two individual CMVs for the DFs
        mode=mode,  # Mode for downselecting the comparison points
        ndownsel=8)

    assert pos == approx(expect, abs=1e-2)


@pytest.mark.parametrize("delay", [-5.0, -2.5, -0.5, 0.0, 0.5, 2.5, 5.0])
def test_compute_delays(delay):
    np.random.seed(2023)
    # Create a simple sinusoidal signal with noise
    fs = 500  # sample rate
    T = 1000.0    # seconds
    t = np.linspace(0, T, int(T*fs), endpoint=False)  # time variable

    x = 0.5 * np.sin(2 * np.pi * 2 * t) + np.random.random(len(t))

    # duplicate and shift the signal
    y1 = np.roll(x, int(delay*fs))
    y2 = np.roll(x, int(2*delay * fs))
    y3 = np.roll(x, int(3*delay * fs))
    y4 = np.roll(x, int(4*delay * fs))

    df = pd.DataFrame(np.array([x,y1,y2,y3,y4]).T, index=pd.TimedeltaIndex(t, 's'), columns=['x1','x2','x3','x4','x5'])
    ref = 'x1'

    delays, coh = field.compute_delays(df, ref, navgs=5, coh_limit=0.6, freq_limit=1)

    assert (delays == approx([0, delay, 2*delay, 3*delay, 4*delay], abs=1e-3))

