import pytest
from pytest import approx

import numpy as np
import pandas as pd

from solartoolbox import field, spatial


@pytest.fixture(params=['coherence', 'global_coherence', 'distance', 'all'])
def mode(request):
    return request.param


@pytest.fixture(params=[0, 1, 2, 3, 4])
def refdat(request, mode, delay_method):
    data = {}
    data['multi'] = {
        "coherence": (
            ('CMB-001-1', (46.35, 1117.45)),
            ('CMB-025-7', (237.82, 531.65)),
            ('CMB-015-3', (367.55, 811.45)),
            ('CMB-003-1', (244.10, 1039.52)),
            ('CMB-008-5', (164.93, 876.94))
        ),
        "global_coherence": (
            ('CMB-001-1', (53.37, 1107.52)),
            ('CMB-025-7', (234.23, 529.77)),
            ('CMB-015-3', (369.15, 809.23)),
            ('CMB-003-1', (245.82, 1054.97)),
            ('CMB-008-5', (164.31, 876.15))
        ),
        "distance": (
            ('CMB-001-1', (-36.97, 1063.08)),
            ('CMB-025-7', (239.94, 540.79)),
            ('CMB-015-3', (368.99, 810.51)),
            ('CMB-003-1', (258.80, 1067.81)),
            ('CMB-008-5', (162.10, 869.84))
        ),
        'all': (
            ('CMB-001-1', (348.09, 653.15)),
            ('CMB-025-7', (291.24, 542.36)),
            ('CMB-015-3', (414.61, 587.06)),
            ('CMB-003-1', (476.56, 652.45)),
            ('CMB-008-5', (359.54, 617.64))
        ),
    }

    data['fit'] = {
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

    datai = data[delay_method][mode]

    return delay_method, mode, datai[request.param]


def test_compute_predicted_position_static(refdat):
    delay_method = refdat[0]
    mode = refdat[1]
    ref = refdat[2][0]
    expect = refdat[2][1]

    datafile = "../docs/sphinx/source/demos/data/sample_plant_2.h5"
    cmv_a = spatial.pol2rect(9.52, 0.62)
    cmv_b = spatial.pol2rect(8.47, 2.17)

    pos_utm = pd.read_hdf(datafile, mode="r", key="utm")

    df_a = pd.read_hdf(datafile, mode="r", key="data_a")
    df_b = pd.read_hdf(datafile, mode="r", key="data_b")

    pos, _ = field.compute_predicted_position(
        [df_a, df_b],  # The dataframes with the two one hour periods
        pos_utm,  # the dataframe specifying the combiner positions
        ref,  # the position within pos_utm to calculate about
        [cmv_a, cmv_b],  # The two individual CMVs for the DFs
        mode=mode,  # Mode for downselecting the comparison points
        ndownsel=8,
        delay_method=delay_method)

    assert pos == approx(expect, abs=1e-2)


@pytest.fixture(params=["fit", "multi"])
def delay_method(request):
    return request.param


@pytest.mark.parametrize("delay", [-5.0, -2.5, -0.5, 0.0, 0.5, 2.5, 5.0])
def test_compute_delays(delay, delay_method):
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

    delays, coh = field.compute_delays(df, ref, navgs=5, coh_limit=0.6, freq_limit=1, method=delay_method)

    assert (delays == approx([0, delay, 2*delay, 3*delay, 4*delay], abs=2e-3))


def test_compute_delays_nan(delay_method):
    delay = 5
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
    y4 = np.nan * np.zeros_like(y3)

    df = pd.DataFrame(np.array([x,y1,y2,y3,y4]).T, index=pd.TimedeltaIndex(t, 's'), columns=['x1','x2','x3','x4','x5'])
    ref = 'x1'

    delays, coh = field.compute_delays(df, ref, navgs=5, coh_limit=0.6, freq_limit=1, method=delay_method)

    assert (delays[0:-1] == approx([0, delay, 2*delay, 3*delay], abs=2e-3))
    assert np.isnan(delays[-1])


@pytest.fixture
def position_data():
    return pd.DataFrame({
        'E': [1, 2, 3, 4, 5],
        'N': [6, 7, 8, 9, 10]
    }, index=['A', 'B', 'C', 'D', 'E'])


@pytest.fixture
def remap_reverse():
    return [('A', 'E'), ('B', 'D'), ('C', 'C'), ('D', 'B'), ('E', 'A')]


@pytest.fixture
def remap_copy():
    return [('A', 'A'), ('B', 'B'), ('C', 'C'), ('D', 'D'), ('E', 'E')]


def test_assign_positions_types(position_data):
    remap_indices, data_out = field.assign_positions(position_data,
                                                     position_data)

    assert isinstance(remap_indices, list)
    assert all(isinstance(item, tuple) for item in remap_indices)
    assert isinstance(data_out, pd.DataFrame)

    assert len(remap_indices) == len(position_data)
    assert set(data_out.index) == set(position_data.index)


def test_assign_positions_duplicate(position_data):
    remap_indices, data_out = field.assign_positions(position_data,
                                                     position_data)

    expected_remap = list(zip(position_data.index,
                              position_data.index))
    assert remap_indices == expected_remap


def test_assign_positions_reverse(position_data):
    # Check whether we can reverse the order of the positions and pick that up
    predicted_pos = position_data.iloc[::-1]
    remap_indices, data_out = field.assign_positions(position_data,
                                                     predicted_pos)

    expected_remap = list(zip(position_data.index,
                              reversed(position_data.index)))
    assert remap_indices == expected_remap


def test_remap_data(position_data, remap_reverse):
    data_out = field.remap_positions(position_data, remap_reverse)

    assert isinstance(data_out, pd.DataFrame)

    for original, remapped in remap_reverse:
        assert data_out.loc[original].equals(position_data.loc[remapped])


def test_cascade_remap_nochange():
    remap1 = [('A', 'B'), ('B', 'C'), ('C', 'A'), ('D', 'D')]
    remap2 = [('A', 'A'), ('B', 'B'), ('C', 'C'), ('D', 'D')]

    result = field.cascade_remap(remap1, remap2)

    assert isinstance(result, list)
    assert all(isinstance(item, tuple) for item in result)

    expected_result = [('A', 'B'), ('B', 'C'), ('C', 'A'), ('D', 'D')]
    assert result == expected_result


def test_cascade_remap_changes():
    remap1 = [('A', 'B'), ('B', 'C'), ('C', 'A'), ('D', 'D')]
    remap2 = [('A', 'B'), ('B', 'C'), ('C', 'A'), ('D', 'D')]

    result = field.cascade_remap(remap1, remap2)

    assert isinstance(result, list)
    assert all(isinstance(item, tuple) for item in result)

    expected_result = [('A', 'C'), ('B', 'A'), ('C', 'B'), ('D', 'D')]
    assert result == expected_result
