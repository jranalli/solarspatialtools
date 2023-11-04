import pytest
from pytest import approx

import numpy as np
import pandas as pd

from solartoolbox import field, spatial

@pytest.fixture(params=[0,1,2,3,4])
def refdat(request):
    data = (
        ('CMB-001-1', (45.79, 1116.99)),
        ('CMB-025-7', (235.84, 531.57)),
        ('CMB-015-3', (368.11, 812.28)),
        ('CMB-003-1', (242.76, 1041.82)),
        ('CMB-008-5', (165.38, 876.23))
    )

    return data[request.param]


@pytest.fixture(params=[0,1,2,3,4])
def refdat_None(request):
    data = (
        ('CMB-001-1', (365.51, 609.80)),
        ('CMB-025-7', (294.15, 506.62)),
        ('CMB-015-3', (411.36, 585.42)),
        ('CMB-003-1', (474.64, 653.06)),
        ('CMB-008-5', (356.92, 612.43))
    )

    return data[request.param]

def test_compute_predicted_position_static(refdat):
    ref = refdat[0]
    expect = refdat[1]

    datafile = "../demos/data/sample_plant_2.h5"
    cmv_a = spatial.pol2rect(9.52, 0.62)
    cmv_b = spatial.pol2rect(8.47, 2.17)

    pos_utm = pd.read_hdf(datafile, mode="r", key="latlon")

    df_a = pd.read_hdf(datafile, mode="r", key="data_a")
    df_b = pd.read_hdf(datafile, mode="r", key="data_b")

    com, pos, _ = field.compute_predicted_position(
        [df_a, df_b],  # The dataframes with the two one hour periods
        pos_utm,  # the dataframe specifying the combiner positions
        ref,  # the position within pos_utm to calculate about
        [cmv_a, cmv_b],  # The two individual CMVs for the DFs
        mode='preavg',  # Mode for downselecting the comparison points
        ndownsel=8)

    assert com == approx(expect, abs=1e-2)


def test_compute_predicted_position_static_none(refdat_None):
    ref = refdat_None[0]
    expect = refdat_None[1]

    datafile = "../demos/data/sample_plant_2.h5"
    cmv_a = spatial.pol2rect(9.52, 0.62)
    cmv_b = spatial.pol2rect(8.47, 2.17)

    pos_utm = pd.read_hdf(datafile, mode="r", key="latlon")

    df_a = pd.read_hdf(datafile, mode="r", key="data_a")
    df_b = pd.read_hdf(datafile, mode="r", key="data_b")

    com, pos, _ = field.compute_predicted_position(
        [df_a, df_b],  # The dataframes with the two one hour periods
        pos_utm,  # the dataframe specifying the combiner positions
        ref,  # the position within pos_utm to calculate about
        [cmv_a, cmv_b],  # The two individual CMVs for the DFs
        mode=None,  # Mode for downselecting the comparison points
        ndownsel=8)

    assert com == approx(expect, abs=1e-2)



@pytest.fixture(params=range(9))
def data(request):
    #  [(Axis1 , Axis2),      (Dist),   (Point)]
    # [((Ax, Ay), (Bx, By)), (Dx, Dy), (Px, Py)]
    srcdata = [
        [((1, 0), (0, 1)), (1, 1), (1, 1)],
        [((1, 1), (1, -1)), (1, 1), (np.sqrt(2), 0)],  # rotated -45 degrees
        [((-1, -1), (-1, 1)), (1, 1), (-np.sqrt(2), 0)],  # rotated +135 degrees
        [((np.sqrt(3), 1), (1, -np.sqrt(3))), (1, 1), (np.sqrt(2)*np.cos(15*np.pi/180), -np.sqrt(2)*np.sin(15*np.pi/180))],  # initial case rotated -60 degrees
        [((1, 0), (0, 1)), (4, 1), (4, 1)],  # Different lengths
        [((1, 0), (0, 1)), (1, 4), (1, 4)],
        [((1, 0), (0, 1)), (3, 4), (3, 4)],
        [((1, 1), (1, 0)), (1, np.sqrt(2)), (np.sqrt(2), 0)],  # 45 degree separation
        [((np.sqrt(3), 1), (np.sqrt(3), -1)), (2, 2), (4/np.sqrt(3), 0)],  # 60 degree separation evenly about x axis
    ]
    return srcdata[request.param]


@pytest.fixture(params=[1, 10])
def n(request):
    return request.param


def test_compute_intersection(data, n):

    axes = data[0]

    distances = (np.repeat(np.array([data[1][0]]), n),
                 np.repeat(np.array([data[1][1]]), n))

    expected_output = np.repeat(np.array([data[2]]), n, axis=0)

    pos = field.compute_intersection(axes, distances)

    assert pos == approx(expected_output)
