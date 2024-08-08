import pytest
from pytest import approx
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

from solarspatialtools import clearsky_index


def test_clearsky_index():
    ghi = np.array([-1., 0., 1., 500., 1000., np.nan])
    ghi_measured, ghi_modeled = np.meshgrid(ghi, ghi)
    # default max_clearsky_index
    with np.errstate(invalid='ignore', divide='ignore'):
        out = clearsky_index(ghi_measured, ghi_modeled)
    expected = np.array(
        [[1.    , 0.    , 0.    , 0.    , 0.    , np.nan],
         [0.    , 0.    , 0.    , 0.    , 0.    , np.nan],
         [0.    , 0.    , 1.    , 2.    , 2.    , np.nan],
         [0.    , 0.    , 0.002 , 1.    , 2.    , np.nan],
         [0.    , 0.    , 0.001 , 0.5   , 1.    , np.nan],
         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
    assert_allclose(out, expected, atol=0.001)
    # specify max_clearsky_index
    with np.errstate(invalid='ignore', divide='ignore'):
        out = clearsky_index(ghi_measured, ghi_modeled, max_clearsky_index=1.5)
    expected = np.array(
        [[1.    , 0.    , 0.    , 0.    , 0.    , np.nan],
         [0.    , 0.    , 0.    , 0.    , 0.    , np.nan],
         [0.    , 0.    , 1.    , 1.5   , 1.5   , np.nan],
         [0.    , 0.    , 0.002 , 1.    , 1.5   , np.nan],
         [0.    , 0.    , 0.001 , 0.5   , 1.    , np.nan],
         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
    assert_allclose(out, expected, atol=0.001)

    # GHI series & CS series
    times = pd.date_range(start='20180601', periods=2, freq='12H')
    ghi_measured = pd.Series([100,  500], index=times)
    ghi_modeled = pd.Series([500, 1000], index=times)
    out = clearsky_index(ghi_measured, ghi_modeled)
    expected = pd.Series([0.2, 0.5], index=times)
    assert isinstance(out, pd.Series)
    assert (out.index == times).all()
    assert out.values == approx(expected.values)

    # GHI 2D frame & CS 1D series
    times = pd.date_range(start='20180601', periods=3, freq='12H')
    ghi_measured = pd.DataFrame([[100,  100], [200, 200], [500, 500]], index=times)
    ghi_modeled = pd.Series([500, 800, 1000], index=times)
    out = clearsky_index(ghi_measured, ghi_modeled)
    expected = pd.DataFrame([[0.2, 0.2], [0.25, 0.25], [0.5, 0.5]], index=times)
    assert isinstance(out, pd.DataFrame)
    assert (out.index == times).all()
    assert out.values == approx(expected.values)
