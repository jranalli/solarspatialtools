import pytest
from pytest import approx
from numpy import testing as npt
import numpy as np

from solarspatialtools.synthirrad import copula

import pandas as pd

class TestBasics:

    def test_sigmoid_centerpoint(self):
        assert copula._sigmoid(0.5,1,0.5) == approx(0.5)
        assert copula._sigmoid(1, 10, 1) == approx(0.5)

    def test_sigmoid_asymptotes(self):
        assert copula._sigmoid(0, 100, 0.5) == approx(0)
        assert copula._sigmoid(1, 100, 0.5) == approx(1)

    def test_sigmoid_slope_numpy(self):
        # The slope at the center point should be a/4
        a = 10
        c = 0.5
        x = np.linspace(0, 1, 1000)
        y = copula._sigmoid(x, a, c)
        dy_dx = np.gradient(y, x)
        center_index = np.argmin(np.abs(x - c))
        assert dy_dx[center_index] == approx(a / 4, abs=0.001)

    def test_sigmoid_testvals(self):
        assert copula._sigmoid(0.25, 2, 0.5) == approx(0.3775, abs=0.001)
        assert copula._sigmoid(0.75, 10, 0.3) == approx(0.989, abs=0.001)
        assert copula._sigmoid(20, 0.5, 30) == approx(0.0067, abs=0.0001)


class TestGMM:
    _params = {
        'comp': [0.8051, 7.3605, 0.7092],
        'mean': [2.2928, 1.0801, 0.4532],
        'sdevClear': [0.3512, 4.8414, 0.6442],
        'sdevCloud': [0.1997, 5.0919, 0.3863],
    }

    _expect = {
        'in': np.array([-1,-0.5,0,0.5,0.8,0.1]),
        'out': np.array([0.0000, 0.0000, 0.0145, 2.1843, 0.2287, 0.1231])
    }

    def test_gaussian_mixture_distribution_scalar_csi(self):
        gm, _ = copula._gaussianMixtureDistribution(0.1, 0.52, self._params)

        assert gm.means_.shape == (2, 1)

    def test_gaussian_mixture_distribution_vector_csi(self):
        csi = np.linspace(-2, 2, 21)
        gm, _ = copula._gaussianMixtureDistribution(csi, 0.52, self._params)

        assert gm.means_.shape == (2, 1)

    def test_gaussian_mixture_distribution_matlabvals(self):
        gm, pdf_val = copula._gaussianMixtureDistribution(self._expect['in'], 0.52, self._params)
        assert pdf_val == approx(self._expect['out'], abs=0.001)
