from pytest import approx
import numpy as np

from solarspatialtools.synthirrad import copula


class TestBasics:

    def test_sigmoid_centerpoint(self):
        assert copula._sigmoid(0.5, 1, 0.5) == approx(0.5)
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

    def test_exponential_decay_parameter(self):
        assert copula._exponential_decay_parameter(0.5, 10) == approx(2.5)
        assert copula._exponential_decay_parameter(0.5, 1) == approx(0.25)
        assert copula._exponential_decay_parameter(0.5, 0) == approx(10**-5)
        assert copula._exponential_decay_parameter(0.01, 10) == approx(0.0990, abs=0.0001)
        assert copula._exponential_decay_parameter(0.99, 10) == approx(0.0990, abs=0.0001)
        assert copula._exponential_decay_parameter(0.0001, 10) == approx(0.0009999, abs=0.000001)
        assert copula._exponential_decay_parameter(0.9999, 10) == approx(0.0009999, abs=0.000001)


class TestGMDistribution:
    _params = {
        'mean': [0.25, 0.75],
        'sigma': [0.025, 0.025],
        'p': [8, 4]
    }
    _expect = {'x': [0.02, 0.49, 0.92],
               'y_cdf': [0.0516, 0.6441, 0.9553]}

    def test_gmdistribution_matlab(self):
        x = np.arange(-2, 2, 0.01)
        _, _, y = copula._gmdistribution(x, self._params['mean'], self._params['sigma'], self._params['p'])
        # extract the y values that correspond to the 3 values in _expect['x']
        y_vals = [y[np.argmin(np.abs(x - target))] for target in self._expect['x']]

        assert y_vals == approx(self._expect['y_cdf'], abs=0.001)


class TestSolarGMM:
    _params = {
        'comp': [0.8051, 7.3605, 0.7092],
        'mean': [2.2928, 1.0801, 0.4532],
        'sdevClear': [0.3512, 4.8414, 0.6442],
        'sdevCloud': [0.1997, 5.0919, 0.3863],
    }

    _expect = {
        'in': np.array([-1, -0.5, 0, 0.5, 0.8, 0.1]),
        'out': np.array([0.0000, 0.0000, 0.0145, 2.1843, 0.2287, 0.1231])
    }

    def test_solar_gmm_scalar_csi(self):
        gm, _, _ = copula._solar_gmm(0.1, 0.52, self._params)

        assert gm.means_.shape == (2, 1)

    def test_solar_gmm_vector_csi(self):
        csi = np.linspace(-2, 2, 21)
        gm, _, _ = copula._solar_gmm(csi, 0.52, self._params)

        assert gm.means_.shape == (2, 1)

    def test_solar_gmm_matlabvals(self):
        gm, pdf_val, _ = copula._solar_gmm(self._expect['in'], 0.52, self._params)
        assert pdf_val == approx(self._expect['out'], abs=0.001)


class TestInverseSample:

    def test_inverse_sample_linear_interp(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        cdf = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        r = np.array([0.125, 0.5, 0.875])

        out = copula._inverse_sample(x, cdf, r)
        assert out == approx(np.array([-1.5, 0.0, 1.5]))

    def test_inverse_sample_complex_shape(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        cdf = np.array([0.0, 0.1, 0.5, 0.9, 1.0])
        r = np.array([0.05, 0.5, 0.95])
        out = copula._inverse_sample(x, cdf, r)
        assert out == approx(np.array([-1.5, 0.0, 1.5]))

    def test_inverse_sample_uses_first_duplicate_index(self):
        x = np.array([-2.0, -1.0, 0.0, 2.0])
        cdf = np.array([0.0, 0.5, 0.5, 1.0])

        out = copula._inverse_sample(x, cdf, 0.5)
        assert out == approx(-1.0)

    def test_inverse_sample_preserves_shape(self):
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        cdf = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        r = np.array([[0.1, 0.2], [0.8, 0.9]])

        out = copula._inverse_sample(x, cdf, r)
        assert out.shape == r.shape
