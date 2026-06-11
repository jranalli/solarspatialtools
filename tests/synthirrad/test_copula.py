import pandas as pd
from pytest import approx
import numpy as np

from scipy.stats import spearmanr, kstest, uniform

from solarspatialtools.synthirrad import copula
from solarspatialtools.synthirrad.copula import downscale


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
               'y_pdf': [7.007469, 8.989724, 5.66465],
               'y_cdf': [0.0516, 0.6441, 0.9553]}

    def test_gmdistribution_matlab(self):
        x = np.arange(-2, 2, 0.01)
        y_p, y_c = copula._gmdistribution(x, self._params['mean'], self._params['sigma'], self._params['p'])
        # extract the y values that correspond to the 3 values in _expect['x']
        y_vals_c = [y_c[np.argmin(np.abs(x - target))] for target in self._expect['x']]
        y_vals_p = [y_p[np.argmin(np.abs(x - target))] for target in self._expect['x']]

        assert y_vals_c == approx(self._expect['y_cdf'], abs=0.001)
        assert y_vals_p == approx(self._expect['y_pdf'], abs=0.001)

    def test_gmdistribution_single(self):
        # Can't test CDF
        x = np.arange(-2, 2, 0.01)
        for x, y_e in zip(self._expect['x'], self._expect['y_pdf']):
            y, _ = copula._gmdistribution(x, self._params['mean'], self._params['sigma'], self._params['p'])
            # Can't test CDF because it scales to 1.
            assert y == approx(y_e, abs=0.001)


class TestSolarGMM:
    _params = {
        'comp': [0.8051, 7.3605, 0.7092],
        'mean': [2.2928, 1.0801, 0.4532],
        'sdevClear': [0.3512, 4.8414, 0.6442],
        'sdevCloud': [0.1997, 5.0919, 0.3863],
        'corr_quadr': 0.0043
    }

    _expect = {
        'in': np.array([-1, -0.5, 0, 0.5, 0.8, 0.1]),
        'out_pdf': np.array([0.0000, 0.0000, 0.0145, 2.1843, 0.2287, 0.1231]),
        'out_cdf': np.array([0.0000, 0.0000, 0.00568, 0.8621, 0.95175, 1.0]),
        'p': 0.00107328
    }

    def test_solar_gmm_matlabvals(self):
        pdf_val, cdf_val, p = copula._solar_gmm(self._expect['in'], 0.52, self._params)
        assert pdf_val == approx(self._expect['out_pdf'], abs=0.001)
        assert cdf_val == approx(self._expect['out_cdf'], abs=0.001)
        assert p == approx(self._expect['p'])


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


class TestDownscale:
    _params = {
        'comp': [0.8051, 7.3605, 0.7092],
        'mean': [2.2928, 1.0801, 0.4532],
        'sdevClear': [0.3512, 4.8414, 0.6442],
        'sdevCloud': [0.1997, 5.0919, 0.3863],
        'corr_quadr': 0.0043
    }

    cs = 1
    cd = 0
    hcsi = 0.52
    Epos = np.array([ 0, 450])
    Npos = np.array([ 0,   0])
    times = pd.date_range(start='2024-01-01 00:00:00',
                          end='2024-01-01 00:59:59', freq='15s')
    seed = 42


    _expect = {
        'ind': np.array([50,100,150,200]),
        'out_csi': np.array([[0.4995, 0.8711, 0.2953, 0.4153],
                            [0.5263, 0.6971, 0.4411 , 0.3162]])
    }

    def test_downscale_stability(self):
        c = downscale(self.times, self.Epos, self.Npos, self.cd, self.cs,
                      self.hcsi, self._params, self.seed, True, True)
        assert c.shape == (len(self.times), len(self.Epos))
        assert c[self._expect['ind']].T == approx(self._expect['out_csi'], abs=0.002)

    def test_downscale_shift(self):
        c = downscale(self.times, self.Epos, self.Npos, self.cd, self.cs,
                      self.hcsi, self._params, self.seed, True, True)
        assert c[:-30,0] == approx(c[30:,1])


class TestCopularndGaussian:
    """Tests for _copularnd_gaussian function."""

    def test_copularnd_gaussian_output_shape(self):
        """Test that output shape matches (n, d) where d is cov_matrix dimension."""
        n = 100
        d = 5
        cov_matrix = np.eye(d)
        
        U = copula._copularnd_gaussian(n, cov_matrix, random_state=42)
        
        assert U.shape == (n, d)

    def test_copularnd_gaussian_uniform_bounds(self):
        """Test that output values are within [0, 1] (uniform distribution)."""
        n = 1000
        d = 3
        cov_matrix = np.eye(d)
        
        U = copula._copularnd_gaussian(n, cov_matrix, random_state=42)
        
        assert np.all(U >= 0)
        assert np.all(U <= 1)

    def test_copularnd_gaussian_reproducibility(self):
        """Test that the same seed produces identical results."""
        n = 100
        d = 4
        cov_matrix = np.eye(d)
        seed = 42
        
        U1 = copula._copularnd_gaussian(n, cov_matrix, random_state=seed)
        U2 = copula._copularnd_gaussian(n, cov_matrix, random_state=seed)
        
        assert np.allclose(U1, U2)

    def test_copularnd_gaussian_different_seeds_differ(self):
        """Test that different seeds produce different results."""
        n = 100
        d = 3
        cov_matrix = np.eye(d)
        
        U1 = copula._copularnd_gaussian(n, cov_matrix, random_state=42)
        U2 = copula._copularnd_gaussian(n, cov_matrix, random_state=123)
        
        assert not np.allclose(U1, U2)

    def test_copularnd_gaussian_correlation_preserved(self):
        """Test that correlation structure is preserved in copula samples."""
        n = 5000
        # Create a covariance matrix with strong correlation
        rho = 0.8
        cov_matrix = np.array([[1.0, rho],
                               [rho, 1.0]])
        
        U = copula._copularnd_gaussian(n, cov_matrix, random_state=42)
        
        # Compute Spearman rank correlation (appropriate for copulas)
        corr, _ = spearmanr(U[:, 0], U[:, 1])
        
        # Should be positive correlation similar to rho
        assert corr > 0.7  # Relaxed tolerance due to sampling

    def test_copularnd_gaussian_independence_preserved(self):
        """Test that independent variables remain approximately independent."""
        n = 5000
        # Create an identity covariance matrix (independent)
        cov_matrix = np.eye(3)
        
        U = copula._copularnd_gaussian(n, cov_matrix, random_state=42)

        corr_12, _ = spearmanr(U[:, 0], U[:, 1])
        corr_13, _ = spearmanr(U[:, 0], U[:, 2])
        corr_23, _ = spearmanr(U[:, 1], U[:, 2])
        
        # All correlations should be close to zero
        assert abs(corr_12) < 0.1
        assert abs(corr_13) < 0.1
        assert abs(corr_23) < 0.1

    def test_copularnd_gaussian_approximately_uniform(self):
        """Test that marginals are approximately uniform (via KS test)."""
        n = 5000
        d = 2
        cov_matrix = np.eye(d)
        
        U = copula._copularnd_gaussian(n, cov_matrix, random_state=42)

        # Test each marginal
        for i in range(d):
            stat, pval = kstest(U[:, i], uniform(0, 1).cdf)
            # p-value should be > 0.05 for uniform distribution
            assert pval > 0.05

    def test_copularnd_gaussian_identity_covariance(self):
        """Test with identity covariance (special case)."""
        n = 100
        d = 2
        cov_matrix = np.eye(d)
        
        U = copula._copularnd_gaussian(n, cov_matrix, random_state=42)
        
        assert U.shape == (n, d)
        assert np.all(U >= 0) and np.all(U <= 1)

    def test_copularnd_gaussian_large_covariance(self):
        """Test with a larger covariance matrix."""
        n = 100
        d = 10
        cov_matrix = np.eye(d)
        
        U = copula._copularnd_gaussian(n, cov_matrix, random_state=42)
        
        assert U.shape == (n, d)
        assert np.all(U >= 0) and np.all(U <= 1)

    def test_copularnd_gaussian_none_seed(self):
        """Test that None seed works (uses random state)."""
        n = 100
        d = 3
        cov_matrix = np.eye(d)
        
        U = copula._copularnd_gaussian(n, cov_matrix, random_state=None)
        
        assert U.shape == (n, d)
        assert np.all(U >= 0) and np.all(U <= 1)


class TestSpaceTimeCopula:
    """Tests for _space_time_copula function."""

    @staticmethod
    def _create_simple_cdf():
        """Create a simple CDF for testing (uniform over [0, 1])."""
        cdf_x = np.linspace(0, 1, 100)
        cdf = np.linspace(0, 1, 100)
        return cdf_x, cdf

    def test_space_time_copula_output_shape(self):
        """Test that output shape matches input spacetime_dist shape."""
        n = 50
        d = 6  # 2 sites, 3 times
        spacetime_dist = np.eye(d)  # Simple identity distance matrix
        cdf_x, cdf = self._create_simple_cdf()
        p = 0.5
        
        gen_csi = copula._space_time_copula(n, spacetime_dist, cdf_x, cdf, p, seed=42)
        
        assert gen_csi.shape == (n, d)

    def test_space_time_copula_values_in_range(self):
        """Test that generated CSI values are within the CDF range."""
        n = 100
        d = 4
        spacetime_dist = np.eye(d)
        cdf_x, cdf = self._create_simple_cdf()
        cdf = np.linspace(0, 1, 100)
        p = 0.5
        
        gen_csi = copula._space_time_copula(n, spacetime_dist, cdf_x, cdf, p, seed=42)
        
        assert np.all(gen_csi >= cdf_x.min())
        assert np.all(gen_csi <= cdf_x.max())

    def test_space_time_copula_reproducibility(self):
        """Test that same seed produces identical results."""
        n = 50
        d = 4
        spacetime_dist = np.eye(d)
        cdf_x, cdf = self._create_simple_cdf()
        p = 0.5
        seed = 42
        
        gen_csi1 = copula._space_time_copula(n, spacetime_dist, cdf_x, cdf, p, seed=seed)
        gen_csi2 = copula._space_time_copula(n, spacetime_dist, cdf_x, cdf, p, seed=seed)
        
        assert np.allclose(gen_csi1, gen_csi2)

    def test_space_time_copula_different_seeds_differ(self):
        """Test that different seeds produce different results."""
        n = 50
        d = 4
        spacetime_dist = np.eye(d)
        cdf_x, cdf = self._create_simple_cdf()
        p = 0.5
        
        gen_csi1 = copula._space_time_copula(n, spacetime_dist, cdf_x, cdf, p, seed=42)
        gen_csi2 = copula._space_time_copula(n, spacetime_dist, cdf_x, cdf, p, seed=123)
        
        assert not np.allclose(gen_csi1, gen_csi2)

    def test_space_time_copula_correlation_decay(self):
        """Test that correlation decays with distance parameter p."""
        n = 1000
        # Create a simple 2x2 block with partial correlation
        spacetime_dist = np.array([[0.0, 1.0],
                                   [1.0, 0.0]])
        cdf_x, cdf = self._create_simple_cdf()
        
        # Small p: less decay, higher correlation
        gen_csi_small_p = copula._space_time_copula(n, spacetime_dist, cdf_x, cdf, p=0.1, seed=42)
        corr_small_p, _ = spearmanr(gen_csi_small_p[:, 0], gen_csi_small_p[:, 1])
        
        # Large p: more decay, lower correlation
        gen_csi_large_p = copula._space_time_copula(n, spacetime_dist, cdf_x, cdf, p=2.0, seed=42)
        corr_large_p, _ = spearmanr(gen_csi_large_p[:, 0], gen_csi_large_p[:, 1])
        
        # Larger p should result in lower correlation
        assert abs(corr_small_p) > abs(corr_large_p)
