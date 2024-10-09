import pytest
from pytest import approx
from numpy import testing as npt
import numpy as np

from solarspatialtools.synthirrad import cloudfield

import pandas as pd


class TestFieldGeneration:
    # TODO add tests for size 1 and size 0 to guarantee that they produce correct output
    def test_random_at_scale_identity(self):
        base, interp = cloudfield._random_at_scale((10,10), (10, 10), False)
        assert base.shape == (10, 10)
        assert interp.shape == (10, 10)
        npt.assert_allclose(base, interp)

    def test_random_at_scale_shape(self):
        rand_size = (10, 10)
        final_size = (20, 20)
        base, interp = cloudfield._random_at_scale(rand_size, final_size)
        assert base.shape == rand_size
        assert interp.shape == final_size

    def test_random_at_scale_values(self):
        rand_size = (10, 10)
        final_size = (20, 20)
        base, interp = cloudfield._random_at_scale(rand_size, final_size)
        assert base.min() >= 0 and base.max() <= 1
        assert interp.min() >= 0 and interp.max() <= 1

    def test_random_at_scale_interpolation(self):
        rand_size = (10, 10)
        final_size = (20, 20)
        base, interp = cloudfield._random_at_scale(rand_size, final_size)
        # Check that the interpolated values are within the range of the original values
        assert interp.min() >= base.min() and interp.max() <= base.max()

    def test_stack_random_field_with_weights(self):
        size = (100, 100)
        scales = np.array([1, 2, 3])
        weights = np.array([0.2, 0.3, 0.5])
        field = cloudfield._stack_random_field(weights, scales, size)
        assert field.shape == size, "Field size should match the input size"
        assert np.min(field) >= 0 and np.max(field) <= 1, "Field values should be normalized between 0 and 1"

    def test_stack_random_field_weights_mismatch(self):
        size = (100, 100)
        scales = np.array([1, 2, 3])
        weights = np.array([0.2, 0.3])
        with pytest.raises(ValueError):
            cloudfield._stack_random_field(weights, scales, size)

    def test_stack_random_field_plot(self):
        size = (100, 100)
        scales = np.array([1, 2, 3])
        weights = np.array([0.2, 0.3, 0.5])
        field = cloudfield._stack_random_field(weights, scales, size, normalize=True)
        assert field.shape == size, "Field size should match the input size"
        assert np.min(field) == 0 and np.max(field) == 1, "Field values should be normalized between 0 and 1"

class TestFieldProcessing:
    def test_calc_clear_mask_basic(self):
        field = np.random.rand(100, 100)
        clear_frac = 0.5
        mask = cloudfield._calc_clear_mask(field, clear_frac)
        assert mask.shape == field.shape, "Mask size should match the input field size"
        assert np.isclose(np.mean(mask), clear_frac, rtol=1e-3), "Clear fraction should be close to the specified value"

    def test_calc_clear_mask_all_clear(self):
        field = np.random.rand(100, 100)
        clear_frac = 1.0
        mask = cloudfield._calc_clear_mask(field, clear_frac)
        assert np.all(mask == 1), "All values should be clear (1) when clear_frac is 1.0"

    def test_calc_clear_mask_all_cloudy(self):
        field = np.random.rand(100, 100)
        clear_frac = 0.0
        mask = cloudfield._calc_clear_mask(field, clear_frac)
        assert np.all(mask == 0), "All values should be cloudy (0) when clear_frac is 0.0"

    def test_calc_clear_mask_invalid_clear_frac(self):
        field = np.random.rand(100, 100)
        with pytest.raises(ValueError):
            cloudfield._calc_clear_mask(field, 1.1)
        with pytest.raises(ValueError):
            cloudfield._calc_clear_mask(field, -0.1)

    def test_find_edges_basic(self):
        base_mask = np.random.randint(0, 2, (100, 100))
        size = 3
        edges, smoothed_binary = cloudfield._find_edges(base_mask, size)
        assert edges.shape == base_mask.shape, "Edges size should match the input mask size"
        assert smoothed_binary.shape == base_mask.shape, "Smoothed binary size should match the input mask size"

    def test_find_edges_all_zeros(self):
        base_mask = np.zeros((100, 100))
        size = 3
        edges, smoothed_binary = cloudfield._find_edges(base_mask, size)
        assert np.all(edges == 0), "Edges should be all zeros for an all-zero input mask"
        assert np.all(smoothed_binary == 0), "Smoothed binary should be all zeros for an all-zero input mask"

    def test_find_edges_all_ones(self):
        base_mask = np.ones((100, 100))
        size = 3
        edges, smoothed_binary = cloudfield._find_edges(base_mask, size)
        assert np.all(edges == 0), "Edges should be all zeros for an all-one input mask"
        assert np.all(smoothed_binary == 0), "Smoothed binary should be all zeros for an all-one input mask"

    def test_find_edges_half(self):
        # make a mask half 1 half zero split vertically
        base_mask = np.zeros((100, 100))
        base_mask[:50, :] = 1
        size = 3
        edges, smoothed_binary = cloudfield._find_edges(base_mask, size)
        assert np.sum(edges > 0) == 200
        assert np.sum(smoothed_binary) == 400

    def test_find_edges_square(self):
        # make a square mask
        base_mask = np.zeros((100, 100))
        base_mask[25:70, 25:75] = 1
        size = 3
        edges, smoothed_binary = cloudfield._find_edges(base_mask, size)
        assert np.sum(edges > 0) == 380
        assert np.sum(smoothed_binary) == 760

    @pytest.fixture
    def sample_data(self):
        field = np.random.rand(100, 100)
        clear_mask = np.zeros_like(field)
        edge_mask = np.zeros_like(field)
        edge_mask[25:75, 25:75] = 1
        clear_mask[30:70, 30:70] = 1
        edge_mask[clear_mask>0] = 0
        return field, clear_mask, edge_mask

    def test_scale_field_lave_basic(self, sample_data):
        field, clear_mask, edge_mask = sample_data
        ktmean = 0.5
        ktmax = 1.08
        kt1pct = 0.2
        result = cloudfield._scale_field_lave(field, clear_mask, edge_mask, ktmean, ktmax, kt1pct)
        assert result.shape == field.shape
        assert np.isclose(np.mean(result), ktmean, atol=0.01)


class TestWeights:
    """Tests generated with AI assistance"""

    def test_calc_vs_weights_basic(self):
        scales = np.array([1, 2, 3])
        vs = 0.5
        weights = cloudfield._calc_vs_weights(scales, vs)
        assert np.isclose(np.sum(weights), 1), "Weights should sum to 1"
        assert len(weights) == len(scales), "Weights length should match scales length"

    def test_calc_vs_weights_zero_vs(self):
        scales = np.array([1, 2, 3])
        vs = 0
        weights = cloudfield._calc_vs_weights(scales, vs)
        assert np.isclose(np.sum(weights), 1), "Weights should sum to 1"
        assert len(weights) == len(scales), "Weights length should match scales length"

    def test_calc_vs_weights_high_vs(self):
        scales = np.array([1, 2, 3])
        vs = 10
        weights = cloudfield._calc_vs_weights(scales, vs)
        assert np.isclose(np.sum(weights), 1), "Weights should sum to 1"
        assert len(weights) == len(scales), "Weights length should match scales length"

    def test_calc_vs_weights_large_scales(self):
        scales = np.array([1, 10, 100])
        vs = 1
        weights = cloudfield._calc_vs_weights(scales, vs)
        assert np.isclose(np.sum(weights), 1), "Weights should sum to 1"
        assert len(weights) == len(scales), "Weights length should match scales length"

    def test_calc_vs_weights_negative_vs(self):
        scales = np.array([1, 2, 3])
        vs = -0.5
        with pytest.raises(ValueError):
            cloudfield._calc_vs_weights(scales, vs)

    def test_calc_wavelet_weights_basic(self):
        waves = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        weights = cloudfield._calc_wavelet_weights(waves)
        assert np.isclose(np.sum(weights), 1), "Weights should sum to 1"
        assert len(weights) == waves.shape[0], "Weights length should match number of wavelet scales"

    def test_calc_wavelet_weights_zeros(self):
        waves = np.zeros((3, 3))
        weights = cloudfield._calc_wavelet_weights(waves)
        assert np.all(np.isnan(weights)), "Weights will be nan for zero input"

    def test_calc_wavelet_weights_large_values(self):
        waves = np.array([[1e10, 2e10, 3e10], [4e10, 5e10, 6e10], [7e10, 8e10, 9e10]])
        weights = cloudfield._calc_wavelet_weights(waves)
        assert np.isclose(np.sum(weights), 1), "Weights should sum to 1"
        assert len(weights) == waves.shape[0], "Weights length should match number of wavelet scales"

    def test_calc_wavelet_weights_negative_values(self):
        waves = np.array([[-1, -2, -3], [-4, -5, -6], [-7, -8, -9]])
        weights = cloudfield._calc_wavelet_weights(waves)
        assert np.isclose(np.sum(weights), 1), "Weights should sum to 1"
        assert len(weights) == waves.shape[0], "Weights length should match number of wavelet scales"

    def test_calc_wavelet_weights_mixed_values(self):
        waves = np.array([[1, -2, 3], [-4, 5, -6], [7, -8, 9]])
        weights = cloudfield._calc_wavelet_weights(waves)
        assert np.isclose(np.sum(weights), 1), "Weights should sum to 1"
        assert len(weights) == waves.shape[0], "Weights length should match number of wavelet scales"

    def test_get_timeseries_props_basic(self):
        kt_ts = pd.Series(np.random.rand(1000))
        ktmean, kt_1_pct, ktmax, frac_clear, vs, weights, scales = cloudfield.get_timeseries_stats(kt_ts, plot=False)
        assert isinstance(ktmean, float), "ktmean should be a float"
        assert isinstance(kt_1_pct, float), "kt_1_pct should be a float"
        assert isinstance(ktmax, float), "ktmax should be a float"
        assert isinstance(frac_clear, float), "frac_clear should be a float"
        assert isinstance(vs, float), "vs should be a float"
        assert isinstance(weights, np.ndarray), "weights should be a numpy array"
        assert isinstance(scales, list), "scales should be a list"

    def test_get_timeseries_props_all_clear(self):
        kt_ts = pd.Series(np.ones(1000))
        ktmean, kt_1_pct, ktmax, frac_clear, vs, weights, scales = cloudfield.get_timeseries_stats(kt_ts, plot=False)
        assert frac_clear == 1.0, "frac_clear should be 1.0 for all-clear time series"

    def test_get_timeseries_props_all_cloudy(self):
        kt_ts = pd.Series(np.zeros(1000))
        ktmean, kt_1_pct, ktmax, frac_clear, vs, weights, scales = cloudfield.get_timeseries_stats(kt_ts, plot=False)
        assert frac_clear == 0.0, "frac_clear should be 0.0 for all-cloudy time series"

