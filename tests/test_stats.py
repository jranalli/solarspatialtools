import pytest
from pytest import approx, raises

import numpy as np
import numpy.testing as npt
import pandas as pd

import solartoolbox.signalproc
from solartoolbox import stats


class TestErrorFunctions:
    """
    rmse - root mean square error
    mse - mean square error
    se - squared error
    mae - mean absolute error
    ae - absolute error
    mbe - mean bias error
    be - bias error
    """

    @pytest.fixture
    def datax(self):
        datax = np.arange(0, 10)  # baseline
        return datax

    @pytest.fixture
    def datay(self):
        datay = np.arange(2, 12)  # baseline + 2
        return datay

    def test_rmse_identity(self, datax):
        assert stats.rmse(datax, datax) == approx(0)

    def test_rmse_value(self, datax, datay):
        assert stats.rmse(datax, datay) == approx(2)
        assert stats.rmse(datay, datax) == approx(2)

    def test_mse_identity(self, datax):
        assert stats.mse(datax, datax) == approx(0)

    def test_mse_value(self, datax, datay):
        assert stats.mse(datax, datay) == approx(4)
        assert stats.mse(datay, datax) == approx(4)

    def test_se_identity(self, datax):
        assert stats.squared_error(datax, datax).sum() == approx(0)

    def test_se_value(self, datax, datay):
        assert stats.squared_error(datax, datay).sum() == approx(40)
        assert stats.squared_error(datay, datax).sum() == approx(40)

    def test_mae_identity(self, datax):
        assert stats.mae(datax, datax).sum() == approx(0)

    def test_mae_value(self, datax, datay):
        assert stats.mae(datax, datay).sum() == approx(2)
        assert stats.mae(datay, datax).sum() == approx(2)

    def test_ae_identity(self, datax):
        assert stats.absolute_error(datax, datax).sum() == approx(0)

    def test_ae_value(self, datax, datay):
        assert stats.absolute_error(datax, datay).sum() == approx(20)
        assert stats.absolute_error(datay, datax).sum() == approx(20)

    def test_mbe_identity(self, datax):
        assert stats.mbe(datax, datax).sum() == approx(0)

    def test_mbe_value(self, datax, datay):
        assert stats.mbe(datax, datay).sum() == approx(2)
        assert stats.mbe(datay, datax).sum() == approx(-2)

    def test_be_identity(self, datax):
        assert stats.bias_error(datax, datax).sum() == approx(0)

    def test_be_value(self, datax, datay):
        assert stats.bias_error(datax, datay).sum() == approx(20)
        assert stats.bias_error(datay, datax).sum() == approx(-20)


class TestQuantile:
    """
    Computations of the quantile level for a time series based on prior
    history of days
    """

    @pytest.fixture
    def quantile_data(self):
        """
        Four days of data with a constant value
        """
        index = pd.date_range(start='2019-04-22 00:00:00',
                              end='2019-04-26 23:59:59', freq='1h')
        data = pd.DataFrame(5, index=index, columns=["ColName"])
        return data

    def test_calc_quantile(self, quantile_data):
        data = quantile_data
        p90 = stats.calc_quantile(data, n_days='2d', quantile=0.9)
        npt.assert_allclose(p90['ColName_quant']['2019-04-23 23:00:00':
                                                 '2019-04-24 00:00:00'],
                            [np.nan, 5])

    def test_calc_quantile_series(self, quantile_data):
        data = quantile_data
        p90 = stats.calc_quantile(data['ColName'], n_days='2d', quantile=0.9)
        npt.assert_allclose(p90['ColName_quant']['2019-04-23 23:00:00':
                                                 '2019-04-24 00:00:00'],
                            [np.nan, 5])

    def test_calc_quantile_frame(self, quantile_data):
        data = quantile_data
        dat = np.ones([len(data), 2])
        data2d = pd.DataFrame(dat, index=data.index,
                              columns=["ColName", "Another"])
        with pytest.raises(ValueError):
            stats.calc_quantile(data2d, n_days='2d', quantile=0.9)


class TestVariabilityMetrics:
    @pytest.fixture
    def ghi(self):
        index = pd.date_range(start='2019-04-22 00:00:00',
                              end='2019-04-22 00:00:05', freq='1s')
        ghi = pd.Series([1, 2, 4, 7, 11, 16], index=index)
        return ghi

    @pytest.fixture
    def darr_data(self):
        index = pd.date_range(start='2019-04-22 00:00:00',
                              end='2019-04-26 02:59:59', freq='1min')
        data = pd.DataFrame(5, index=index, columns=["ColName"])
        return data

    @pytest.fixture
    def variability_score_data(self):
        index = pd.date_range(start='2019-04-22 00:00:00',
                              end='2019-04-26 02:59:59', freq='1min')
        data = pd.DataFrame(5, index=index, columns=["ColName"])
        cs = pd.DataFrame(5, index=index, columns=["ColName"])
        data.iloc[5] = 6
        return data, cs

    def test_variability_index_null(self, variability_score_data):
        _, cs = variability_score_data
        assert stats.variability_index(cs, cs, moving_avg_tau=1, norm=False)\
               == approx(1)

    def test_variability_index_basic(self, variability_score_data):
        data, cs = variability_score_data
        assert stats.variability_index(data, cs, moving_avg_tau=1, norm=False)\
               == approx((5937 + 2 * np.sqrt(2)) / 5939)

    def test_variability_index_illegal(self, variability_score_data):
        data, cs = variability_score_data
        with raises(TypeError):
            stats.variability_index(data.values, cs)

    def test_variability_index_movingavg(self, variability_score_data):
        data, cs = variability_score_data
        assert stats.variability_index(data, cs, moving_avg_tau=2, norm=False)\
               == approx((2 * 2967 + 2 * np.sqrt(2 ** 2 + 0.5 ** 2))
                         / (2 * 2969))

    def test_darr_zero(self, darr_data):
        data = darr_data
        npt.assert_allclose(stats.darr(data, pct=False), 0)

    def test_darr_basic(self, darr_data):
        data = darr_data
        data.iloc[2] = 6

        assert stats.darr(data, moving_avg=False, pct=False) == approx(2)
        assert stats.darr(data, pct=False) == approx(2)

    def test_darr_illegal(self, darr_data):
        data = darr_data
        data.iloc[2] = 6
        with raises(TypeError):
            stats.darr(data.values, moving_avg=False, pct=False)

    def test_darr_pct(self, darr_data):
        data = darr_data
        data.iloc[2] = 6
        assert stats.darr(data, pct=True) == approx(2 * 100 / 1000)

    def test_darr_tau(self, darr_data):
        data = darr_data
        data.iloc[2] = 6
        assert stats.darr(data, tau=2, moving_avg=False, pct=False)\
               == approx(2)

    def test_darr_movingavg(self, darr_data):
        data = darr_data
        data.iloc[2] = 6
        assert stats.darr(data, tau=2, moving_avg=True, pct=False) == approx(1)

    def test_variability_score_basic(self, ghi):
        vs = stats.variability_score(ghi, tau=1, moving_avg=False, pct=False)
        assert vs == approx(1.2)  # 2 * 50% chance of being greater

    def test_variability_score_pandas(self, ghi):
        vs = stats.variability_score(pd.Series(ghi), tau=1,
                                     moving_avg=False, pct=False)
        assert vs == approx(1.2)  # 2 * 50% chance of being greater
        vs = stats.variability_score(pd.DataFrame(ghi), tau=1,
                                     moving_avg=False, pct=False)
        assert vs == approx(1.2)  # 2 * 50% chance of being greater

    def test_variability_score_pct(self, ghi):
        vs = stats.variability_score(100 * ghi, tau=1,
                                     moving_avg=False, pct=True)
        assert vs == approx(12)

    def test_variability_score_tau(self, ghi):
        vs = stats.variability_score(ghi, tau=2, moving_avg=False, pct=False)
        assert vs == approx(2.5)  # 3 * 66.667% chance of being greater

    def test_variability_score_movingavg(self, ghi):
        vs = stats.variability_score(ghi, tau=2, moving_avg=True, pct=False)
        assert vs == approx(2)  # 1.5 * 66.667% chance of being greater
