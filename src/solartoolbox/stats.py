import numpy as np
import pandas as pd


def rmse(baseline, estimation):
    """
    Computation of Root Mean Squared Error

    Parameters
    ----------
    baseline : numeric or pandas.Series
        Baseline time series

    estimation : numeric or pandas.Series
        Reference/Estimation time series

    Returns
    -------
    result : numeric
        the root mean squared error
    """
    mse_val = mse(baseline, estimation)
    result = np.sqrt(mse_val)
    return result


def mse(baseline, estimation):
    """
    Computation of Mean Squared Error

    Parameters
    ----------
    baseline : numeric or pandas.Series
        Baseline time series

    estimation : numeric or pandas.Series
        Reference/Estimation time series

    Returns
    -------
    result : numeric
        the mean squared error
    """
    result = np.mean(squared_error(baseline, estimation))
    return result


def squared_error(baseline, estimation):
    """
    Time series of Squared Error

    Parameters
    ----------
    baseline : numeric or pandas.Series
        Baseline time series

    estimation : numeric or pandas.Series
        Reference/Estimation time series

    Returns
    -------
    result : numeric or pandas.Series
        a time series of squared error values
    """
    return bias_error(baseline, estimation) ** 2


def mae(baseline, estimation):
    """
    Computation of Mean Absolute Error

    Parameters
    ----------
    baseline : numeric or pandas.Series
        Baseline time series

    estimation : numeric or pandas.Series
        Reference/Estimation time series

    Returns
    -------
    result : numeric
        the mean absolute error
    """
    result = np.mean(absolute_error(baseline, estimation))
    return result


def absolute_error(baseline, estimation):
    """
    Time series of Absolute Error

    Parameters
    ----------
    baseline : numeric or pandas.Series
        Baseline time series

    estimation : numeric or pandas.Series
        Reference/Estimation time series

    Returns
    -------
    result : numeric or pandas.Series
        a time series of absolute error values
    """
    result = np.abs(bias_error(baseline, estimation))
    return result


def mbe(baseline, estimation):
    """
    Computation of Mean Bias Error

    Parameters
    ----------
    baseline : numeric or pandas.Series
        Baseline time series

    estimation : numeric or pandas.Series
        Reference/Estimation time series

    Returns
    -------
    result : numeric
        the mean bias error
    """
    result = np.mean(bias_error(baseline, estimation))
    return result


def bias_error(baseline, estimation):
    """
    Time series of Bias Error

    Parameters
    ----------
    baseline : numeric or pandas.Series
        Baseline time series

    estimation : numeric or pandas.Series
        Reference/Estimation time series

    Returns
    -------
    result : numeric or pandas.Series
        a time series of bias error values
    """
    result = estimation - baseline
    return result


def variability_score(series, tau=1, moving_avg=True, pct=False):
    """
    Compute the variability score as proposed by Lave et al. [1]. The
    Variability Score is computed by calculating the product of each ramp rate
    that occurs in the time series and the probability of occurrence of larger
    ramp rates than that ramp rate of interest. The maximum value of that
    product over all possible ramp rates is the Variability Score. Its value
    may be represented as a percent.

    The original source computes this quantity using the GHI, but we offer the
    possibility of computing it for clear sky index.

    Parameters
    ----------
    series : pandas.Series or pandas.DataFrame
        a time series for which to calculate the Variability Score. VS will
        be calculated along axis 0.

    tau : numeric, default 1
        The number of timesteps for the increment calculation. series must use
        a temporal index to use a dt greater than 1.

    moving_avg : bool, default True
        When tau specified with a value different from 1, should the timeseries
        be resampled via moving avg to the frequency of tau for computation?

    pct : bool, default False
        should we scale the score as a percent of 1000 W/m2 (0-100%)?

    Returns
    -------
    variability_score : numeric
        the variability score

    [1] M. Lave, M. J. Reno, and R. J. Broderick, “Characterizing local high-
    frequency solar variability and its impact to distribution studies,” Solar
    Energy 118, 327–337 (2015). https://www.osti.gov/pages/biblio/1497655
    """
    if moving_avg and tau > 1:
        dt = (series.index[1] - series.index[0]).total_seconds()
        rs = series.resample(str(int(tau * dt)) + 's').mean()
        return variability_score(rs, tau=1, moving_avg=False, pct=pct)
    if pct:  # Scale to a percentage of STC and multiply by 100 (Lave et al)
        return 100./1000 * variability_score(series, tau=tau,
                                             moving_avg=False, pct=False)

    x = series.diff(tau)
    vs = (x.abs() * (1. - x.abs().rank(pct=True))).max(axis=0)

    return vs


def variability_index(ghi, clearsky, moving_avg_tau=1, norm=False):
    """
    Compute the variability index, defined by Stein et al. [1]. Variability
    Index represents the ratio between the path length of the actual timeseries
    as compared to that of the clearsky.

    As it requires comparison with the clear sky value, this should only be
    computed using the GHI (or other irradiance), and not clear sky index.

    Parameters
    ----------
    ghi : pandas.Series or pd.DataFrame with only a single column
        a series containing the ghi

    clearsky : pandas.Series or pd.DataFrame with only a single column
        a series with the clear sky ghi. Must have a temporal index.

    moving_avg_tau : numeric
        The number of averaging timesteps to use

    norm : bool
        should the output be scaled to represent a dt of 1 minute?

    Returns
    -------
    variability_index : numeric
        the variability index value

    [1] J. S. Stein, C. W. Hansen, and M. J. Reno, “The Variability Index: A
    New and Novel Metric for Quantifying Irradiance and PV Output Variability,”
    in Proceedings of the World Renewable Energy Forum (Denver, CO, 2012)
    pp. 13–17. https://www.osti.gov/biblio/1078490
    """
    if not isinstance(ghi, (pd.Series, pd.DataFrame)):
        raise TypeError('series must be a pandas Series or DataFrame')

    if moving_avg_tau > 1:
        dti = (ghi.index[1] - ghi.index[0]).total_seconds()
        sample_pd = str(int(moving_avg_tau*dti))+'s'
        rs = ghi.resample(sample_pd).mean()
        cs = clearsky.resample(sample_pd).mean()
        return variability_index(rs, cs, moving_avg_tau=1, norm=norm)
    dt = (ghi.index[1] - ghi.index[0]).total_seconds() / 60
    num = np.sqrt(ghi.diff() ** 2 + dt ** 2).sum(axis=0)
    den = np.sqrt(clearsky.diff() ** 2 + dt ** 2).sum(axis=0)
    if norm:  # Stein reports VI_dt = VI_1min/sqrt(dt)
        vi = num/den * np.sqrt(dt)
    else:
        vi = num/den

    return vi

def darr(series, tau=1, moving_avg=True, pct=False):
    """
    Compute the Daily Aggregate Ramp Rate as described by Van Haaren et al. [1]

    The DARR represents the sum of all possible ramp rates, and according to
    the original source, could be normalized as a percent of STC irradiance.

    The original source computes this quantity using the GHI, but we offer the
    possibility of computing it for clear sky index.


    Parameters
    ----------
    series : pandas.Series
        a time series for which to calculate the Variability Score

    tau : numeric, default 1
        The number of timesteps for the increment calculation. series must use
        a temporal index to use a dt greater than 1.

    moving_avg : bool, default True
        When tau specified with a value different from 1, should the timeseries
        be resampled via moving avg to the frequency of tau for computation?

    pct : bool, default False
        should we scale the score as a percent of 1000 W/m2 (0-100%)?

    Returns
    -------
    darr : numeric
        The value of the metric

    [1] R. van Haaren, M. Morjaria, and V. Fthenakis, “Empirical assessment
    of short-term variability from utility-scale solar PV plants,” Progress in
    Photovoltaics: Research and Applications 22, 548–559 (2014),
    https://www.researchgate.net/publication/261603714_Empirical_assessment_of_short-term_variability_from_utility-scale_solar_PV_plants
    """
    if not isinstance(series, (pd.Series, pd.DataFrame)):
        raise TypeError('series must be a pandas Series or DataFrame')

    if moving_avg and tau > 1:
        dt = (series.index[1] - series.index[0]).total_seconds()
        rs = series.resample(str(int(tau * dt)) + 's').mean()
        # Rerun it without MA
        return darr(rs, tau=1, moving_avg=False, pct=pct)
    if pct:
        return darr(series * 100 / 1000, tau=tau, moving_avg=False, pct=False)

    darr_val = np.abs(series.diff(tau)).sum(axis=0)

    return darr_val


def calc_quantile(timeseries, n_days="30d", quantile=0.9):
    """
    Calculate a single-day percentile-based summary of data by aggregating
    multiple days.

    So the timeseries output on any given day represents the quantile-th
    percentile value over the past n-days during any particular timestamp.

    An example usage of this is in the QCPV algorithm of Killinger et al. [1],
    who use it on a time series of PV data to compute an empirical 'Clear Sky
    PV' value that can be used to simulate a PV clear sky index.

    Parameters
    ----------
    timeseries : pandas.Series or pandas.DataFrame
        the timeseries of input data. Needs to just be a single column or a
        Series

    n_days : str, default '30d'
        the number of days window to use for the calculation on quantile in a
        format readable by pandas.to_timedelta()

    quantile : numeric, default 0.9
        the quantile level as a fraction

    Returns
    -------
    output : pandas.DataFrame
        output timeseries. nan for first days of the series until enough days
        for the window to be met. The data will be stored in a column based on
        the input column name with "_quant" added.

    [1] Killinger, Sven & Engerer, Nick & Müller, Björn. (2017). QCPV: A
    quality control algorithm for distributed photovoltaic array power output.
    Solar Energy. 143. 120-131.
    https://www.researchgate.net/publication/312145487_QCPV_A_quality_control_algorithm_for_distributed_photovoltaic_array_power_output
    """

    ts = timeseries.copy()  # copy the data

    out_df = pd.Series()  # Create an empty holder

    end_day = ts.index.date[-1]  # the final day of the dataset

    # Loop over all unique days in the timeseries
    for date_start in pd.unique(ts.index.date):
        # Window starts on date_start. Window ends on date_end
        date_end = date_start + pd.to_timedelta(n_days)

        # stop when the window tries to reach beyond the end of the data
        if date_end > end_day:
            break

        # Copy the subset of data representing the window
        # and separate the date and time stamps
        statdata = ts[date_start:date_end]
        dates = statdata.index.date
        times = statdata.index.time

        # Build into a new DataFrame that can be pivoted
        statdata = pd.DataFrame({'data': statdata.values.flatten(), 'd': dates,
                                 't': times}, index=statdata.index)

        # Create a pivot table based on the time as the index so that we can
        # compute statistics across all the days
        piv = pd.pivot(statdata, index='t', columns='d', values='data')

        # Calculate the quantile
        p90 = piv.quantile(quantile, axis=1)
        p90.name = date_end  # Assign a name based on the end date

        # p90's index is only time, so modify it to include the full date
        p90.index = pd.to_datetime(np.repeat(str(date_end) + " ",
                                             len(p90.index)) +
                                   p90.index.astype(str))

        # Concat this day onto the output object
        out_df = pd.concat((out_df, p90), axis=0)

    # localize the timeseries and the column names back to the input
    out_df = out_df.tz_localize(ts.index.tz).reindex(ts.index)
    out_df = pd.DataFrame(out_df)
    try:
        out_df.columns = [name+'_quant' for name in ts.columns]
    except AttributeError:
        try:
            out_df.columns = [ts.name + '_quant']
        except AttributeError:
            raise AttributeError('Unable to generate name of column')
        except TypeError:
            out_df.columns = ['quantile']

    return out_df
