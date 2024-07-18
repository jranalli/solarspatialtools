import numpy as np
import pandas as pd
import scipy.signal
from scipy import signal
from scipy.interpolate import interp1d
from scipy.optimize import curve_fit, leastsq

from solartoolbox import spatial


def correlation(baseline, estimation, scaling='coeff'):
    """
    Compute the cross correlation between two signals, including the full range
    of possible time lags.

    The value of correlation at lags==0 is the traditional cross-correlation
    without allowing for time shifting.

    This function is essentially a wrapper for scipy.stats.correlate, providing
    pre-baked scaling options, and always returning the lags.

    Parameters
    ----------
    baseline : numeric or pandas.Series
        The baseline signal

    estimation : numeric or pandas.Series
        The predicted signal

    scaling : str
        a string for the type scaling to use.

        Options are:
            `energy`:
                scales by the energy of the autocorrelation of the input
            `coeff`:
                scales the output to the correlation coefficient (always
                removes the mean from both signals)
            `unbiased_energy`:
                similar to energy, but normalizes based on lag to account for
                the fewer points used in the convolution. Removes bias towards
                small lags.
            `unbiased_coeff`:
                similar to coeff, but normalizes based on lag to
                account for the fewer points used in the
                convolution. Removes bias towards small lags.

    Returns
    -------
    corr : numeric
        A vector of the cross correlations

    lag : numeric
        A vector of the lag for each cross correlation
    """

    # The time lags
    lags = scipy.signal.correlation_lags(len(baseline), len(estimation))

    if scaling.lower() == 'energy':
        autocorr = scipy.signal.correlate(baseline, baseline)
        energy = np.max(autocorr)
        corr = scipy.signal.correlate(baseline, estimation) / energy
    elif scaling.lower() == 'coeff':
        num = scipy.signal.correlate(baseline - np.mean(baseline),
                                     estimation - np.mean(estimation))
        den = len(baseline) * (np.std(baseline) * np.std(estimation))
        corr = num / den
    elif scaling.lower() == 'unbiased_energy':
        autocorr = scipy.signal.correlate(baseline, baseline)
        energy = np.max(autocorr)
        corr = scipy.signal.correlate(baseline, estimation) / energy
        scale = (len(baseline) - np.abs(lags)) / len(baseline)
        corr /= scale
    elif scaling.lower() == 'unbiased_coeff':
        num = scipy.signal.correlate(baseline - np.mean(baseline),
                                     estimation - np.mean(estimation))
        den = len(baseline) * (np.std(baseline) * np.std(estimation))
        corr = num / den
        scale = (len(baseline) - np.abs(lags)) / len(baseline)
        corr /= scale
    elif scaling.lower() == 'none':
        corr = scipy.signal.correlate(baseline, estimation)
    else:
        raise ValueError("Illegal scaling specified.")

    return corr, lags


def _fftcorrelate(baseline, estimation, scaling='coeff'):
    """
    Compute the cross correlation between two signals, including the full range
    of possible time lags, using cross correlation. Can handle multiple time
    series.

    Parameters
    ----------
    baseline: pd.DataFrame
    estimation: pd.DataFrame
    scaling: str (default 'coeff')
        Type of scaling to use for cross correlation. Note that debiased
        scaling options are not available here. Valid options are:
        'coeff' - computes the correlation coefficient
        'none' - no scaling is applied

    Returns
    -------
    corr : numeric
        A vector of the cross correlations for each pairing in the inputs

    lag : numeric
        A vector of the lag for each cross correlation
    """

    # Compute the lags
    lags = signal.correlation_lags(len(baseline), len(estimation))

    # Rename for brevity
    ts_inm = baseline
    ts_outm = estimation

    # Condition pandas input types
    if isinstance(ts_inm, pd.DataFrame):
        ts_inm = np.array(ts_inm).T
    if isinstance(ts_outm, pd.DataFrame):
        ts_outm = np.array(ts_outm).T

    # Condition vector inputs
    if ts_inm.ndim == 1:
        ts_inm = np.expand_dims(ts_inm, axis=0)
    if ts_outm.ndim == 1:
        ts_outm = np.expand_dims(ts_outm, axis=0)

    # Perform the scaling calculations
    if scaling == 'coeff':
        # Subtract means
        ts_inm = ts_inm - np.expand_dims(np.mean(ts_inm, axis=1), axis=1)
        ts_outm = ts_outm - np.expand_dims(np.mean(ts_outm, axis=1), axis=1)
        # Compute the scaling factor (normalizing by stdev)
        corr_scale = (np.size(ts_inm, axis=1)
                      * np.std(ts_inm, axis=1)
                      * np.std(ts_outm, axis=1)) ** -1
    elif scaling == 'none':
        corr_scale = 1
    else:
        raise ValueError(f"Illegal scaling specified: {scaling}.")

    na = np.size(ts_inm, axis=1)
    nl = 2 * na - 1
    addon = np.zeros((ts_inm.shape[0], nl - na))
    ts_inm = np.concatenate([ts_inm, addon], axis=1)
    ts_outm = np.concatenate([ts_outm, addon], axis=1)

    # Compute correlation via fft and re-center to match lags
    ffta = scipy.fft.fft(ts_inm, axis=1)
    fftb = scipy.fft.fft(ts_outm, axis=1)
    corrxy = np.real(np.fft.ifft(ffta * np.conj(fftb), axis=1)) * corr_scale

    corrxy = np.roll(corrxy, nl // 2, axis=1)

    return corrxy, lags


def averaged_psd(input_tsig, navgs, overlap=0.5,
                 window='hamming', detrend='linear', scaling='density'):
    """
    Calculate an averaged power spectral density for a signal.

    Parameters
    ----------
    input_tsig : numeric
        Pandas type with the TF input time signal. Index must be time.

    navgs : int
        The number of averages to use based on zero overlap. Overlap will
        result in more averages.

    overlap : float (default 0.5)
        Percentage overlap between the averages

    window : string (default 'hamming')
        The window type to use.

    detrend : string
        Detrend type ('linear' or 'constant'). See scipy.signal.welch for more
        information.

    scaling : string (default 'density')
        The type of scaling to request from scipy. See scipy.signal.welch for
        more info

    Returns
    -------
    output : Series
        Pandas Series containing the power spectral density with an index of
        the frequency.
    """
    dt = (input_tsig.index[1] - input_tsig.index[0]).total_seconds()
    fs = 1/dt

    if isinstance(input_tsig, pd.Series):
        return averaged_psd(pd.DataFrame(input_tsig), navgs, overlap, window,
                            detrend)

    cols = input_tsig.columns
    input_tsig = input_tsig.values.T

    nperseg = int(input_tsig.shape[1] // navgs)
    noverlap = int(nperseg * overlap)

    f, psdxx = signal.welch(input_tsig, fs=fs, window=window,
                            nperseg=nperseg, detrend=detrend,
                            noverlap=noverlap, scaling=scaling)
    # Reported units from scipy are V**2/Hz
    return pd.DataFrame(psdxx.T, index=f, columns=cols)


def averaged_tf(input_tsig, output_tsig,
                navgs, overlap=0.5, window='hamming', detrend='linear'):
    """
    Calculate a transfer function between two signals, along with their
    coherence.

    Parameters
    ----------
    input_tsig : numeric
        Pandas type with the TF input time signal. Index must be time.

    output_tsig : numeric
        Pandas type with the TF output time signal. Index must be time.

    navgs : int
        The number of averages to use based on zero overlap. Overlap will
        result in more averages.

    overlap : float (default 0.5)
        Percentage overlap between the averages

    window : string (default 'hamming')
        The window type to use.

    detrend : string or None
        Detrend type ('linear' or 'constant'). See scipy.signal.psdxx for more
        information.

    Returns
    -------
    tf : pd.DataFrame
        Complex valued transfer function indexed by the frequency and
        containing the same columns as the input
    coh : pd.DataFrame
        the coherence indexed by the frequency and containing the same
        columns as the input
    """
    dt = (input_tsig.index[1] - input_tsig.index[0]).total_seconds()
    fs = 1/dt

    if isinstance(output_tsig, pd.Series):
        return averaged_tf(input_tsig, pd.DataFrame(output_tsig),
                           navgs, overlap, window, detrend)
    if isinstance(input_tsig, pd.Series):
        return averaged_tf(pd.DataFrame(input_tsig), output_tsig, navgs,
                           overlap, window, detrend)

    if not (isinstance(output_tsig, pd.DataFrame)
            and isinstance(input_tsig, pd.DataFrame)):
        raise ValueError('Input and output timesignals must be pandas types')

    out_cols = output_tsig.columns
    in_cols = input_tsig.columns
    output_tsig = output_tsig.values.T
    input_tsig = input_tsig.values.T

    if len(out_cols) == len(in_cols):
        cols = [f"{in_cols[i]}_{out_cols[i]}" for i in range(len(out_cols))]
    elif len(out_cols) == 1:
        cols = [f"{in_cols[i]}_{out_cols[0]}" for i in range(len(in_cols))]
    elif len(in_cols) == 1:
        cols = [f"{in_cols[0]}_{out_cols[i]}" for i in range(len(out_cols))]
    else:
        raise RuntimeError("Shouldn't Be Possible")

    nperseg = int(input_tsig.shape[1] // navgs)
    noverlap = int(nperseg * overlap)

    # Calculate the transfer function
    _, psdxx = signal.welch(input_tsig, fs=fs, window=window,
                            nperseg=nperseg, detrend=detrend,
                            noverlap=noverlap, scaling='density')
    _, psdyy = signal.welch(output_tsig, fs=fs, window=window,
                            nperseg=nperseg, detrend=detrend,
                            noverlap=noverlap, scaling='density')
    freqs, csdxy = signal.csd(input_tsig, output_tsig, fs=fs, window=window,
                              nperseg=nperseg, detrend=detrend,
                              noverlap=noverlap)

    # Calculating manually allows for vectorization and is faster
    with np.errstate(divide='ignore', invalid='ignore'):
        tf = csdxy / psdxx
        coh = np.abs(csdxy) ** 2 / (psdxx * psdyy)

    # Calculate the coherence
    # _, coh = signal.coherence(input_tsig, output_tsig, fs=fs, window=window,
    #                           nperseg=nperseg, noverlap=noverlap,
    #                           detrend=detrend)

    tf = pd.DataFrame(tf.T, index=freqs, columns=cols)
    coh = pd.DataFrame(coh.T, index=freqs, columns=cols)

    return tf, coh


def interp_tf(new_freq, input_tf):
    """
    Interpolate a transfer function in the frequency domain by magnitude and
    phase independently. This is necessary because the complex interpolation
    doesn't really do the job on its own.

    Parameters
    ----------
    new_freq : np.array or pd.Index
        The new frequency index to interpolate onto.

    input_tf : pd.Series or pd.DataFrame
        The transfer function to be interpolated

    Returns
    -------
    interp_tf : pd.Series or pd.DataFrame
        The transfer function interpolated to the new frequency axis. Type will
        match the type of the input.
    """
    sortinds = input_tf.index.argsort()
    if type(input_tf) is type(pd.Series()):
        use_tf = pd.DataFrame(input_tf)
    else:
        use_tf = input_tf

    # Generate a function handle and interpolate the magnitude
    interp_mag_func = interp1d(use_tf.index[sortinds],
                               np.abs(use_tf.iloc[sortinds, :]),
                               axis=0)
    interp_mag = interp_mag_func(new_freq)

    # Generate a function handle and interpolate the phase
    # Work on the unwrapped angle to make sure that we don't have weird
    # results in the middle of wraps.
    interp_phase_func = interp1d(use_tf.index[sortinds],
                                 np.unwrap(np.angle(use_tf.iloc[sortinds, :]),
                                           axis=0),
                                 axis=0)
    interp_phase = interp_phase_func(new_freq)

    # Recreate the complex TF
    interp_filt = interp_mag * np.exp(1j * interp_phase)

    # Appropriately recast the type
    if type(input_tf) is type(pd.Series()):
        interp_filt = pd.Series(interp_filt[:, 0], index=new_freq)
    else:
        interp_filt = pd.DataFrame(interp_filt, columns=input_tf.columns,
                                   index=new_freq)
    return interp_filt


def tf_delay(tf, coh, coh_limit=0.6, freq_limit=0.02, method='fit'):
    """
    Compute the delay based on the phase of a transfer function

    Parameters
    ----------
    tf : pd.DataFrame
        The transfer functions, produced by signalproc.averaged_tf.
    coh : pd.DataFrame
        The coherence, produced by signalproc.averaged_tf. Must match shape
        of tf.
    coh_limit : float
        The coherence limit to use for filtering the phase. Set to None to
        include all points.
    freq_limit : float
        The frequency limit to use for filtering the phase. Set to None to
        include all points.
    method : str
        The method to use for computing the delay.

        Options are:
            'diff' - unwrap phase and take derivative (not well validated)
            'fit' - fit a line to the phase
            'multi' - fit a line to the phase for multiple tfs at once

    Returns
    -------
    delay : float
        The delay in seconds
    ix : np.array(bool)
        The mask used for filtering the phase
    """

    def _delay_fitter(x, delval):
        """
        Helper function for use with curve_fit. Computes the modeled phase.
        Parameters
        ----------
        x :
            the transfer function frequency
        delval :
            The 'real' phase to fit the group delay to
        Returns
        -------
        model :
            the modeled phase
        """
        # model = np.unwrap(np.angle(np.ones_like(x) *
        #                            np.exp(2 * np.pi * 1j * x * -delval)))
        # Simplifies to a linear relationship
        model = 2 * np.pi * x * -delval
        return model

    def _residual(dels, freqs, phases, mask):
        """
        Helper function for use with leastsq. Computes the residual between
        the modeled phase and the measured phase.
        Parameters
        ----------
        dels : float
            The array of delay to use for the model
        freqs : float
            The frequency array (must be multidimensional)
        phases : float
            The measured phase array (must be multidimensional)
        mask : bool
            The mask to use for filtering the phase

        Returns
        -------
        residual : float
            The residual between the model and the measured phase
        """
        # model = np.unwrap(np.angle(np.ones_like(freqs) *
        #                            np.exp(2 * np.pi * 1j * freqs * -dels)),
        #                   axis=0)
        # Simplifies to a linear relationship
        model = 2*np.pi*freqs*-dels
        diff = (phases - model) * mask
        return np.nansum(diff, axis=0)

    if isinstance(tf, pd.Series):
        return tf_delay(pd.DataFrame(tf), coh, coh_limit, freq_limit, method)
    if isinstance(coh, pd.Series):
        return tf_delay(tf, pd.DataFrame(coh), coh_limit, freq_limit, method)

    if coh_limit is None:
        ix_coh = np.ones_like(coh, dtype=bool)
    else:
        ix_coh = (coh.values > coh_limit)
    if freq_limit is None:
        ix_freq = np.ones_like(tf, dtype=bool)
    else:
        ix_freq = np.zeros_like(tf, dtype=bool)
        ix_freq[tf.index < freq_limit, :] = True
    ix = np.bitwise_and(ix_coh, ix_freq)

    if method == 'diff':
        if not np.any(np.array(np.shape(tf)) == 1):
            raise ValueError('tf must be a 1D array for method: fit')

        # # Method 1: unwrap phase and take derivative
        tfv = tf.values.flatten()
        gd = -np.diff(np.unwrap(np.angle(tfv))) / np.diff(tf.index)
        gd = np.append(gd, gd[-1])
        gd = gd/(2*np.pi)
        avg_del = np.sum(gd * ix.flatten() / np.sum(ix))
        return avg_del, ix

    elif method == 'fit':
        # Method 2: curve fit the phase
        if not np.any(np.array(np.shape(tf)) == 1):
            raise ValueError('tf must be a 1D array for method: fit')
        try:
            return (curve_fit(_delay_fitter,
                              np.expand_dims(tf.index, axis=1)[ix],
                              np.unwrap(np.angle(tf.loc[ix]).flatten()))[0],
                    ix)
        except ValueError:
            from warnings import warn
            if not ix.any():
                warn('Curve fit failed due to coherence limit. Returning NaN')
                return np.nan, ix

            else:
                warn('Curve fit failed for unknown reason. Returning NaN')
                return np.nan, ix
    elif method == "multi":
        # Method 3 : Operate on multiple tfs at a time using fit method

        # The guess for the delays
        guess = np.zeros((1, np.size(tf, axis=1)))

        # Get subsets on the frequency axis to simplify fitting
        ix_freq1 = ix_freq[:, 0]
        if isinstance(tf, (pd.DataFrame, pd.Series)):
            tf_subset = tf.values[ix_freq1, :]
        else:
            tf_subset = tf[ix_freq1, :]

        # The arguments for the residual function
        args = (np.expand_dims(tf.index[ix_freq1], axis=1),
                np.unwrap(np.angle(tf_subset), axis=0),
                ix_coh[ix_freq1, :])
        # Perform the simultaneous least squares fit
        best, cov = leastsq(_residual, guess, args=args)

        best[np.all(np.isnan(tf), axis=0)] = np.nan
        return best, ix
    else:
        raise ValueError(f'Invalid method: {method}')


def xcorr_delay(ts_in, ts_out, scaling='coeff'):
    """
    Compute the delay between two timeseries using cross correlation.

    Parameters
    ----------
    ts_in : pd.Series
        The input timeseries. Requires that the index operate as datetime.
    ts_out : pd.Series
        The output timeseries
    scaling : str
        Type of scaling to use for cross correlation.

        Options are:
            `energy`:
                scales by the energy of the autocorrelation of the input
            `coeff`:
                scales the output to the correlation coefficient (always
                removes the mean from both signals)
            `unbiased_energy`:
                similar to energy, but normalizes based on lag to account for
                the fewer points used in the convolution. Removes bias towards
                small lags.
            `unbiased_coeff`:
                similar to coeff, but normalizes based on lag to account for
                the fewer points used in the convolution. Removes bias towards
                small lags.

    Returns
    -------
    delay : float
        Time lag between the two timeseries at the maximum value of the cross
        correlation. Values are always integer multiples of the sampling period
        as the max correlation values are limited to the discrete time steps.
    corr : float
        The peak value of the cross correlation at the identified delay.
    """
    # Method 3: Cross Correlation
    xcorr_i, lags = correlation(ts_in, ts_out, scaling)
    dt = (ts_in.index[1] - ts_in.index[0]).total_seconds()
    lags = lags * dt
    peak_lag_index = xcorr_i.argmax()  # Index of peak correlation
    delay = -lags[peak_lag_index]
    return delay, xcorr_i[peak_lag_index]


def compute_delays(ts_in, ts_out, mode='loop', scaling='coeff'):
    """
    Compute the delay between two sets of timeseries using cross correlation.
    Uses similar methodologies to xcorr_delay, but allows for multiple input
    timeseries to be computed simultaneously.

    Parameters
    ----------
    ts_in: pd.DataFrame
        The time series representing the input signal
    ts_out: pd.DataFrame
        The time series representing the output signal
    mode: str (default 'loop')
        The method to use for computing the delay.

        Options are:
            `loop`:
                loop over each input timeseries pair and compute the xcorr
            `fft`:
                use the fft to compute the xcorr (seems to be slower)

    scaling: str (default 'coeff')
        Type of scaling to use for cross correlation. Note that debiased
        scaling options are not available here.

        Valid options are:
            `coeff`:
                computes the correlation coefficient
            `none`:
                no scaling is applied

    Returns
    -------
    delay : array(float)
        Time lag between the two timeseries at the maximum value of the cross
        correlation. Values are always integer multiples of the sampling period
        as the max correlation values are limited to the discrete time steps.
    extra_data : dict{}
        Additional data about the correlations. Values are:
        'peak_corr' - the peak value of the correlation for each input ts
        'mean_corr' - the mean correlation for each input timeseries
        'zero_corr' - the correlation at zero_lag for each input timeseries
    """
    lags = signal.correlation_lags(len(ts_in), len(ts_in))
    dt = (ts_in.index[1] - ts_in.index[0]).total_seconds()
    lags = lags * dt

    ts_inm = ts_in
    ts_outm = ts_out
    if isinstance(ts_in, pd.DataFrame):
        ts_inm = np.array(ts_inm).T
    if isinstance(ts_out, pd.DataFrame):
        ts_outm = np.array(ts_outm).T

    extras = {'peak_corr': np.zeros(ts_inm.shape[0]),
              'mean_corr': np.zeros(ts_inm.shape[0]),
              'zero_corr': np.zeros(ts_inm.shape[0])
              }
    zero_lag_ind = np.where(lags == 0)

    # Perform the scaling calculations
    if scaling == 'coeff':
        # Subtract means
        ts_inm = ts_inm - np.expand_dims(np.mean(ts_inm, axis=1), axis=1)
        ts_outm = ts_outm - np.expand_dims(np.mean(ts_outm, axis=1), axis=1)
        # Compute the scaling factor (normalizing by stdev)
        corr_scale = (np.size(ts_inm, axis=1)
                      * np.std(ts_inm, axis=1)
                      * np.std(ts_outm, axis=1)) ** -1
    elif scaling == 'none':
        corr_scale = 1
    else:
        raise ValueError(f"Illegal scaling specified: {scaling}.")

    if mode == 'loop':
        # signal.correlate can't handle 2D inputs without performing 2D conv.

        # Precomputing method saves time
        method = signal.choose_conv_method(ts_inm[0, :], ts_outm[0, :])

        # Preallocate and Compute for individual pairs
        delay = np.zeros(ts_inm.shape[0])
        for i in range(ts_outm.shape[0]):
            xcorr = signal.correlate(ts_inm[i, :], ts_outm[i, :],
                                     method=method)
            # xcorr, _ = correlation(ts_inm[i, :], ts_outm[i, :], 'none')
            peak_lag_index = xcorr.argmax()
            delay[i] = -lags[peak_lag_index]

            extras['peak_corr'][i] = xcorr[peak_lag_index]
            extras['mean_corr'][i] = np.mean(xcorr)
            extras['zero_corr'][i] = xcorr[zero_lag_ind]

        # Scale
        extras['peak_corr'] *= corr_scale
        extras['mean_corr'] *= corr_scale
        extras['zero_corr'] *= corr_scale

    elif mode == 'fft':
        # Have to pad with zeros to get equivalent vals to normal xcorr.
        na = np.size(ts_inm, axis=1)
        nl = 2*na - 1
        addon = np.zeros((ts_inm.shape[0], nl - na))
        ts_inm = np.concatenate([ts_inm, addon], axis=1)
        ts_outm = np.concatenate([ts_outm, addon], axis=1)

        # Need to recompute the lags to accommodate the shifted zero for fft
        lags = np.roll(lags, -nl // 2 + 1)
        zero_lag_ind = np.where(lags == 0)

        # Compute correlation via fft and re-center to match lags
        ffta = scipy.fft.fft(ts_inm, axis=1)
        fftb = scipy.fft.fft(ts_outm, axis=1)
        corrxy = np.real(np.fft.ifft(ffta * np.conj(fftb), axis=1))

        # Extract value and lag of peak correlation
        peak_lag_indices = corrxy.argmax(axis=1)
        delay = -lags[peak_lag_indices]

        extras['peak_corr'] = (corrxy[range(ts_inm.shape[0]), peak_lag_indices]
                               * corr_scale)
        extras['mean_corr'] = np.mean(corrxy, axis=1) * corr_scale
        extras['zero_corr'] = corrxy[:, zero_lag_ind] * corr_scale
    else:
        raise ValueError(f'Invalid mode: {mode}')

    # Sometimes produce erroneous delay for uncorrelated values
    delay[extras['peak_corr'] < 1e-10] = 0

    return delay, extras


def apply_delay(tf, delay):
    """
    Apply a time delay to a transfer function. This is equivalent to rotating
    the phase as a linear function of frequency.

    To compare with a unity transfer function, consider:
        apply_delay(tf * 0 + 1, delay)

    Parameters
    ----------
    tf : pd.DataFrame
        The transfer function, produced by signalproc.averaged_tf. The index
        must contain the frequency in Hz.
    delay : float
        The delay to apply to the transfer function. Units are in seconds.

    Returns
    -------
    tf : pd.DataFrame
        A copy of the transfer function with the delay applied in rotating
        the phase.
    """

    tfi = tf.copy().values.flatten()

    # Equation for rotating the phase by a delay
    tf = tfi * np.exp(2 * np.pi * 1j * tf.index * -delay)

    return tf


def get_1d_plant(centers, ref_center=0,
                 width=None, shape="square",
                 dx=1, xmax=500000):
    """
    Generate a one dimensional plant array based on a list of center positions.
    Plant is essentially a comb filter with a site of a given shape placed at
    each specified center position

    Parameters
    ----------
    centers : numeric
        List of centers of the individual measurement locations. Commonly the
        output of spatial.project_vectors().

    ref_center : numeric
        Position of the reference, will be used as the zero of the x coordinate

    width : numeric
        The size of each individual plant component. If None, is equivalent to
        the dx for the plant.

    shape : string
        The shape to use for each individual plant component. Choices are:
        'square', 'triangle', 'gaussian'

    dx : numeric
        The x axis spacing to use for the numerical plant layout.

    xmax : numeric
        The maximum x size to use for the plant domain

    Returns
    -------
    plant : numeric
        A vector representing the plant's density of generation along the x
        axis

    x_vec : numeric
        The position axis for the plant.
    """

    if width is None:
        w = dx
    else:
        w = width

    centers = np.array(centers).flatten()
    centers -= ref_center

    # Initialize the empty plant
    x_vec = np.arange(-xmax//2, xmax//2, dx)
    plant = np.zeros(x_vec.shape, dtype=float)

    # Creating the individual plant windows ###############

    if shape.lower() == "square":
        # Square individual plants
        # Lc = L total plant
        # north = # of sites
        # Lc/north = separation
        for center in centers:
            inds = np.bitwise_and(x_vec >= (center - w / 2),
                                  x_vec < (center + w / 2))
            plant[inds] = 1
    elif shape.lower() == "triangle":
        for center in centers:
            inds = np.bitwise_and(x_vec >= (center - w / 2),
                                  x_vec < (center + w / 2))
            plant[inds] = x_vec[inds] - center+w/2
    elif shape.lower() == "gaussian":
        # Gaussian Window
        for center in centers:
            plant += np.exp(-(x_vec-center)**2/(2*(w/2.355)**2))  # FWHM is STD
    else:
        raise ValueError("No info for plant shape: {}".format(shape))

    return plant, x_vec


def plant1d_to_camfilter(plant, x_plant, cloud_speed):
    """
    Take a 1D plant and compute the Cloud Advection Model representation

    Parameters
    ----------
    cloud_speed : numeric
        The cloud motion vector speed

    plant : np.array
        An array-based representation of the plant generation density. Will be
        normalized to produce a transfer function DC magnitude of 1. See
        get_1d_plant().

    x_plant : np.array
        The plant's x-coordinate. Should have a value of zero at the location
        of the reference point. See get_1d_plant().

    Returns
    -------
    filter : pd.Series
        A pandas series with the complex valued transfer function, indexed by
        the corresponding frequency.
    """
    # TODO needs to be validated

    dx = x_plant[1]-x_plant[0]

    plant = plant / np.sum(plant)  # normalize the plant
    camfilt = np.fft.fft(plant)  # What does it look like in f domain
    spatialdt = dx / np.abs(cloud_speed)  # Effective dt for cloud motion
    camfreq = np.fft.fftfreq(plant.shape[-1], spatialdt)

    # Shift the phase
    t_delay = np.min(x_plant) / cloud_speed
    if cloud_speed > 0:
        camfilt = camfilt * np.exp(
            1j * camfreq * (2 * np.pi) * t_delay)
    else:
        camfilt = np.conj(
            camfilt * np.exp(1j * camfreq * (2 * np.pi) * -t_delay))
    return pd.Series(camfilt, index=camfreq)


def apply_filter(input_tsig, comp_filt):
    """
    Apply a filter to a signal, and return the filtered signal. Works to align
    the frequency axis of the computed filter with the input signal.

    Parameters
    ----------
    input_tsig : pandas.Series or DataFrame
        Pandas type that contains the time signal

    comp_filt : Series, DataFrame
        Pandas type containing the complex valued filter to apply with its
        frequency in the index. See for example: get_camfilter

    Returns
    -------
    filtered_sig : numeric
        The filtered time series.
    """
    # Get the fft of the input signal, including its frequency axis
    dt = (input_tsig.index[1] - input_tsig.index[0]).total_seconds()
    input_fft = np.fft.fft(input_tsig) * 2 / len(input_tsig)
    f_vec = np.fft.fftfreq(input_tsig.shape[-1], dt)

    if np.max(f_vec) > np.max(comp_filt.index):
        raise ValueError('Error: the TF to apply does not cover the entire '
                         'frequency axis needed for the signal. Please '
                         'provide a TF with a higher maximum frequency.')

    # Interpolate the computational
    interp_filt = interp_tf(f_vec, comp_filt)

    # Apply the filter and invert.
    filtered_fft = input_fft * interp_filt
    filtered_sig = np.fft.ifft(filtered_fft * len(input_tsig) / 2)
    filtered_sig = np.real(filtered_sig)
    filtered_sig = pd.Series(filtered_sig, index=input_tsig.index)

    return filtered_sig


def get_camfilter(positions, cloud_speed, cloud_dir, ref_position, dx=1,
                  **kwargs):
    """
    Compute the filter for the CAM model

    Parameters
    ----------
    positions : pandas.DataFrame
        Pandas object containing locations of each reference site within the
        overall plant. Must be indexed by the site id. See data storage format.

        If positions contain 'lat' and 'lon' columns, they will be converted
        to UTM assuming latitude and longitude in degrees. Otherwise, it will
        be assumed that they are already in a UTM-like coordinate system.

    cloud_speed : numeric
        The cloud motion speed

    cloud_dir : tuple
        A tuple (dx,dy) representing the cloud motion direction. Will be
        converted to a unit vector, so length is not important.

    ref_position : pandas.DataFrame
        A subset of positions that represents the position of the reference.

    dx : numeric
        The spatial spacing that should be used in representing the plant.
        Affects the frequency band that can be represented.

    **kwargs : various
        Parameters that will be passed to get_1D_plant(). Include
            'width' - numeric width of each centered object
            'shape' - shape of each centered object (e.g. 'square')
            'xmax' - numeric maximum value in the spatial domain for the plant.
                     Affects the frequency resolution of the filter.
    Returns
    -------
    camfilter : Series
        A pandas Series containing the complex valued filter, along with its
        frequency vector along the index.
    """
    try:
        pos_utm = spatial.latlon2utm(positions['lat'], positions['lon'])
    except KeyError:
        pos_utm = positions
    try:
        ref_utm = spatial.latlon2utm(ref_position['lat'], ref_position['lon'])
        ref_utm = pd.Series(ref_utm, index=['E', 'N', 'zone'])
    except KeyError:
        ref_utm = ref_position

    pos_vecs = spatial.compute_vectors(pos_utm['E'], pos_utm['N'],
                                       ref_utm[['E', 'N']])
    pos_dists = spatial.project_vectors(pos_vecs, cloud_dir)

    plant, x_plant = get_1d_plant(pos_dists, dx=dx, **kwargs)
    camfilter = plant1d_to_camfilter(plant, x_plant, cloud_speed)
    return camfilter


def get_marcosfilter(s, freq=None):
    """
    Compute the filter for the Marcos model

    Parameters
    ----------
    s : numeric
        plant size in Hectares

    freq : numeric (default None)
        A vector of frequencies to include. A reference array will be computed
        if no frequency is provided.

    Returns
    -------
    output : Series
        A pandas Series with the complex valued filter. Index is frequency.

    """
    if freq is None:
        freq = np.linspace(0, 0.5, 100)
    k = 1
    fc = 0.02 / np.sqrt(s)
    filt = k / (1j * freq / fc + 1)
    return pd.Series(filt, index=freq, dtype=np.complex64)


def cleanfreq(sig):
    """
    Cleanup the bidirectional frequencies of a filter object for better
    visualization without lines wrapping across the zero.

    Parameters
    ----------
    sig : pandas.Series
        An object with an index of frequency that will be adjusted

    Returns
    -------
    The signal object with modified frequency
    """
    idxlist = sig.index.to_list()
    idxlist[len(sig.index) // 2] = None
    sig.index = idxlist
