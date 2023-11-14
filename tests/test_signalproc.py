import pytest
from pytest import approx, raises
import pandas as pd
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from solartoolbox.signalproc import averaged_psd, averaged_tf, interp_tf, tf_delay, xcorr_delay, apply_delay, correlation, compute_delays, _fftcorrelate


@pytest.fixture(params=[0, 0.2, -0.2, 0.4, -0.4, 1, -1])
def corr_data(request):
    dt = 0.01
    dly = request.param  # variable delay
    t = np.arange(0, 10, 0.01)
    f = 1

    # Single half-period pulse of sine wave
    x1 = np.sin(2 * np.pi * f * t)
    x1[t < 4*(1/f)] = 0
    x1[t > 4.5*(1/f)] = 0

    # Delayed version
    x2 = np.sin(2 * np.pi * f * (t - dly))  # delay by 0.2 seconds
    x2[t < 4*(1/f) + dly] = 0
    x2[t > 4.5*(1/f) + dly] = 0
    return dt, t, x1, x2, dly


@pytest.fixture(params=['energy', 'coeff', 'unbiased_energy', 'unbiased_coeff'])
def scaling(request):
    return request.param


def test_correlation_identity(corr_data, scaling):
    dt, t, x1, x2, dly = corr_data
    c, lag = correlation(x1, x1, scaling=scaling)
    imax = np.argmax(c)

    assert c[imax] == approx(1)
    assert -lag[imax]*(t[1]-t[0]) == approx(0)


def test_correlation_shift(corr_data, scaling):
    d, t, x1, x2, dly = corr_data
    c, lag = correlation(x1, x2, scaling=scaling)
    imax = np.argmax(c)
    assert -lag[imax]*(t[1]-t[0]) == approx(dly)  # -lag * dt == t_shift


def test_correlation_illegal(corr_data):
    d, t, x1, x2, dly = corr_data
    with raises(ValueError):
        c, lag = correlation(x1, x2, scaling="illegal")


@pytest.mark.parametrize("scaling", ['coeff', 'none'])
def test_fftcorrelate_identity(corr_data, scaling):
    dt, t, x1, x2, dly = corr_data
    c, l = _fftcorrelate(x1, x1, scaling)
    cr, lag = correlation(x1, x1, scaling=scaling)
    assert np.allclose(c, cr)
    assert np.allclose(l, lag)


@pytest.mark.parametrize("scaling", ['coeff', 'none'])
def test_fftcorrelate_shift(corr_data, scaling):
    d, t, x1, x2, dly = corr_data
    c, l = _fftcorrelate(x1, x2, scaling)
    cr, lag = correlation(x1, x2, scaling=scaling)
    assert np.allclose(c, cr)
    assert np.allclose(l, lag)


def test_averaged_psd():
    # Create a simple sinusoidal signal
    fs = 10  # sample rate
    T = 5.0    # seconds
    t = np.linspace(0, T, int(T*fs), endpoint=False)  # time variable
    x = 0.5*np.sin(2*np.pi*2*t)  # +0.1*np.sin(2*np.pi*10*t)
    input_tsig = pd.Series(x, index=pd.TimedeltaIndex(t, 's'))

    # Calculate the averaged PSD using the function
    navgs = 5
    overlap = 0.5
    window = 'hamming'
    detrend = 'linear'
    scaling = 'density'
    psd = averaged_psd(input_tsig, navgs, overlap, window, detrend, scaling)

    # Calculate the PSD directly using scipy.signal.welch
    freqs, psd_direct = signal.welch(x, fs, window, nperseg=len(x)//navgs,
                                     noverlap=int(overlap*len(x)//navgs),
                                     detrend=detrend, scaling=scaling)

    # Check that the PSDs match
    assert np.allclose(psd, psd_direct, atol=1e-5)


def test_averaged_tf():
    # Create a simple sinusoidal signal
    fs = 10  # sample rate
    T = 5.0 # seconds
    t = np.linspace(0, T, int(T*fs), endpoint=False)  # time variable
    x = 0.5*np.sin(2*np.pi*2*t)  # input signal
    y = 0.5*np.sin(2*np.pi*2*t + np.pi/4)  # output signal with phase shift
    x_tsig = pd.Series(x, index=pd.TimedeltaIndex(t, 's'))
    y_tsig = pd.Series(y, index=pd.TimedeltaIndex(t, 's'))

    # Calculate the averaged transfer function using the function
    navgs = 5
    overlap = 0.5
    window = 'hamming'
    detrend = 'linear'
    tf = averaged_tf(x_tsig, y_tsig, navgs, overlap, window, detrend)

    # Calculate the transfer function directly using scipy csd and welch
    freqs, Pxy = signal.csd(x, y, fs, window, nperseg=len(x)//navgs,
                            noverlap=int(overlap*len(x)//navgs),
                            detrend=detrend)
    freqs, Pxx = signal.welch(x, fs, window, nperseg=len(x)//navgs,
                              noverlap=int(overlap*len(x)//navgs),
                              detrend=detrend)
    tf_direct = Pxy / Pxx

    # Check that the transfer functions match
    assert np.allclose(tf['tf'], tf_direct, atol=1e-5)


def test_interp_tf():
    # Create a simple transfer function
    freqs = np.linspace(0, 1, 100)
    mag = np.abs(np.sin(2*np.pi*freqs))
    phase = np.unwrap(np.sin(2*np.pi*freqs)+freqs*6)
    tf = pd.Series(mag * np.exp(1j * phase), index=freqs)

    # Calculate the interpolated transfer function using the function
    new_freq = np.linspace(0, 1, 200)
    interp_tf_result = interp_tf(new_freq, tf)

    # Calculate the interpolated transfer function directly
    interp_mag_func = interp1d(freqs, mag, axis=0)
    interp_phase_func = interp1d(freqs, phase, axis=0)
    interp_mag_direct = interp_mag_func(new_freq)
    interp_phase_direct = interp_phase_func(new_freq)
    interp_tf_direct = interp_mag_direct * np.exp(1j * interp_phase_direct)

    # Check that the interpolated transfer functions match
    assert np.allclose(interp_tf_result, interp_tf_direct, atol=1e-5)


@pytest.mark.parametrize("delay", [-5.0, -2.5, -0.5, 0.0, 0.5, 2.5, 5.0])
def test_tf_delay(delay):
    # delay = 0.05  # delay in seconds
    fs = 100  # sample rate
    T = 500.0  # total seconds
    np.random.seed(2023)
    t = np.linspace(0, T, int(T*fs), endpoint=False)  # time variable
    noise = np.random.random(len(t))/5  # add broadband noise
    x = 0.5*np.sin(2*np.pi*2*t) + noise  # noisy signal
    y = np.roll(x, int(delay*fs))  # Explicitly delay the signal
    x_tsig = pd.Series(x, index=pd.TimedeltaIndex(t, 's'))
    y_tsig = pd.Series(y, index=pd.TimedeltaIndex(t, 's'))

    # Calculate the averaged transfer function using the function
    navgs = 5
    overlap = 0.5
    window = 'hamming'
    detrend = 'linear'
    tf = averaged_tf(x_tsig, y_tsig, navgs, overlap, window, detrend)

    # Calculate the delay using the function
    delay_result, _ = tf_delay(tf, coh_limit=0.6, freq_limit=1, method='fit')

    # Check that the delay matches the expected delay
    assert np.isclose(delay_result, delay, atol=1e-2)


@pytest.mark.parametrize("delay", [-5.0, -2.5, -0.5, 0.0, 0.5, 2.5, 5.0])
def test_xcorr_delay(delay):
    # delay = -0.05  # delay in seconds
    fs = 100  # sample rate
    T = 500.0  # total seconds
    np.random.seed(2023)
    t = np.linspace(0, T, int(T*fs), endpoint=False)  # time variable
    noise = np.random.random(len(t))/5  # add broadband noise
    x = 0.5*np.sin(2*np.pi*0.01*t) + noise  # noisy signal
    y = np.roll(x, int(delay*fs))  # Explicitly delay the signal
    x_tsig = pd.Series(x, index=pd.TimedeltaIndex(t, 's'))
    y_tsig = pd.Series(y, index=pd.TimedeltaIndex(t, 's'))

    # Calculate the averaged transfer function using the function
    delay_result, _ = xcorr_delay(x_tsig, y_tsig, "coeff")

    # Check that the delay matches the expected delay
    assert np.isclose(delay_result, delay, atol=1e-2)


@pytest.mark.parametrize("delay", [-5.0, -2.5, -0.5, 0.0, 0.5, 2.5, 5.0])
def test_apply_delay(delay):
    # Create a simple transfer function
    freqs = np.linspace(0.01, 1, 100)  # frequencies
    phase = -2 * np.pi * freqs * delay  # phase delay
    mag = np.ones_like(freqs)  # magnitude
    coh = np.ones_like(freqs)  # coherence
    tf = pd.DataFrame({'tf': mag * np.exp(1j * phase), 'coh': coh}, index=freqs)

    # Apply the delay using the function
    tf_delayed = apply_delay(tf, delay)

    # Calculate the expected delayed transfer function
    tf_expected = tf.copy()
    tf_expected['tf'] = tf['tf'] * np.exp(2 * np.pi * 1j * tf.index * -delay)

    # Check that the delayed transfer function matches the expected one
    assert np.allclose(tf_delayed['tf'], tf_expected['tf'], atol=1e-5)


def test_averaged_psd_multi():
    np.random.seed(2023)
    # Create a simple sinusoidal signal
    fs = 10  # sample rate
    T = 5.0    # seconds
    t = np.linspace(0, T, int(T*fs), endpoint=False)  # time variable

    x1 = 0.5 * np.sin(2 * np.pi * 2 * t) + np.random.random(len(t))
    x2 = 0.5 * np.sin(2 * np.pi * 2 * t) + np.random.random(len(t))
    x3 = 0.5 * np.sin(2 * np.pi * 2 * t) + np.random.random(len(t))
    x4 = 0.5 * np.sin(2 * np.pi * 2 * t) + np.random.random(len(t))

    xs = [x1, x2, x3, x4]
    tsigs = [pd.Series(x, index=pd.TimedeltaIndex(t, 's')) for x in xs]
    navgs = 5
    overlap = 0.5
    window = 'hamming'
    detrend = 'linear'
    scaling = 'density'
    psds = [averaged_psd(tsig, navgs, overlap, window, detrend, scaling) for tsig in tsigs]

    # Calculate the PSD directly using scipy.signal.welch
    freqs, psd_direct = signal.welch(xs, fs, window, nperseg=len(x1)//navgs,
                                     noverlap=int(overlap*len(x1)//navgs),
                                     detrend=detrend, scaling=scaling)

    # Check that the PSDs match
    assert np.allclose(psds, psd_direct, atol=1e-5)


def test_averaged_tf_multi():
    np.random.seed(2023)
    # Create a simple sinusoidal signal
    fs = 500  # sample rate
    T = 1000.0    # seconds
    t = np.linspace(0, T, int(T*fs), endpoint=False)  # time variable
    delay = 3

    x = 0.5 * np.sin(2 * np.pi * 2 * t) + np.random.random(len(t))

    y1 = np.roll(x, int(delay*fs))
    y2 = np.roll(x, int(2*delay * fs))
    y3 = np.roll(x, int(3*delay * fs))
    y4 = np.roll(x, int(4*delay * fs))

    ys = [y1, y2, y3, y4]

    x_tsig = pd.Series(x, index=pd.TimedeltaIndex(t, 's'))
    ysigs = [pd.Series(y, index=pd.TimedeltaIndex(t, 's')) for y in ys]
    navgs = 5
    overlap = 0.5
    window = 'hamming'
    detrend = 'linear'
    tfs = [averaged_tf(x_tsig, ysig, navgs, overlap, window, detrend)['tf'] for ysig in ysigs]

    ##########################
    # Seems like this works as long as you do a list of signals for the y, rather than a dataframe of signals.
    # If you have a dataframe of signals, you can convert by dff.values.T.tolist()
    ##########################
    freqs, somePxy = signal.csd([x_tsig, ysigs[0], ysigs[1], ysigs[2]], ysigs, fs, window,
               nperseg=len(x) // navgs,
               noverlap=int(overlap * len(x) // navgs),
               detrend=detrend)
    freqs, somePxx = signal.welch([x_tsig, ysigs[0], ysigs[1], ysigs[2]], fs, window, nperseg=len(x)//navgs,
                              noverlap=int(overlap*len(x)//navgs),
                              detrend=detrend)
    sometf_direct = somePxy / somePxx
    # import matplotlib.pyplot as plt
    # plt.plot(freqs, np.abs(sometf_direct).T)
    # plt.figure()
    # plt.plot(freqs, np.angle(sometf_direct).T)
    # plt.show()


    # Calculate the PSD directly using scipy.signal.welch
    # Calculate the transfer function directly using scipy csd and welch
    freqs, Pxy = signal.csd(x_tsig, ysigs, fs, window, nperseg=len(x)//navgs,
                            noverlap=int(overlap*len(x)//navgs),
                            detrend=detrend)
    freqs, Pxx = signal.welch(x_tsig, fs, window, nperseg=len(x)//navgs,
                              noverlap=int(overlap*len(x)//navgs),
                              detrend=detrend)
    tf_direct = Pxy / Pxx

    # Check that the PSDs match
    assert np.allclose(tfs, tf_direct, atol=1e-5)


@pytest.fixture(params=['loop', 'fft'])
def compute_delays_modes(request):
    return request.param


@pytest.mark.parametrize("delay", [-200, -50.0, -25, -5, 0.0, 5, 25, 50, 200])
def test_compute_delays(delay, compute_delays_modes):
    np.random.seed(2023)
    # Create a simple sinusoidal signal
    fs = 100  # sample rate
    T = 500.0  # seconds
    t = np.linspace(0, T, int(T * fs), endpoint=False)  # time variable
    # delay = 0.25

    x = 0.5 * np.sin(2 * np.pi * 2 * t) + 5*np.random.random(len(t))

    ys = [x]
    delay_ins = [0]
    n = 50
    for i in range(n):
        deli = delay / n * i
        yi = np.roll(x, int(deli * fs))
        ys.append(yi)
        delay_ins.append(deli)

    x_tsig = pd.Series(x, index=pd.TimedeltaIndex(t, 's'))
    ysigs = [pd.Series(y, index=pd.TimedeltaIndex(t, 's')) for y in ys]
    xsigs = [x_tsig for y in ys]
    xsigs = pd.DataFrame(np.array(xsigs).T, index=x_tsig.index)

    delays, _ = compute_delays(xsigs, ysigs, compute_delays_modes)

    assert np.allclose(delays, delay_ins)
