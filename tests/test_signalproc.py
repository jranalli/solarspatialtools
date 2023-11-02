import pytest
import pandas as pd
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d
from solartoolbox.signalproc import averaged_psd, averaged_tf, interp_tf, tf_delay, xcorr_delay, apply_delay


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