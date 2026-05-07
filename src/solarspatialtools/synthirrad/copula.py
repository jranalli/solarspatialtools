import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from solarspatialtools import spatial
from scipy.stats import norm

# [1] Widen, J. and Munkhammar, J., "Spatio-Temporal Downscaling of Hourly Solar irradiance Data Using Gaussian Copulas," 2019 IEEE 46th Photovoltaic Specialists Conference (PVSC), Chicago, IL, USA, 2019, pp. 3172-3178, doi: 10.1109/PVSC40753.2019.8980922.

def _sigmoid(x, a, c):
    """
    Generate a sigmoid membership function. Equation 11 from the paper.

    Parameters
    ----------
    x : np.array
        The input variable, typically in the range [0, 1].
    a : float
        The slope parameter, controlling how steep the transition is.
    c : float
        The center parameter, controlling the center of the transition.

    Returns
    -------
    np.array
        The output of the sigmoid function, in the range [0, 1].
    """
    return 1 / (1 + np.exp(-a * (x - c)))


def _gaussianMixtureDistribution(csi, meanCSI, params, debug=False):
    """Build a 1D Gaussian mixture and evaluate its PDF at csi values."""
    comp_cloud = 1 - params['comp'][0]*_sigmoid(meanCSI, params['comp'][1],params['comp'][2])
    comp_clear = 1 - comp_cloud
    mean_clear = params['mean'][0] * comp_cloud * meanCSI + params['mean'][1] * (1 - comp_cloud) * (meanCSI - params['mean'][2])
    mean_cloud = (meanCSI - comp_clear * mean_clear) / comp_cloud
    sdev_clear = params['sdevClear'][0] * (1 - _sigmoid(meanCSI, params['sdevClear'][1],params['sdevClear'][2]))
    sdev_cloud = params['sdevCloud'][0] * _sigmoid(meanCSI, params['sdevCloud'][1], params['sdevCloud'][2])

    mu = [mean_cloud, mean_clear]
    p = [comp_cloud, comp_clear]

    sigma = [sdev_cloud**2, sdev_clear**2]

    mu = np.array(mu).reshape(-1, 1)
    sigma = np.array(sigma).reshape(-1, 1, 1)   # 1D full covariance form
    p = np.array(p)

    gm = GaussianMixture(n_components=len(p), covariance_type="full")
    gm.weights_ = p
    gm.means_ = mu
    gm.covariances_ = sigma
    gm.precisions_cholesky_ = 1 / np.sqrt(sigma)

    csi_input = np.asarray(csi)
    csi_samples = np.atleast_1d(csi_input).reshape(-1, 1)
    pdf_val = np.exp(gm.score_samples(csi_samples))
    if csi_input.ndim == 0:
        pdf_val = float(pdf_val[0])

    if debug:
        import matplotlib.pyplot as plt
        plt.plot(csi,pdf_val)
        plt.show()

    return gm, pdf_val

def _exponential_decay_parameter(mean_csi, p):
    k = p * mean_csi * (1-mean_csi)
    if k < 10**-5:
        k = 10**-5
    return k

def _space_time_copula(N, sites, times, csi, cdf, funchand, p, cs, cd):
    xref, yref = spatial.latlon2lcs(sites[0], sites[1], sites[0][0], sites[1][0])
    xref, yref = spatial.lla2flat(sites[0], sites[1], sites[0][0], sites[1][0])

    t0 = times[0]
    dur = (times-t0).total_seconds().values

    # Build a time-dependent drift term and broadcast against site coordinates.
    x_drift = np.asarray(cs) * dur * np.sin(np.asarray(cd))
    y_drift = np.asarray(cs) * dur * np.cos(np.asarray(cd))

    X = np.asarray(xref)[:, None] - np.asarray(x_drift)[None, :]
    Y = np.asarray(yref)[:, None] - np.asarray(y_drift)[None, :]
    X = X.T
    Y = Y.T

    x = np.asarray(X).reshape(-1, order='F')
    y = np.asarray(Y).reshape(-1, order='F')
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    D = np.sqrt(dx**2 + dy**2)

    C = funchand(p, D)

    U = copularnd_gaussian(C, N)

    CSI = inverse_sample(csi, cdf, U)
    CSI = CSI.reshape(N, times.shape[0], sites[0].shape[0])
    return CSI

def inverse_sample(x, cdf, r):
    x = np.asarray(x, dtype=float).reshape(-1)
    cdf = np.asarray(cdf, dtype=float).reshape(-1)

    if x.size == 0 or cdf.size == 0:
        raise ValueError("x and cdf must be non-empty.")
    if x.size != cdf.size:
        raise ValueError("x and cdf must have the same length.")

    # MATLAB-style unique(cdf) with indices used to subset x.
    Fxu, inds = np.unique(cdf, return_index=True)

    if Fxu.size == 1:
        Fxu = np.array([-99999.0, 99999.0])
        xu = np.array([-2.0, 2.0])
    else:
        xu = x[inds].astype(float, copy=True)
        Fxu = Fxu.astype(float, copy=True)
        xu = np.concatenate([[-2.0], xu, [2.0]])
        Fxu = np.concatenate([[-99999.0], Fxu, [99999.0]])

    r_arr = np.asarray(r, dtype=float)
    s = np.interp(r_arr.reshape(-1), Fxu, xu)

    if r_arr.ndim == 0:
        return float(s[0])
    return s.reshape(r_arr.shape)


def copularnd_gaussian(C, N, random_state=None):

    rng = np.random.default_rng(random_state)
    d = C.shape[0]
    z = rng.multivariate_normal(mean=np.zeros(d), cov=C, size=N)
    U = norm.cdf(z)
    return U

if __name__ == "__main__":
    param = {
        'comp': [
            0.8051,
            7.3605,
            0.7092
        ],
        'mean': [
            2.2928,
            1.0801,
            0.4532
        ],
        'sdevClear': [
            0.3512,
            4.8414,
            0.6442
        ],
        'sdevCloud': [
            0.1997,
            5.0919,
            0.3863
        ],
        'corr_quadr': 0.0043
    }

    # Cloud speed and direction in radians
    cs = np.array([5, 5, 5, 5, 5, 5])
    cd = np.array([90, 90, 90, 90, 90, 90]) * 2 * np.pi / 360

    # Hourly clearsky
    hcsi = np.array([0.52,0.71,0.5,0.84,0.63,0.11])

    # lat
    lat = np.array([21.31236, 21.31303, 21.32357])
    lon = np.array([-158.08463, -158.08505, -158.08424])

    times = pd.date_range(start='2024-01-01 00:00:00', end='2024-01-01 00:59:59', freq='15s')

    noneg = True
    scale = True

    c = []

    # Loop
    for i in range(len(hcsi)):

        csi = np.arange(-2, 2, 0.01)
        mean_csi = hcsi[i]

        gm, pdf = _gaussianMixtureDistribution(csi, mean_csi, param, debug=False)
        cdf = np.cumsum(pdf) * (csi[1] - csi[0])

        # Skip matplotlib for now
        # import matplotlib.pyplot as plt
        # plt.plot(csi, cdf)
        # plt.show()

        fun = lambda p, d: np.exp(-p * d)
        p = _exponential_decay_parameter(mean_csi, param['corr_quadr'])

        M = _space_time_copula(1, (lat, lon), times, csi, cdf, fun, p, cs[i], cd[i])

        cm = M[0,:,:]

        if noneg:
            cm[cm<0] = 0

        if scale:
            m = np.mean(cm)
            scaling_factor = hcsi[i] / m
            cm *= scaling_factor

        # Collect each segment; we concatenate into one long time series after the loop.
        c.append(cm)

    c = np.concatenate(c, axis=0)

    import matplotlib.pyplot as plt

    # Build a blocky hourly reference aligned with the high-resolution output.
    n_per_hour = times.shape[0]
    hcsi_block = np.repeat(hcsi, n_per_hour)

    plt.plot(c, alpha=0.8)
    plt.step(np.arange(hcsi_block.size), hcsi_block, where='post', color='k', linewidth=2, label='hcsi (hourly step)')
    plt.legend()

    plt.figure()
    plt.hist(c)
    plt.show()
