import numpy as np
import pandas as pd
from optype._core import _has
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

from solarspatialtools import spatial
from scipy.stats import norm, multivariate_normal

# [1] Widen, J. and Munkhammar, J., "Spatio-Temporal Downscaling of Hourly
# Solar irradiance Data Using Gaussian Copulas," 2019 IEEE 46th Photovoltaic
# Specialists Conference (PVSC), Chicago, IL, USA, 2019, pp. 3172-3178,
# doi: 10.1109/PVSC40753.2019.8980922.

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


def _gmdistribution(x, mu, vars, p, debug=False):
    """
    Wrap sklearn's GaussianMixture to behave more like the matlab
    gmdistribution function.

    Parameters
    ----------
    x : np.array
        The x values at which to compute the probability of the distribution
    mu : np.array
        The means for each component, shape (n_components,)
    vars : np.array
        The variances for each component, shape (n_components,)
    p : np.array
        The weights for each component, shape (n_components,)
    debug : bool, optional
        If True, plot the pdf and cdf for visual inspection. Default is False.

    Returns
    -------
    np.array
        The pdf values at the input x, shape (len(x),)
    np.array
        The cdf values at the input x, shape (len(x),), scaled 0-1
    """

    # Reshape inputs to the correct form
    x_inp = np.atleast_1d(np.asarray(x)).reshape(-1, 1)
    mu = np.array(mu).reshape(-1, 1)  # means
    vars = np.array(vars).reshape(-1, 1, 1)  # stdevs
    p = np.array(p)  # weights

    # Apply to create the GaussianMixture
    gm = GaussianMixture(n_components=len(p), covariance_type="full")
    gm.weights_ = p
    gm.means_ = mu
    gm.covariances_ = vars
    gm.precisions_cholesky_ = _compute_precision_cholesky(vars, 'full')

    # use score_samples to predict the log-likelihood of each input.
    pdf_val = np.exp(gm.score_samples(x_inp))
    cdf_val = np.cumsum(pdf_val) / np.max(np.cumsum(pdf_val))  # scale 0-1

    # Realign output dimensions
    if np.array(x).ndim == 0:
        pdf_val = float(pdf_val[0])
        cdf_val = float(cdf_val[0])

    if debug:
        import matplotlib.pyplot as plt
        plt.plot(x, pdf_val)
        plt.figure()
        plt.plot(x, cdf_val)
        plt.show()

    return pdf_val, cdf_val


def _inverse_sample(x, cdf, r):
    """
    Perform inverse sampling using linear interpolation. Given a CDF defined
    by (x, cdf), and a random value r in [0, 1], return the corresponding x
    value whose CDF is equal to r.

    Parameters
    ----------
    x : np.array
        The x values corresponding to the CDF, shape (n,) - in this case CSIs

    cdf : np.array
        The CDF values, shape (n,) - probability of CSI value less than given
        CSI.

    r : np.array or float
        The random value(s) in [0, 1] for which to compute the inverse sample,
        shape (m,) or scalar i.e. r=0.5 finds what CSI has probability 50%?

    Returns
    -------
    np.array or float
        The x-values whose CDF is equal to r.
    """

    x = np.asarray(x)
    cdf = np.asarray(cdf)

    # MATLAB-style unique(cdf) with indices used to subset x.
    Fxu, inds = np.unique(cdf, return_index=True)
    xu = x[inds].astype(float, copy=True)

    # Pad on either end to ensure that small overflows still have meaning
    xu = np.concatenate([[-2.0], xu, [2.0]])  # X represents CSI
    Fxu = np.concatenate([[-0.0], Fxu, [1.0]])  # Fxu Represents the CDF

    # Perform interpolation
    r_arr = np.asarray(r, dtype=float)
    s = np.interp(r_arr.reshape(-1), Fxu, xu)

    # Retain dtype of input
    if r_arr.ndim == 0:
        return float(s[0])
    return s.reshape(r_arr.shape)


def _solar_gmm(csi, meanCSI, params, debug=False):
    """
    Compute the parameters of a gaussian mixture model of clear and cloudy
    moments. Return the PDF and CDF of the model.

    Parameters
    ----------
    csi : np.array
        The CSI values at which to compute the PDF and CDF, shape (n,)
    meanCSI : float
        The mean CSI for the hour, used to compute the parameters of the GMM.
    params : dict
        The parameters of the model, containing at least:
        - 'comp': [a, c] parameters for the sigmoid function that determines
            the cloudy component weight based on meanCSI.
        - 'mean': [m_cloud, m_clear, c] parameters for computing the means of
            the cloud and clear components based on meanCSI and the cloudy
            component weight.
        - 'sdevClear': [sdev_clear_max, a, c] parameters for computing the
            standard deviation of the clear component based on meanCSI.
        - 'sdevCloud': [sdev_cloud_max, a, c] parameters for computing the
            standard deviation of the cloud component based on meanCSI.

    Returns
    -------
    np.array
        The PDF values at the input CSI, shape (n,)
    np.array
        The CDF values at the input CSI, shape (n,), scaled 0-1
    """
    comp_cloud = (1 -
                  params['comp'][0] *
                  _sigmoid(meanCSI, params['comp'][1], params['comp'][2]))
    comp_clear = 1 - comp_cloud
    mean_clear = (params['mean'][0] * comp_cloud * meanCSI +
                  params['mean'][1] * (1 - comp_cloud) * (meanCSI - params['mean'][2]))
    mean_cloud = (meanCSI - comp_clear * mean_clear) / comp_cloud
    sdev_clear = params['sdevClear'][0] * (1 - _sigmoid(meanCSI, params['sdevClear'][1],params['sdevClear'][2]))
    sdev_cloud = params['sdevCloud'][0] * _sigmoid(meanCSI, params['sdevCloud'][1], params['sdevCloud'][2])

    mu = [mean_cloud, mean_clear]
    p = [comp_cloud, comp_clear]
    vars = [sdev_cloud**2, sdev_clear**2]

    pdf_val, cdf_val = _gmdistribution(csi, mu, vars, p, debug)

    return pdf_val, cdf_val


def _exponential_decay_parameter(K, p):
    """
    Compute the exponential decay parameter for the copula based on the mean
    CSI and the quadratic correlation parameter.

    Parameters
    ----------
    K : float
        The mean CSI for the hour, used to compute the exponential
        decay parameter.

    p : float
        The quadratic correlation parameter, controlling how quickly the
        correlation decays with distance.

    Returns
    -------
    float
        The exponential decay parameter for the copula, which controls how
        quickly the correlation decays with distance. Higher values of p lead
        to faster decay, while higher values of K (mean CSI) lead to slower
        decay.
    """
    k = p * K * (1-K)
    if k < 10**-5:
        k = 10**-5
    return k

def _space_time_copula(N, Epos, Npos, times, csi, cdf, p, cs, cd, seed=None):
    """
    Generate the synthetic CSI for a position field using a spacetime copula.

    Parameters
    ----------
    N : int
        The number of versions of the samples to generate.
    Epos : float
        The eastward positions of the sites, shape (n_sites,)
    Npos : float
    The northward positions of the sites, shape (n_sites,)
    times : pd.DatetimeIndex
    A DatetimeIndex of shape (n_times,) representing the time points for
        which to compute distances.
    csi : float
    The CSI values corresponding to the CDF, shape (n_csi,). This is used for inverse sampling after generating the copula samples.
    cdf : float
    The CDF values corresponding to the CDF, shape (n_cdf,). This is used for inverse sampling after generating the copula samples.
    p : float
    The quadratic correlation parameter, controlling how quickly the correlation decays
    with distance.
    cs : float
    The cloud speed in the same units as Epos and Npos per unit time. This is used to compute the spacetime distances.
    cd : float
    The cloud direction in radians, where 0 is eastward and pi/2 is northward. This is used to compute the spacetime distances.
    seed : int or None
    Seed for the random number generator.

    Results
    -------
    np.array
        The synthetic CSI values generated by the copula, shape (N, n_times, n_sites).
    """

    D = spatial.spacetime_distances(Epos, Npos, times, cs, cd)

    funchand = lambda p, d: np.exp(-p * d)
    C = funchand(p, D)

    # Generate uniform samples from the copula with appropriate correlation
    U = _copularnd_gaussian(C, N, seed)

    # Invert uniform samples based on the original CDF back to the CSI
    # Reshape to match the individual time series desired for each site.
    CSI = _inverse_sample(csi, cdf, U)
    CSI = CSI.reshape(N, Epos.shape[0], times.shape[0]).transpose([0,2,1])

    return CSI


def _copularnd_gaussian(C, N, random_state=None):
    """
    Generate the copula samples using a Gaussian copula. This involves
    generating multivariate normal samples with covariance C, and then applying
    the standard normal CDF to transform them into uniform samples. They can
    be converted in a second step.

    Parameters
    ----------
    C : float
        The covariance matrix for the Gaussian copula, shape (d, d), where d is
        the number of sites times the number of spacetime elements. All
        spacetime elements are concatenated.

    N : int
        The number of versions of the samples to generate.

    random_state : int or None
        Seed for the random number generator.

    Results
    -------
    np.array
        The uniform samples generated by the Gaussian copula, shape is
        (N, n_sites * n_times).
    """
    rng = np.random.default_rng(random_state)
    d = C.shape[0]
    z = rng.multivariate_normal(mean=np.zeros(d), cov=C, size=N)
    U = norm.cdf(z)
    return U

def __basic_testing():
    # Generate the shape of a gaussian mixture model
    x0 = np.arange(-2, 2, 0.01)
    mu0 = np.array([0.25, 0.75])
    sigma0 = np.array([0.01,0.01])
    p0 = np.array([1,4])

    # Generate the mixture distribution pdf and cdf
    _, pdf, cdf = _gmdistribution(x0, mu0, sigma0, p0)

    # Test Inverse Sampling on the CDF
    yi = [0.6, 0.9]
    xi = _inverse_sample(x0, cdf, yi)

    print(xi)

    # Plot the output
    import matplotlib.pyplot as plt
    plt.plot(x0, pdf)
    plt.figure()
    plt.plot(x0, cdf)
    plt.plot(xi, yi, 'x')
    plt.show()

    ### CONFIDENT IN THIS OUTPUT, IT WORKS

def __lla2flat_testing():
    lat = np.array([21.31236, 21.31303, 21.32357])
    lon = np.array([-158.08463, -158.08505, -158.08424])

    print(spatial.lla2flat(lat, lon, lat[0], lon[0], "matlab"))
    print(spatial.lla2flat(lat, lon, lat[0], lon[0], "mercator"))


def __matlab_compare():


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
    cd = np.array([0, 0, 0, 0, 0, 0]) * 2 * np.pi / 360

    # Hourly clearsky
    hcsi = np.array([0.52,0.71,0.5,0.84,0.63,0.11])

    # lat
    lat = np.array([21.31236, 21.31303, 21.32357])
    lon = np.array([-158.08463, -158.08505, -158.08424])
    # Epos, Npos = spatial.latlon2lcs(lat, lon, lat[0], lon[0])
    Epos, Npos = spatial.lla2flat(lat, lon, lat[0], lon[0], method="tmerc")

    times = pd.date_range(start='2024-01-01 00:00:00', end='2024-01-01 00:59:59', freq='15s')

    noneg = True
    scale = True

    c = []

    # Loop
    for i in range(len(hcsi)):

        csi = np.arange(-2, 2, 0.01)
        mean_csi = hcsi[i]

        pdf, cdf = _solar_gmm(csi, mean_csi, param, debug=False)

        p = _exponential_decay_parameter(mean_csi, param['corr_quadr'])

        M = _space_time_copula(1, Epos, Npos, times, csi, cdf, p, cs[i], cd[i], 42)

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


def __spatial_compare():

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
    cs = 1
    cd = 0

    # Hourly clearsky
    hcsi = 0.52

    Epos = np.array([ 0, 500,  0, 500, 0])
    Npos = np.array([ 0,   0, 20,  20, 200])

    times = pd.date_range(start='2024-01-01 00:00:00', end='2024-01-01 00:59:59', freq='15s')

    csi = np.arange(-2, 2, 0.01)
    mean_csi = hcsi

    pdf, cdf = _solar_gmm(csi, mean_csi, param, debug=False)

    p = _exponential_decay_parameter(mean_csi, param['corr_quadr'])

    M = _space_time_copula(1, Epos, Npos, times, csi, cdf, p, cs, cd, 42)

    c = M[0,:,:]

    import matplotlib.pyplot as plt
    plt.plot(times, c, alpha=0.8)
    plt.legend([1,2,3,4,5])
    plt.show()



if __name__ == "__main__":
    # __basic_testing()  # Validated - confident in what this is doing
    # __lla2flat_testing()  # Mercator and lla2flat algorithm are accurate to centimeters for the latlon given
    __matlab_compare()
    __spatial_compare()