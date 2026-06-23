import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.mixture._gaussian_mixture import _compute_precision_cholesky

from solarspatialtools import spatial
from scipy.stats import norm

# [1] Widen, J. and Munkhammar, J., "Spatio-Temporal Downscaling of Hourly
# Solar irradiance Data Using Gaussian Copulas," 2019 IEEE 46th Photovoltaic
# Specialists Conference (PVSC), Chicago, IL, USA, 2019, pp. 3172-3178,
# doi: https://dx.doi.org/10.1109/PVSC40753.2019.8980922.

# Default fitting parameters as given by the paper in Table I
DEFAULT_PARAMS = {
    'comp': [0.8051, 7.3605, 0.7092],
    'mean': [2.2928, 1.0801, 0.4532],
    'sdevClear': [0.3512, 4.8414, 0.6442],
    'sdevCloud': [0.1997, 5.0919, 0.3863],
    'corr_quadr': 0.0043
}


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


def _gmdistribution(x, mu, variances, p, debug=False):
    """
    Wrap sklearn's GaussianMixture to behave more like the matlab
    gmdistribution function.

    Parameters
    ----------
    x : np.array
        The x values at which to compute the probability of the distribution

    mu : np.array
        The means for each component, shape (n_components,)

    variances : np.array
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
    variances = np.array(variances).reshape(-1, 1, 1)  # stdevs
    p = np.array(p)  # weights

    # Apply to create the GaussianMixture
    gm = GaussianMixture(n_components=len(p), covariance_type="full")
    gm.weights_ = p
    gm.means_ = mu
    gm.covariances_ = variances
    gm.precisions_cholesky_ = _compute_precision_cholesky(variances, 'full')

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


def _process_params(meanCSI, params):
    """
    Convert the params dict into the necessary values for subsequent parts of
    the model.

    This represents equations 9-14 of the paper [1].

    Parameters
    ----------
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
    list
        The means (mu1, mu2) for the cloud and clear gmm components, shape (2,)

    list
        The weights (w1, w2) for the cloud and clear gmm components, shape (2,)

    list
        The variances (sig1**2, sig2**2) for the cloud and clear gmm
        components, shape (2,)

    list
        The exponential decay parameter for the copula, k' in the paper

    References
    ----------
    [1] Widen, J. and Munkhammar, J., "Spatio-Temporal Downscaling of Hourly
    Solar irradiance Data Using Gaussian Copulas," 2019 IEEE 46th Photovoltaic
    Specialists Conference (PVSC), Chicago, IL, USA, 2019, pp. 3172-3178,
    doi: 10.1109/PVSC40753.2019.8980922.
    """
    # Weights - 1 cloudy, 2 clear
    w1 = (1 - params['comp'][0] * _sigmoid(meanCSI, params['comp'][1], params['comp'][2]))
    w2 = 1 - w1

    # GMM Means - 1 cloudy, 2 clear (equations 12 & 13 in the paper)
    # Note that equation 12 is actually a mistake - the fit params provided
    # by the paper are actually the fit params for the clear mean, not cloudy
    mu2 = (params['mean'][0] * w1 * meanCSI + params['mean'][1] * (1 - w1) * (meanCSI - params['mean'][2]))
    mu1 = (meanCSI - w2 * mu2) / w1

    # GMM Std devs - Eq 14
    sig2 = params['sdevClear'][0] * (1 - _sigmoid(meanCSI, params['sdevClear'][1],params['sdevClear'][2]))
    sig1 = params['sdevCloud'][0] * _sigmoid(meanCSI, params['sdevCloud'][1], params['sdevCloud'][2])

    mu = [mu1, mu2]
    w = [w1, w2]
    variances = [sig1**2, sig2**2]
    k = _exponential_decay_parameter(meanCSI, params['corr_quadr'])
    return mu, w, variances, k


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

    float
        The exponential decay parameter for the copula, which controls how
        quickly the correlation decays with distance.
    """
    mu, w, variances, k = _process_params(meanCSI, params)

    pdf_val, cdf_val = _gmdistribution(csi, mu, variances, w, debug)

    return pdf_val, cdf_val, k


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
    try:
        if k < 10**-5:
            k = 10**-5
    except ValueError:
        k[k < 10**-5] = 10**-5
    return k


def _space_time_copula(n, spacetime_dist, cdf_x, cdf, p, seed=None):
    """
    Generate the synthetic CSI for a position field using a spacetime copula.

    Parameters
    ----------
    n : int
        The number of versions of the samples to generate.

    spacetime_dist : np.array
        spacetime distances, output of spatial.spacetime_distance

    cdf_x : np.array
        The CSI values corresponding to the CDF, shape (n_cdf,). This is used
        for inverse sampling after generating the copula samples.

    cdf : np.array
        The CDF values corresponding to the CDF, shape (n_cdf,). This is used
        for inverse sampling after generating the copula samples.

    p : float
        The quadratic correlation parameter, controlling how quickly the
        correlation decays with distance.

    seed : int or None
        Seed for the random number generator.

    Returns
    -------
    np.array
        The synthetic CSI values generated by the copula, shape
        (N, spacetime_dist.size[0]). Since spacetime_dist concatenates the
        multiple sites, recommend conditioning output with similar to:

        output = output.reshape(n, n_sides, n_times).transpose([0, 2, 1])
    """

    # Function for converting the distances to covariance between sites
    cov_matrix = np.exp(-p * spacetime_dist)

    # Generate uniform samples from the copula with appropriate correlation
    unif_samples = _copularnd_gaussian(n, cov_matrix, seed)

    # Invert uniform samples based on the original CDF back to the CSI
    # Reshape to match the individual time series desired for each site.
    gen_csi = _inverse_sample(cdf_x, cdf, unif_samples)

    return gen_csi


def _copularnd_gaussian(n, cov_matrix, random_state=None):
    """
    Generate the copula samples using a Gaussian copula. This involves
    generating multivariate normal samples with covariance C, and then applying
    the standard normal CDF to transform them into uniform samples. They can
    be converted in a second step.

    Parameters
    ----------
    n : int
        The number of versions of the samples to generate.

    cov_matrix : np.array
        The covariance matrix for the Gaussian copula, shape (d, d), where d is
        the number of sites times the number of spacetime elements. All
        spacetime elements are concatenated.

    random_state : int or None
        Seed for the random number generator.

    Returns
    -------
    np.array
        The uniform samples generated by the Gaussian copula, shape is
        (N, n_sites * n_times).
    """

    # Initialize rng
    rng = np.random.default_rng(random_state)

    # Generate random samples
    z = rng.multivariate_normal(mean=np.zeros_like(cov_matrix[0]), cov=cov_matrix, size=n)

    # Convert those to a uniform cdf
    uniform_samples = norm.cdf(z)
    return uniform_samples


def downscale(times, e_pos, n_pos, cloud_spd, cloud_dir, mean_csi, params,
              seed=None, scale=True, noneg=True):
    """
    Downscale the synthetic CSI for a position field using a copula. This
    function produces a synthetic time series for stationary data, e.g. a
    single hour or related time period. Method is based on approaches developed
    by Widen and Munkhammar [1].

    Parameters
    ----------
    times : pd.DatetimeIndex
        A DatetimeIndex of shape (n_times,) representing the time points for
        which to compute distances.

    e_pos : np.array
        The eastward positions of the sites, shape (n_sites,)

    n_pos : np.array
        The northward positions of the sites, shape (n_sites,)

    cloud_spd : float
        The cloud speed in the same units as Epos and Npos per unit time. This
        is used to compute the spacetime distances.

    cloud_dir : float
        The cloud direction in radians, where 0 is eastward and pi/2 is
        northward. This is used to compute the spacetime distances.

    mean_csi : float
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

    seed : int or None
        Seed for the random number generator.

    scale : bool
        Should the output be scaled to match mean_csi?

    noneg : bool
        Should negative values be set to zero?

    Returns
    -------
    np.array
        The synthetic CSI values generated by the copula, shape
        (n_times, n_sites).

    References
    ----------
    [1] Widen, J. and Munkhammar, J., "Spatio-Temporal Downscaling of Hourly
    Solar irradiance Data Using Gaussian Copulas," 2019 IEEE 46th Photovoltaic
    Specialists Conference (PVSC), Chicago, IL, USA, 2019, pp. 3172-3178,
    doi: https://dx.doi.org/10.1109/PVSC40753.2019.8980922.
    """
    csi = np.arange(-2, 2, 0.01)

    pdf, cdf, p = _solar_gmm(csi, mean_csi, params, debug=False)

    spacetime_dist = spatial.spacetime_distances(e_pos, n_pos, times, cloud_spd, cloud_dir)

    n = 1

    samples = _space_time_copula(n, spacetime_dist, csi, cdf, p, seed=seed)
    samples = samples.reshape(n, e_pos.shape[0], times.shape[0]).transpose([0, 2, 1])

    # Take only the first
    cm = samples[0, :, :]

    if noneg:
        cm[cm < 0] = 0

    if scale:
        cm *= mean_csi / np.mean(cm)
    return cm


def downscale_multihour(times, e_pos, n_pos, cloud_spd, cloud_dir, mean_csi,
                        params, seed=None, scale=True, noneg=True):
    """
    Downscale the synthetic CSI for a position field using a copula. This
    implementation allows for multiple hours to be computed in one call, with
    different specification of cloud_spd, cloud_dir, mean_csi and params.

    Based on methods developed by Widen and Munkhammar [1].

    Parameters
    ----------
    times : pd.DatetimeIndex or list(pd.DatetimeIndex)
        A DatetimeIndex of shape (n_times,) representing the time points
        for which to compute distances. Should represent a single hour or
        time period, corresponding to each entry in mean_csi.

    e_pos : np.array
        The eastward positions of the sites, shape (n_sites,)

    n_pos : np.array
        The northward positions of the sites, shape (n_sites,)

    cloud_spd : np.array
        The cloud speed in the same units as Epos and Npos per unit time.
        This is used to compute the spacetime distances. Multiple values
        can be used for multiple hours.

    cloud_dir : np.array
        The cloud direction in radians, where 0 is eastward and pi/2 is
        northward. This is used to compute the spacetime distances.
        Multiple values can be used for multiple hours.

    mean_csi : np.array
        The mean CSI for the hour, used to compute the parameters of the
        GMM. Multiple values can be used for multiple hours.

    params : dict or list(dict)
        The parameters of the model, containing at least:
        - 'comp': [a, c] parameters for the sigmoid function that determines
        the cloudy component weight based on meanCSI.
        - 'mean': [m_cloud, m_clear, c] parameters for computing the means
        of the cloud and clear components based on meanCSI and the
        cloudy component weight.
        - 'sdevClear': [sdev_clear_max, a, c] parameters for computing the
        standard deviation of the clear component based on meanCSI.
        - 'sdevCloud': [sdev_cloud_max, a, c] parameters for computing the
        standard deviation of the cloud component based on meanCSI.

    seed : int or None
        Seed for the random number generator.

    scale : bool
        Should the output be scaled to match mean_csi? Scale will occur on
        hourly basis.

    noneg : bool
        Should negative values be set to zero?

    Returns
    -------
    np.array
        The synthetic CSI values generated by the copula, shape
        (n_times * n_hours, n_sites). Concatenated for all hours.

    References
    ----------
    [1] Widen, J. and Munkhammar, J., "Spatio-Temporal Downscaling of Hourly
    Solar irradiance Data Using Gaussian Copulas," 2019 IEEE 46th Photovoltaic
    Specialists Conference (PVSC), Chicago, IL, USA, 2019, pp. 3172-3178,
    doi: https://dx.doi.org/10.1109/PVSC40753.2019.8980922.
    """
    n_hours = len(mean_csi)

    multi_times_mode = False
    multi_cloud_dir_mode = False
    multi_cloud_spd_mode = False
    multi_params_mode = False

    if not isinstance(times, pd.DatetimeIndex):
        if len(times) != n_hours:
            raise ValueError('Providing a list of times must match the length'
                             ' of mean_csi.')
        if not all(isinstance(t, pd.DatetimeIndex) for t in times):
            raise ValueError('Each entry in list of times must be a '
                             'pd.DatetimeIndex.')
        multi_times_mode = True

    if not isinstance(params, dict):
        if len(params) != n_hours:
            raise ValueError('Providing a list of params must match the length'
                             ' of mean_csi.')
        if not all(isinstance(p, dict) for p in params):
            raise ValueError('Each entry in params must be a dict.')
        multi_params_mode = True

    if not isinstance(cloud_dir, (int, float)):
        if len(cloud_dir) != 1:
            if len(cloud_dir) != n_hours:
                raise ValueError('cloud_dir must be either a single value or a'
                                 ' list matching the length of mean_csi.')
            multi_cloud_dir_mode = True

    if not isinstance(cloud_dir, (int, float)):
        if len(cloud_spd) != 1:
            if len(cloud_spd) != n_hours:
                raise ValueError('cloud_spd must be either a single value or a'
                                 ' list matching the length of mean_csi.')
            multi_cloud_spd_mode = True

    # Loop over all hours
    c = []

    # Loop over all hours
    for i in range(n_hours):
        time = times[i] if multi_times_mode else times
        param = params[i] if multi_params_mode else params
        cld_s = cloud_spd[i] if multi_cloud_spd_mode else cloud_spd
        cld_d = cloud_dir[i] if multi_cloud_dir_mode else cloud_dir
        csi = mean_csi[i]

        cm = downscale(time, e_pos, n_pos, cld_s, cld_d, csi, param, seed,
                       scale=scale, noneg=noneg)
        c.append(cm)

    c = np.concatenate(c, axis=0)

    return c
