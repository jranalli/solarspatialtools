import numpy as np
from sklearn.mixture import GaussianMixture

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
        ]
    }

    csi = np.arange(-2, 2, 0.01)

    _gaussianMixtureDistribution(csi, 0.52, param, True)

