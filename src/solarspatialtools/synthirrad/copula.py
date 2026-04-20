import numpy as np

def _sigmoid(x, a, c):
    """
    Generate a sigmoid membership function.

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

def comp_param_model(x, k, a, c):
    """
    Gaussian mixture model component proportions

    Parameters
    ----------
    x : np.array
        The input variable, typically in the range [0, 1].
    k : float
        The scalar parameter, controlling the maximum value of the output.
    a : float
        The slope parameter, controlling how steep the transition is.
    c : float
        The center parameter, controlling the center of the transition.

    Returns
    -------

    """
    if not k < 1:
        raise ValueError('k should be less than 1 to avoid negative numbers')

    return 1 - k * _sigmoid(x, a, c)
