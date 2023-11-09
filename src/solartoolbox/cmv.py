import numpy as np
from scipy.optimize import minimize_scalar, leastsq

from solartoolbox.signalproc import correlation
from solartoolbox.spatial import project_vectors, compute_vectors
from solartoolbox.spatial import pol2rect, magnitude

from enum import Enum
import itertools


class Flag(Enum):
    GOOD = 0
    NOSIGNAL = 1
    LOWVAR_S = 2
    LOWCCF = 3
    OVERCAST = 4
    NOLAG = 5
    LOWCORR = 6
    STATS = 7
    VELLIMIT = 8
    ALLBAD = 9
    TREND = 10
    SAME = 11

    DESCS = {
        GOOD: "good data",
        NOSIGNAL: "one signal has zero variance, likely bogus",
        LOWVAR_S: "Low variation ratio in a signal",
        LOWCCF: "Low cloud cover fraction in a signal (i.e. too clear)",
        OVERCAST: "low overall kt in both signals",
        NOLAG: "no lag computed (infinite velocity)",
        LOWCORR: "low lagging correlation (<0.8)",
        VELLIMIT: "Jamaly and Kleissl Low velocity limit",

        ALLBAD: "no good datapoints",
        TREND: "trend",
        SAME: "pair is same site"}


class WindspeedData:
    """
    A holder data structure
    """
    corr_raw = None  # Zero-lag cross-correlations relative to ref point
    corr_lag = None  # Max-lag cross-correlation
    corr_gain = None  # Gain in cross-correlation by allowing lag
    allpairs = None  # Points used for windspeed measurement
    pair_lag = None  # Point-wise maximum lag
    pair_dists = None  # Point-wise distance to reference in m
    wind_speed = None  # Wind speed in m/s
    wind_angle = None  # Wind dir in radians
    pair_flag = None
    flag = None  # A data quality flag
    vectors = None
    method_data = None  # A holder for method-specific data


def _get_pairs(all_points, must_contain=None, replacement=True):
    """
    Build the array of all possible site pairs from a given list of points.
    Works by generating number of combinations (with replacement)

    Parameters
    ----------
    all_points : iterable
        List of all point ids

    must_contain : iterable (optional, default: None)
        List of points that must be contained within all entries of the final
        list. Use for example to force all combinations to contain a reference
        site.

    replacement : boolean (optional, default: True)
        Should the list of pairs be executed with replacement? True will result
        in the identity pair (i.e. 1-1) being generated, while False will lead
        to only non-identical pairs.

    Returns
    -------
    pairs : np.array
        An N x 2 numpy array containing all the combination pairs.

    """

    # # Old Method - All possible permutations (2-way)
    # pairs_all = list(itertools.permutations(all_points, 2))
    # # Permutations with replacement
    # pairs_all = list(itertools.product(all_points, repeat=2))

    # New method, combination not permutation
    if replacement:
        pairs_all = list(
            itertools.combinations_with_replacement(all_points, 2))
    else:
        pairs_all = list(itertools.combinations(all_points, 2))

    if must_contain is not None:
        # Downselect to those that contain the parts
        pairs = []
        for pair in pairs_all:
            if pair[0] in must_contain or pair[1] in must_contain:
                pairs.append(pair)
        pairs = np.array(pairs)
        return np.array(pairs)
    else:
        return np.array(pairs_all)


def compute_cmv(timeseries, positions, reference_id=None, method="jamaly",
                corr_scaling='coeff', options=None):
    """
    Find Cloud Motion Vector based on clear sky index timeseries from a cluster
    of sensors using the method by Jamaly and Kleissl [1]. An alternate method
    described by Gagne [2] is available. Optionally computes relative
    to a single reference point rather than a global computation across all
    possible site pairs

    Parameters
    ----------
    timeseries : pandas.DataFrame
        dataframe of kt timeseries for all sensors. Columns should be labelled
        with the sensor id. All sensors should share a uniform time index.

    positions : pandas.DataFrame
        dataframe of the site positions. Index must be the site IDs. The first
        column should be the x coordinate of each site, and the second column
        should be the y coordinate. Coordinates must represent a rectilinear
        coordinate system, measured in meters. Consider converting to UTM using
        spatial.latlon2utm().

    reference_id : numeric, str, iterable, or None (default None)
        The identifier of a single reference site id within the dataframe.
          OR
        a list/tuple of identifiers of references within the dataframe.
          OR
        None to signify that all possible site pairs should be considered.

    method : str (default 'jamaly')
        Method to use. Currently accepted methods are 'jamaly' and 'gagne'

    corr_scaling : str (default 'coeff')
        Scaling for correlation. Choices are 'energy' and 'coeff'. Coeff
        scaling automatically normalizes to the mean of the timeseries.

    options : dict (default {})
        Dictionary of detailed arguments for the methods.
            Jamaly:
                minvelocity : float (default 0 m/s)
                    Minimum permissible pairwise velocity in m/s
                maxvelocity : float (default 70 m/s)
                    Maximum permissible pairwise velocity in m/s

    Returns
    -------
    cmv_vel : numeric
        The cloud motion vector magnitude in meters per second

    cmv_theta : numeric
        The cloud motion direction as an angle measured CCW from east in rads

    outdata : WindspeedData
        An object containing detailed data about the individual sites

    [1] M. Jamaly and J. Kleissl, “Robust cloud motion estimation by
    spatio-temporal correlation analysis of irradiance data,” Solar
    Energy, vol. 159, pp. 306–317, Jan. 2018. [Online]. Available:
    https://www.sciencedirect.com/science/article/pii/S0038092X17309556

    [2] A. Gagné, N. Ninad, J. Adeyemo, D. Turcotte, and S. Wong, “Directional
    Solar Variability Analysis,” in 2018 IEEE Electrical Power and Energy
    Conference (EPEC) (2018) pp. 1–6, iSSN: 2381-2842
    https://www.researchgate.net/publication/330877949_Directional_Solar_Variability_Analysis
    """

    # Validate method
    methods = ['jamaly', 'gagne']
    method = method.lower()
    if method not in methods:
        raise ValueError('Method must be one of: ' + str(methods) + '.')

    # Validate options
    if options is None:
        options = {}
    # Specify default options by method
    if method == 'jamaly':
        defaults = {'minvelocity': 0,  # m/s
                    'maxvelocity': 70  # m/s (about 160 mph)
                    }
        method_out = {
                    'error_index': None,  # The Error Index
                    'velocity': None,  # Velocity for each pair
                    'v60': None,  # 60th percentile velocity
                    'v40': None,  # 40th percentile velocity
                    'r_qc': [],  # Ratio of peak to mean xcorr
                    'var_s': [],  # Variation ratio of signals [s0, s1]
                      }
    if method == 'gagne':
        defaults = {}
        method_out = {
            'pcov': None,  # Covariance matrix from least squares
        }

    # Validate option keys are part of method
    for key in options:
        if key not in defaults:
            raise ValueError(f'Invalid option for {method}: ' + str(key) + '.')

    # Build out options with defaults if they're not specified
    for key in defaults:
        if key not in options:
            options[key] = defaults[key]

    # Ignore some numpy printouts, we'll deal with them manually
    np.seterr(divide='ignore')
    np.seterr(invalid='ignore')

    # Find all the needed info
    nsamples, _ = timeseries.shape
    point_ids = positions.index.to_list()

    # Set ref_ids to a list of ref_ids that should be used
    if reference_id is not None:
        # Check if it's already an iterable (not raw str though)
        if (hasattr(reference_id, "__iter__") and
                (not isinstance(reference_id, str))):
            ref_ids = reference_id
        else:
            ref_ids = [reference_id]
    else:
        # If it's none, get the full list
        ref_ids = point_ids.copy()

    # Get all possible point pairs
    pairs = _get_pairs(point_ids, ref_ids, replacement=True)
    npts = len(pairs)

    # Get some parameters about the time series
    ts = timeseries  # Shorten the name for simplicity
    dt = (timeseries.index[1] - timeseries.index[0]).total_seconds()

    # Set up output placeholders
    corr_zero = np.zeros(npts)   # xcorr at no lag for this pair
    corr_lag = np.zeros(npts)   # Peak xcorr allowing lag for this pair
    corr_mean = np.zeros(npts)  # Mean xcorrelation for this pair at all times
    corr_gain = np.zeros(npts)  # Xcorr gained by considering lag
    delay = np.zeros(npts)      # Lag of peak xcorr
    vectors_cart = np.zeros([npts, 2])  # vectors for each point pair
    pair_flags = np.empty(npts, dtype=Flag)
    pair_flags.fill(Flag.GOOD)  # flags to show if this particular pair was bad
    overall_flag = None

    # This is the alternative to looping over all the pairs
    A = ts[pairs[:, 0]]
    B = ts[pairs[:, 1]]

    from solartoolbox import signalproc
    delay, corr_csd = signalproc.compute_delays(A, B, 'csd')


    # Loop over all the pairs
    for i, pair in enumerate(pairs):
        # Get index of point within list and ids of the signals for this pair
        pair_ind = i
        ref_id = pair[0]
        pt_id = pair[1]

        # Skip the case where they're the same point. No value in this corr.
        if ref_id == pt_id:
            pair_flags[pair_ind] = Flag.SAME
            continue

        # Pick out the reference signals
        sig0 = ts[ref_id]
        sig1 = ts[pt_id]

        # Compute vector between pair
        ref_pt = np.array(positions.loc[ref_id])[0:2]
        target_pt = np.array(positions.loc[pt_id])[0:2]
        vec = compute_vectors([target_pt[0]], [target_pt[1]], ref_pt)[0]
        dist = magnitude(vec)

        # Compute the site-pair cross correlation.
        xcorr_i, lags = correlation(sig0, sig1, corr_scaling)
        lags = lags * dt
        peak_lag_index = xcorr_i.argmax()  # Index of peak correlation

        # Extract the data for the various correlations
        corr_zero[pair_ind] = xcorr_i[lags == 0]
        corr_lag[pair_ind] = xcorr_i[peak_lag_index]
        delay[pair_ind] = -lags[peak_lag_index]  # Positive lag leads the ref
        corr_mean[pair_ind] = np.mean(xcorr_i)
        corr_gain[pair_ind] = corr_lag[pair_ind] - np.abs(corr_zero[pair_ind])
        vectors_cart[pair_ind] = vec

        # ##### Pairwise QC ##### #

        # Base STD rejects faulty sensors with no variance at all
        if np.std(sig0) < 0.001 or np.std(sig1) < 0.001:
            pair_flags[pair_ind] = Flag.NOSIGNAL
            continue
        # Zero lag Reject sensor pairs with no delay
        if lags[peak_lag_index] == 0:
            pair_flags[pair_ind] = Flag.NOLAG
            continue

        if method == 'jamaly':
            # Jamaly and Kleissl - low variation ratio
            # var_s_lim = 0.1  # Jamaly and Kleissl limit
            var_s_lim = 0.05  # TODO Justify Override Jamaly value
            var_s0 = 1 - np.nanmean(sig0) / np.nanmax(sig0)
            var_s1 = 1 - np.nanmean(sig1) / np.nanmax(sig1)
            method_out['var_s'].append([var_s0, var_s1])
            if var_s0 < var_s_lim or var_s1 < var_s_lim:
                pair_flags[pair_ind] = Flag.LOWVAR_S
                continue

            # # Jamaly and Kleissl - low cloud cover fraction (i.e. too clear)
            # ccf_lim = 0.1  # More than 10% of points should have kt < 0.85
            # ccf0 = np.nansum(sig0 < 0.85) / len(sig0)  # Fraction of kt<0.85
            # ccf1 = np.nansum(sig1 < 0.85) / len(sig1)
            # if ccf0 < ccf_lim or ccf1 < ccf_lim:
            #     pair_flags[pair_ind] = Flag.LOWCCF
            #     continue

            # Ranalli - extremely overcast
            ktlim = 0.4
            if np.nanmax(sig0) < ktlim and np.nanmax(sig1) < 0.4:
                pair_flags[pair_ind] = Flag.OVERCAST
                continue

            # Jamaly and Kleissl - require minimum lagged xcorr and minimum
            # cross correlation ratio
            r_qc = 1 - np.mean(xcorr_i) / np.max(xcorr_i)
            method_out['r_qc'].append(r_qc)
            if np.max(xcorr_i) < 0.8 or r_qc < 0.8:
                pair_flags[pair_ind] = Flag.LOWCORR
                continue

            # Jamaly and Kleissl distance limit is confusing; I modified here
            # Original appears to require that the max velocity be less than
            # 1 m/s I implement it as a literal velocity limit rather than dist
            min_v = options['minvelocity']  # m/s (could be introduced)
            max_v = options['maxvelocity']  # m/s (70 m/s, around 160 mph)
            max_lag_fraction = 1  # fraction of timeseries for max lag
            max_lag = np.abs(np.max(lags)) * max_lag_fraction  # max dt allowed
            if (delay[pair_ind] > max_lag or
                    not (min_v < np.abs(dist/lags[peak_lag_index]) < max_v)):
                pair_flags[pair_ind] = Flag.VELLIMIT
                continue

        elif method == 'gagne':
            pass  # Gagne QC isn't pairwise

    # Done looping over pairs, begin to compute the aggregate performance

    # Perform the Gagne QC, which looks for correlation on the whole set to
    # decide which pairs are worth using
    if method == 'gagne':
        corr_inds = corr_lag >= 0.99
        if sum(corr_inds[pair_flags == Flag.GOOD]) < 12:
            corr_inds = corr_lag >= 0.95
        if sum(corr_inds[pair_flags == Flag.GOOD]) < 12:
            corr_inds = corr_lag >= 0.9
        if sum(corr_inds[pair_flags == Flag.GOOD]) < 12:
            overall_flag = Flag.ALLBAD
        pair_flags[np.bitwise_not(corr_inds)] = Flag.LOWCORR

    # Jump out if no pairs passed QC
    if np.sum(pair_flags == Flag.GOOD) <= 1:  # they all fail QC
        overall_flag = Flag.ALLBAD
        w = WindspeedData()
        w.allpairs = pairs
        w.flag = overall_flag
        w.pair_flag = pair_flags
        return 0, 0, w

    # Select the subsets of delay and vectors to use
    delay_good = delay[pair_flags == Flag.GOOD]
    vectors_good = vectors_cart[pair_flags == Flag.GOOD]

    if method == 'jamaly':
        # Function that we'll minimize
        def vel_variance(theta, lag, vecs):
            # Project each vector into the direction of theta
            dir_dist = project_vectors(vecs, np.array(pol2rect(1, theta)))

            # Compute wind speed for each pair, ignoring infinites
            vel = dir_dist / lag
            vel[vel == np.inf] = np.nan
            vel[vel == -np.inf] = np.nan

            # compute the variance for this value of theta
            variance = np.nanvar(vel)
            return variance

        # Minimize the variance in the computed velocities to get theta. Use
        # only the Flag.Good values in delay and the vectors
        opt = minimize_scalar(vel_variance, args=(delay_good, vectors_good))
        cmv_theta = opt.x

        # Compute the velocity for all the useful pairs at this theta. The
        # actual velocity is the median over all useful pairs
        cmv_dir_dist = project_vectors(vectors_good,
                                       np.array(pol2rect(1, cmv_theta)))
        velocity = cmv_dir_dist / delay_good
        velocity[velocity == np.inf] = np.nan
        velocity[velocity == -np.inf] = np.nan
        cmv_vel = np.nanmedian(velocity)

        # QC - Compute the error index from Shinozaki et al (Jamaly eq 3)
        v60 = np.nanpercentile(velocity, 60)
        v40 = np.nanpercentile(velocity, 40)
        err = (v60 - v40) / cmv_vel
        if np.abs(err) > 0.4:  # Set flag if appropriate
            overall_flag = Flag.TREND

        method_out['v60'] = v60
        method_out['v40'] = v40
        method_out['error_index'] = err
        method_out['velocity'] = velocity

    elif method == 'gagne':
        # Function to apply least squares to
        def resid(p, lag, coeff_vecs):
            dx, dy = coeff_vecs[:, 0], coeff_vecs[:, 1]
            ax, ay = p[0], p[1]
            # Residual as defined in Gagne 2018 equation 3
            return lag - (dx * ax + dy * ay)

        [ax_opt, ay_opt], pcov = leastsq(resid, [1, 1],
                                         args=(delay_good, vectors_good))

        method_out['pcov'] = pcov

        # Gagne 2018 equation 4
        vx = 1 / (ax_opt ** 2 + ay_opt ** 2) * ax_opt
        vy = 1 / (ax_opt ** 2 + ay_opt ** 2) * ay_opt

        cmv_vel = magnitude([vx, vy])
        cmv_theta = np.arctan2(vy, vx)

    # Make velocity always positive and change distances to accommodate
    if cmv_vel < 0:
        cmv_vel *= -1
        cmv_theta += np.pi
    cmv_theta %= (2*np.pi)
    cmv_dir_dist = project_vectors(vectors_cart,
                                   np.array(pol2rect(1, cmv_theta)))

    # Create the output data object
    outdata = WindspeedData()
    outdata.flag = overall_flag
    outdata.pair_flag = pair_flags
    outdata.pair_lag = delay
    outdata.corr_raw = corr_zero
    outdata.allpairs = pairs
    outdata.corr_lag = corr_lag
    outdata.pair_dists = cmv_dir_dist
    outdata.vectors = vectors_cart
    outdata.corr_gain = corr_gain
    outdata.wind_angle = cmv_theta
    outdata.wind_speed = cmv_vel
    outdata.method_data = method_out

    return cmv_vel, cmv_theta, outdata
