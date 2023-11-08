import warnings

import numpy as np
import pandas as pd

from solartoolbox import spatial, cmv, signalproc

warnings.filterwarnings("ignore",
                        message=".*Covariance of the parameters.*")


def compute_predicted_position(dfs, pos_utm, ref, cld_vecs=None,
                               mode='coherence', ndownsel=8,
                               navgs=5, coh_limit=0.6, freq_limit=0.02):
    """
    Compute the predicted position of a combiner based upon the cloud movement.
    Requires two separate inputs with different CMV directions. Inputs are
    provided as lists containing the data about the two different cases.

    Parameters
    ----------
    dfs : list[pandas.DataFrame]
        Time series of the data. Columns are the points, rows are the time.
    pos_utm : pandas.DataFrame
        The known positions of all sensors. Must be in UTM (x,y) coordinates.
        Index should match the columns of df
    ref : str
        The name of the reference point. Must be a column in df and a row in
        pos_utm.
    cld_vecs : np.ndarray
        The cloud motion vectors, [[Vx1, Vy1], [Vx2, Vy2]]
    mode : str
        The method for downselecting the points to use for computing the
        position. Options are 'coherence', 'global_coherence', 'distance', and
        'all'. 'coherence' will choose the n best points for each CMV that
        experience the best coherence. 'global_coherence' computes an
        orthogonal sum of coherence for both CMV's and chooses the same best n
        points across both. Default is 'coherence'.
    ndownsel : int
        The number of points to downselect to. Default is 8.
    navgs : int
        The number of averages to use when computing the transfer function.
        Affects both the coherence and the frequency resolution. Default is 5.
        see solartoolbox.signalproc.averaged_tf for more information.
    coh_limit : float
        The minimum coherence required for computing the delay. Default is 0.6.
        See solartoolbox.signalproc.tf_delay for more information.
    freq_limit : float
        The maximum frequency that will be used when computing the delay.
        See solartoolbox.signalproc.tf_delay for more info.

    Returns
    -------
    pos : np.ndarray
        The predicted position of the combiner (X,Y) in the coordinate system
        of pos_utm
    fulldata : list[pandas.DataFrame]
        The full data set for each of the two cases. Each element of the list
        is a pandas.DataFrame with columns 'cloud_dist', 'delay', and 'coh'
        and an index matching pos_utm. Cloud_dist is the distances implied by
        the cloud motion for that particular time along the cloud motion dir.
        Delay is the raw value of tf delay for each of the points relative to
        the reference. coh is the average coherence for that transfer function.
    """

    # Note: This approach computes results for a single reference. The sub-
    # function compute_cloud_dist needs the tf between every single location
    # and that reference point. Thus, when computing for a whole field, we
    # duplicate the work of computing the tf for every point pair. Efficiency
    # could be improved on plant-level calculations by pre-computing all delays
    # for every point pair.

    combined_data = []

    if cld_vecs is None:
        cld_vecs = []
        for df in dfs:
            cld_spd, cld_dir, dat = cmv.compute_cmv(df, pos_utm,
                                                    method='jamaly')
            cld_vecs.append(spatial.pol2rect(cld_spd, cld_dir))

    for df, cld_vec in zip(dfs, cld_vecs):

        # Get the pairwise delays
        delay, coh = compute_delays(df, ref, navgs=navgs,
                                    coh_limit=coh_limit,
                                    freq_limit=freq_limit)

        # Convert them to distances along the cloud motion vector
        delay_dist = -delay * spatial.magnitude(cld_vec)

        # Compute the actual separation distance for comparison
        # Negative gives dist from all points to ref, rather than vice versa
        pos_vecs = -spatial.compute_vectors(pos_utm['E'], pos_utm['N'],
                                            pos_utm.loc[ref][['E', 'N']])
        # Project the vec from point to ref into the cloud vector dir
        plan_dist = spatial.project_vectors(pos_vecs, cld_vec)

        error_dist = delay_dist - plan_dist['dist']

        # Aggregate
        pos_data = pd.DataFrame({'plan_dist': plan_dist['dist'],
                                 'delay_dist': delay_dist,
                                 'error_dist': error_dist,
                                 'delay': delay,
                                 'coh': coh},
                                index=pos_utm.index)

        combined_data.append(pos_data)

    if mode == 'coherence' or mode == 'preavg':  # preavg is for back compat.
        # Downselect by best coherence in each axis
        coh1 = combined_data[0]['coh']
        ix1 = coh1.sort_values().iloc[-ndownsel:-1].index
        coh2 = combined_data[1]['coh']
        ix2 = coh2.sort_values().iloc[-ndownsel:-1].index
    elif mode == 'global_coherence':
        # Downselect to the points with the strongest overall coherence
        # An orthogonal sum of coherence for the two axes
        truecoh = combined_data[0]['coh'] ** 2 + combined_data[1]['coh'] ** 2
        truecoh = pd.DataFrame(truecoh, columns=['coh'],
                               index=combined_data[0]['coh'].index)
        ix = truecoh.sort_values('coh').iloc[-ndownsel:-1, :].index
        ix1 = ix2 = ix
    elif mode == 'distance':
        # Downselect to the closest Points
        separations = pd.DataFrame({'sep': [
            spatial.magnitude(row[1] - pos_utm.loc[ref]) for row in
            pos_utm.iterrows()]}, index=pos_utm.index)
        # Zeroth index is always "self", but needs to be included
        ix = separations.sort_values('sep').iloc[:ndownsel, :].index
        ix1 = ix2 = ix
    elif mode == 'all':
        # Just compute using all points
        ix1 = ix2 = pos_utm.index
    else:
        raise ValueError(f"Invalid mode: {mode}")

    # Average the positions in the A and B directions
    combined_data = [combined_data[0].loc[ix1], combined_data[1].loc[ix2]]
    e_a = np.mean(combined_data[0]['error_dist'])
    e_b = np.mean(combined_data[1]['error_dist'])

    # The vectors to intersect are parallel to the CMVs and have magntiude
    # equal to e_a and e_b.
    pos = spatial.compute_intersection(
        np.array([spatial.pol2rect(e_a, spatial.rect2pol(*cld_vecs[0])[1])]),
        np.array([spatial.pol2rect(e_b, spatial.rect2pol(*cld_vecs[1])[1])])
    )

    # Algorithm above assumes the origin is (0, 0). Shift by the ref pt
    pos = (pos + pos_utm.loc[ref].values).flatten()

    return pos, combined_data


def compute_delays(df, ref, navgs=5, coh_limit=0.6, freq_limit=0.02):
    """
    Computes delay between groups of signals. Will find the delay between the
    reference and every using a transfer function between the reference and
    those possible points.

    Parameters
    ----------
    df : pandas.DataFrame
        Time series of the data. Columns are the points, rows are the time.
    ref : str
        The name of the reference point. Must be a column in df.
    navgs : int
        The number of averages to use when computing the transfer function.
        Affects both the coherence and the frequency resolution. Default is 5.
        see solartoolbox.signalproc.averaged_tf for more information.
    coh_limit : float
        The minimum coherence required for computing the delay. Default is 0.6.
        See solartoolbox.signalproc.tf_delay for more information.
    freq_limit : float
        The maximum frequency that will be used when computing the delay.
        Default is 0.02. See solartoolbox.signalproc.tf_delay for more info.

    Returns
    -------
    delay : np.ndarray
        The raw delay for the point pair computed from the TF phase
    coh : np.ndarray
        The average coherence for each transfer function within the window.
    """

    # Compute delay for every point pair for this reference
    delay = np.zeros_like(df.columns, dtype=float)
    coh = np.zeros_like(df.columns, dtype=float)
    ts_in = df[ref]
    for i, point in enumerate(df.columns):
        ts_out = df[point]

        # Compute TF
        tf = signalproc.averaged_tf(ts_in, ts_out, navgs=navgs, overlap=0.5,
                                    window='hamming', detrend=None)
        # Find the time delay from the TF phase
        delay[i], ix = signalproc.tf_delay(tf,
                                           coh_limit=coh_limit,
                                           freq_limit=freq_limit)

        # How good was the coherence? Average across TF
        tfsub = tf['coh'][tf.index < freq_limit]
        coh[i] = np.sum(tfsub) / len(tfsub)
        # coh[i] = np.sum(tf['coh']) / len(tf['coh'])  # Alt average of all TF

    return delay, coh


def compute_cloud_dist(df, ref, cld_vec, navgs=5,
                       coh_limit=0.6, freq_limit=0.02):
    """
    Computes the implied distance between a point and a reference based upon
    the cloud motion. Will find the delay between the reference and every
    possible point using a transfer function between the reference and those
    possible points. Will compute the rate * time = distance implied by that
    delay, along the cloud motion direction.

    Parameters
    ----------
    df : pandas.DataFrame
        Time series of the data. Columns are the points, rows are the time.
    ref : str
        The name of the reference point. Must be a column in df.
    cld_vec : np.ndarray
        The cloud motion vector, [Vx, Vy]
    navgs : int
        The number of averages to use when computing the transfer function.
        Affects both the coherence and the frequency resolution. Default is 5.
        see solartoolbox.signalproc.averaged_tf for more information.
    coh_limit : float
        The minimum coherence required for computing the delay. Default is 0.6.
        See solartoolbox.signalproc.tf_delay for more information.
    freq_limit : float
        The maximum frequency that will be used when computing the delay.
        Default is 0.02. See solartoolbox.signalproc.tf_delay for more info.

    Returns
    -------
    cloud_dist : np.ndarray
        The distances implied by the cloud motion along the cloud motion dir.
    delay : np.ndarray
        The raw delay for the point pair computed from the TF phase
    coh : np.ndarray
        The average coherence for each transfer function.
    """

    # Compute delay for every point pair for this reference
    delay = np.zeros_like(df.columns, dtype=float)
    cloud_dist = np.zeros_like(df.columns, dtype=float)
    coh = np.zeros_like(df.columns, dtype=float)
    ts_in = df[ref]
    for i, point in enumerate(df.columns):
        ts_out = df[point]

        # Compute TF
        tf = signalproc.averaged_tf(ts_in, ts_out, navgs=navgs, overlap=0.5,
                                    window='hamming', detrend=None)
        # Find the time delay from the TF phase
        delay[i], ix = signalproc.tf_delay(tf,
                                           coh_limit=coh_limit,
                                           freq_limit=freq_limit)

        # Each delay and cloud vector implies a distance
        cloud_dist[i] = -delay[i] * spatial.magnitude(cld_vec)
        # How good was the coherence? Average across TF
        tfsub = tf['coh'][tf.index < freq_limit]
        coh[i] = np.sum(tfsub) / len(tfsub)
        # coh[i] = np.sum(tf['coh']) / len(tf['coh'])  # Alt average of all TF

    return cloud_dist, delay, coh
