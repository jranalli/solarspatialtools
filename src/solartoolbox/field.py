import warnings

import numpy as np
import pandas as pd

from solartoolbox import spatial, cmv, signalproc

warnings.filterwarnings("ignore",
                        message=".*Covariance of the parameters.*")


def compute_predicted_position(dfs, pos_utm, ref, cld_vecs=None, mode='preavg',
                               ndownsel=8, navgs=5,
                               coh_limit=0.6, freq_limit=0.02):
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
        position. Options are 'coherence', 'distance', and 'preavg'. Default is
        'preavg'. Any other option will use all points.
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
    com : np.ndarray
        The predicted position of the combiner (X,Y) in the coordinate system
        of pos_utm
    pos : np.ndarray
        The predicted position based on every individual point's relationship
        to the reference point. (X,Y) in the coordinate system of pos_utm
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

    fulldata = []

    if cld_vecs is None:
        cld_vecs = []
        for df in dfs:
            cld_spd, cld_dir, dat = cmv.compute_cmv(df, pos_utm,
                                                    method='jamaly')
            cld_vecs.append(spatial.pol2rect(cld_spd, cld_dir))

    distances = []
    for df, cld_vec in zip(dfs, cld_vecs):

        # Compute the distance along the CMV implied by the delay
        cloud_dist, delay, coh = compute_cloud_dist(df, ref, cld_vec,
                                                    navgs=navgs,
                                                    coh_limit=coh_limit,
                                                    freq_limit=freq_limit)

        # Compute the actual separation distance for comparison
        # Negative gives dist from all points to ref, rather than vice versa
        pos_vecs = -spatial.compute_vectors(pos_utm['E'], pos_utm['N'],
                                            pos_utm.loc[ref][['E', 'N']])
        # Project the vec from point to ref into the cloud vector dir
        true_dist = spatial.project_vectors(pos_vecs, cld_vec)

        # Aggregate
        pos_dist = pd.DataFrame({'dist': true_dist['dist'],
                                 'cloud_dist': cloud_dist,
                                 'delay': delay,
                                 'coh': coh},
                                index=pos_utm.index)

        # Store them for both different wind directions
        distances.append(pos_dist['cloud_dist'])

        fulldata.append(pos_dist)

    if mode == 'coherence':
        # Downselect to the points with the strongest overall coherence
        # An orthogonal sum of coherence for the two axes
        truecoh = fulldata[0]['coh'] ** 2 + fulldata[1]['coh'] ** 2
        truecoh = pd.DataFrame(truecoh, columns=['coh'],
                               index=fulldata[0]['coh'].index)
        strongest = truecoh.sort_values('coh').iloc[-ndownsel:-1, :]
        distances_new = [distances[0][strongest.index],
                         distances[1][strongest.index]]
        pos_utm_new = pos_utm.loc[strongest.index]
        pos = compute_intersection(cld_vecs, distances_new)
        pos = pos + pos_utm_new.values
        fulldata = [fulldata[0].loc[strongest.index],
                    fulldata[1].loc[strongest.index]]
    elif mode == 'distance':
        # Downselect to the closest Points
        separations = pd.DataFrame({'sep': [
            spatial.magnitude(row[1] - pos_utm.loc[ref]) for row in
            pos_utm.iterrows()]}, index=pos_utm.index)
        # Zeroth index is always "self", but needs to be included
        closest = separations.sort_values('sep').iloc[:ndownsel, :]
        distances_new = [distances[0][closest.index],
                         distances[1][closest.index]]
        pos_utm_new = pos_utm.loc[closest.index]
        pos = compute_intersection(cld_vecs, distances_new)
        pos = pos + pos_utm_new.values
        fulldata = [fulldata[0].loc[closest.index],
                    fulldata[1].loc[closest.index]]
    elif mode == 'preavg':
        # Compute the average position on the two axes first, prior to
        # computing the implied position

        # Downselect by best coherence in each axis
        coh1 = fulldata[0]['coh']
        strongest1 = coh1.sort_values().iloc[-ndownsel:-1]
        coh2 = fulldata[1]['coh']
        strongest2 = coh2.sort_values().iloc[-ndownsel:-1]

        x = np.mean(distances[0][strongest1.index] -
                    fulldata[0].loc[strongest1.index]['dist'])
        y = np.mean(distances[1][strongest2.index] -
                    fulldata[1].loc[strongest2.index]['dist'])
        pos = compute_intersection(cld_vecs,
                                        [np.array([x]), np.array([y])])
        fulldata = [fulldata[0].loc[strongest1.index],
                    fulldata[1].loc[strongest2.index]]
        pos = pos_utm.loc[[ref]] + pos.flatten()
    else:
        # Just compute using all points
        pos = compute_intersection(cld_vecs, distances)
        pos = pos + pos_utm.values

    # Algorithm above assumes the origin is 0, 0. So offset with the actual
    # origin of each source position
    com = np.nanmean(pos, axis=0)
    return com, pos, fulldata


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


def compute_intersection(axes, magnitudes):
    """
    Computes intersection of the lines perpendicular to two vectors.

    Parameters
    ----------
    axes : list[list] or np.array
        a 2x2 array of the vectors defining the two axes. Outer index is the
         axis, inner index is the coordinate x,y.
        in the form [[x1,y1], [x2,y2]]
    magnitudes : list[pd.Series] or np.array
        Magnitude of vector along each axis. Outer index is the
        axis, inner index is the individual magnitude, e.g.
        [[d1_1,d1_2,d1_3, ...], [d2_1,d2_2,d2_3, ...]]

    Returns
    -------
    pos : pd.DataFrame or np.array
        The computed position for each individual source position
    """

    # Alternate method available via line intersections at:
    # https://stackoverflow.com/questions/3252194/numpy-and-line-intersections

    # Derive by computing unit vectors a & b in dir of A&B
    # Then compute a dot C and b dot C. These should be equal to the
    # magnitudes of A and B. You now have an equation for x, y in terms
    # of the known a, b, A, B. Thanks @ Nate Williams and David Starling.
    cld_unit_A = spatial.unit(axes[0])
    cld_unit_B = spatial.unit(axes[1])

    if isinstance(magnitudes[0], pd.Series):
        A = magnitudes[0].values[:, np.newaxis] * cld_unit_A
        B = magnitudes[1].values[:, np.newaxis] * cld_unit_B
    else:
        A = magnitudes[0][:, np.newaxis] * cld_unit_A
        B = magnitudes[1][:, np.newaxis] * cld_unit_B

    Ax = A[:, 0]
    Ay = A[:, 1]
    Bx = B[:, 0]
    By = B[:, 1]
    Amag2 = Ax**2 + Ay**2  # magnitude squared
    Bmag2 = Bx**2 + By**2
    x = (Bmag2 * Ay - By * Amag2) / (Bx*Ay-Ax*By)
    y = (Amag2 * Bx - Ax * Bmag2) / (Bx*Ay-Ax*By)

    if isinstance(magnitudes[0], (pd.Series, pd.DataFrame)):
        # Build back to dataframe
        pos = pd.DataFrame(np.array([x, y]).T, columns=['E', 'N'],
                           index=magnitudes[0].index)
    else:
        pos = np.array([x, y]).T

    return pos
