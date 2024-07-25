import warnings

import numpy as np
import pandas as pd

from solarspatialtools import spatial, cmv, signalproc
from scipy.optimize import linear_sum_assignment

warnings.filterwarnings("ignore",
                        message=".*Covariance of the parameters.*")


def compute_predicted_position(dfs, pos_utm, ref, cld_vecs=None,
                               mode='coherence', ndownsel=8,
                               navgs=5, coh_limit=0.6, freq_limit=0.02,
                               delay_method="multi"):
    """
    Compute the predicted position of a combiner based upon the cloud movement.
    Requires two separate inputs with different CMV directions. Inputs are
    provided as lists containing the data about the two different cases. For
    reference on the method, refer to [1]_.

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
        see solarspatialtools.signalproc.averaged_tf for more information.
    coh_limit : float
        The minimum coherence required for computing the delay. Default is 0.6.
        See solarspatialtools.signalproc.tf_delay for more information.
    freq_limit : float
        The maximum frequency that will be used when computing the delay.
        See solarspatialtools.signalproc.tf_delay for more info.
    delay_method : str
        The method for computing the delay. Options are 'fit' and 'multi'.

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

    References
    ----------
    .. [1] Ranalli, J., Hobbs, W., 2024. PV Plant Equipment Labels and
       Layouts Can Be Validated by Analyzing Cloud Motion in Existing Plant
       Measurements. IEEE Journal of Photovoltaics.
       DOI: `https://doi.org/10.1109/JPHOTOV.2024.3366666`
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
                                    freq_limit=freq_limit,
                                    method=delay_method)

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
        coh1 = coh1[np.isfinite(coh1)]
        ix1 = coh1.sort_values().iloc[-ndownsel:-1].index
        coh2 = combined_data[1]['coh']
        coh2 = coh2[np.isfinite(coh2)]
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


def compute_delays(df, ref, navgs=5, coh_limit=0.6, freq_limit=0.02,
                   method="multi"):
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
        see solarspatialtools.signalproc.averaged_tf for more information.
    coh_limit : float
        The minimum coherence required for computing the delay. Default is 0.6.
        See solarspatialtools.signalproc.tf_delay for more information.
    freq_limit : float
        The maximum frequency that will be used when computing the delay.
        Default is 0.02. See solarspatialtools.signalproc.tf_delay for more info.
    method : str
        The method for computing the delay. Options are 'fit' and 'multi'.

    Returns
    -------
    delay : np.ndarray
        The raw delay for the point pair computed from the TF phase
    coh : np.ndarray
        The average coherence for each transfer function within the window.
    """

    # Compute all TFs
    tf, tfcoh = signalproc.averaged_tf(df[ref], df, navgs=navgs, overlap=0.5,
                                       window='hamming', detrend=None)

    if method == "fit":  # A looping method
        delay = np.zeros_like(df.columns, dtype=float)
        coh = np.zeros_like(df.columns, dtype=float)
        for i, col in enumerate(tf.columns):
            tf_i = tf[col]
            coh_i = tfcoh[col]

            # Find the time delay from the TF phase
            delay[i], ix = signalproc.tf_delay(tf_i, coh_i,
                                               coh_limit=coh_limit,
                                               freq_limit=freq_limit,
                                               method='fit')

            # How good was the coherence? Average across TF
            tfsub = coh_i[tf.index < freq_limit]
            coh[i] = np.nansum(tfsub.values) / len(tfsub)

    elif method == "multi":  # The most efficient method
        delay, ix = signalproc.tf_delay(tf, tfcoh, coh_limit=coh_limit,
                                        freq_limit=freq_limit, method='multi')
        freq_ix = tf.index < freq_limit
        coh = np.nansum(tfcoh.values[freq_ix, :], axis=0) / np.nansum(freq_ix,
                                                                      axis=0)
    else:
        raise ValueError(f"Invalid method: {method}")
    return delay, coh


def remap_positions(data, remap, columns=None):
    """
    Remaps the data based on the remap indices. Each tuple is a pair of
    entity names (indices), where the first index is the original location, and
    the second index is the name of the combiner whose location is the true
    location of the entity identified in the first index.

    One common use case might be to operate on plant layout definition, where
    the index is the name of the entity and the columns are the UTM
    coordinates of the entity. This function could be used to swap entity
    positions.

    For more info on the remap input, see the `assign_positions` function.

    Parameters
    ----------
    data : pd. DataFrame
        The original data to be remapped. Its index must be the names of the
        plant entities, with the columns representing data about them. One
        common use case might be the UTM coordinates of the entities with
        columns 'E' and 'N'.
    remap : list
        The remap indices are a list of tuples, where the first element is the
        identifier of the entity, and the second element is the entity whose
        location is the true location of the first element index.
    columns : list, optional
        Specifies which columns to remap. If None, all columns are remapped.

    Returns
    -------
    data_fix : pd.DataFrame
        The remapped data frame
    """
    if columns is None:
        columns = data.columns

    # Fix the data to reflect the remapping
    data_fix = data.copy()
    for item in remap:
        for column in columns:
            data_fix.loc[item[0], column] = data[column][item[1]]
    return data_fix


def cascade_remap(remap1, remap2):
    """
    Cascades the effect of a remapping of a remapping. This function is useful
    for situations where you've remapped the data once, and then you've
    reprocessed that data under the new locations, and obtained a new
    remapping. This function will combine the two remappings into a single
    remapping that maps from the original data to the final remapped positions.

    After applying an initial remapping, the second remapping has indices that
    are now effectively scrambled by the new positions. Thus applying this in a
    sense has the effect of merging the values of the first column from the
    second remap into the second column from the first remap, and keeping the
    outer values as the result. This is a bit confusing, but the example below
    should help clarify.

    remap1 = [('A', 'B'), ('B', 'C'), ('C', 'A')]
    remap2 = [('A', 'B'), ('B', 'C'), ('C', 'A')]

    result = [('A', 'C'), ('B', 'A'), ('C', 'B')]

    Typically, values of remap inputs would be obtained from the
    `assign_positions` function.

    Parameters
    ----------
    remap1 : list of tuples
        The original remapping. Each tuple is a pair of indices, where the
        first index is the original location, and the second index is the name
        of the combiner whose location is the true location of the first index.

    remap2 : list of tuples
        The second remapping. Each tuple is a pair of indices, where the first
        index is the position within the remap1 space, and the second index is
        the new remapping also within that already remapped space.

    Returns
    -------
    out : list of tuples
        The new remapping that maps from the original locations to the new
        remapped positions after reprocessing.
    """

    # Put both into a dictionary
    A = dict(remap1)
    B = dict(remap2)

    # Create an empty dictionary to hold the outputs
    G = {}
    for key in A:
        G[key] = A[B[key]]

    out = [(k, v) for k, v in G.items()]
    out.sort(key=lambda tup: tup[0])  # sorts by 1st element in tuple
    return out


def assign_positions(original_pos, predicted_pos):
    """
    Compute the assignment solution to determine which predicted combiner
    position corresponds to which original expected combiner position. The
    assignment is done by minimizing the distance between the predicted and
    expected combiner positions.

    Understanding the outputs is a bit confusing, because we are mapping the
    expected combiner positions to the predicted combiner positions and each
    name therefore occurs twice, once in each column. So this will explain in
    excruciating detail to make sure that it's totally clear.

    The format of the the outputs is a list of tuples, where the first element
    is the name of the name of the column in the original data. The second
    element is the name of the element from the original data whose position in
    the site plan corresponds to the first element.

    For example:

        [('A', 'B'), ('B', 'C'), ('C', 'A')]

    means that the position for combiner 'B' in the original data should used
    as the true position for combiner 'A'. The position for combiner 'C' in the
    original data should be used as the true position for combiner 'B'. The
    position for combiner 'A' in the original data should be used as the true
    position for combiner 'C'.

    Stated another way, the data for 'A' indicates that those time series are
    really positioned at the ground position that was originally thought to be
    'B'.

    Parameters
    ----------
    original_pos : pd.DataFrame
        A dataframe containing the expected combiner positions. The
        dataframe should have at least two columns, the first two
        columns are assumed to be the easting and northing positions of the
        combiners in the site plan. Additional columns are ignored.

    predicted_pos : pd.DataFrame
        A dataframe containing the predicted combiner positions. The
        dataframe should have at least two columns, the first two
        columns are assumed to be the easting and northing positions of the
        combiners in the predicted site plan. Additional columns are ignored.

    Returns
    -------
    remap_indices : list of tuples
        A list of tuples, where the first element is the name of the column in
        the original data. The second element is the name of the element from
        the original data whose original position corresponds to the first
        element.

    data_out : pd.DataFrame
        A copy of the original data, but with the rows remapped to the optimal
        solution.
    """
    # If we have NaN in the data, calculations will fail. Assume that the NaN
    # values stay in place and remap the rest. This would only ever happen in
    # the predictions, because the original should come from the site plan.
    if np.any(np.isnan(predicted_pos)):
        from warnings import warn
        warn("Warning: Some NaN values in the data, analyzing only for real "
             "valued positions.")
        # Remove the nans
        sub_pred = predicted_pos.dropna()
        sub_orig = original_pos.loc[sub_pred.index]
        # Get an optimized subset without the NaNs
        subremap, _ = assign_positions(sub_orig, sub_pred)

        # Build the indices by manually adding back the NaNs.
        remap_indices = []
        for k in original_pos.index:  # Loop over all the original entries
            if np.any(np.isnan(predicted_pos.loc[k])):
                # This was an NaN case, so just remap it to itself
                remap_indices.append((k, k))
            else:
                # Find which optim. entry in subremap corresponds to this key
                for pair in subremap:
                    if pair[0] == k:
                        # Insert it
                        remap_indices.append(pair)
    else:
        # We have a good set of data, so compute the optimum

        # Construct the cost matrix, which shows the distance for each combo
        # from original to plan
        C = np.zeros((len(original_pos), len(predicted_pos)))
        E_plan = original_pos.values[:, 0]
        N_plan = original_pos.values[:, 1]
        E_delay = predicted_pos.values[:, 0]
        N_delay = predicted_pos.values[:, 1]
        for i in range(len(E_plan)):
            for j in range(len(E_plan)):
                # Error is the distances between the inferred and expected pts
                C[i, j] = np.sqrt((E_plan[i] - E_delay[j]) ** 2 +
                                  (N_plan[i] - N_delay[j]) ** 2
                                  )

        # Perform the optimization using scipy algorithm
        row_ind, col_ind = linear_sum_assignment(C)

        # Define the remapping from the original data to the new data
        remap_indices = list(zip(original_pos.index[col_ind],
                                 original_pos.index[row_ind]))
        remap_indices.sort(key=lambda tup: tup[0])

    # Create a fixed copy of the data to reflect the remapping
    data_out = original_pos.copy()
    data_out = remap_positions(data_out, remap_indices)

    return remap_indices, data_out
