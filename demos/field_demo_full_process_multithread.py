import itertools
from multiprocessing import Pool

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from solartoolbox import cmv, spatial, field

# This demonstration is identical to field_demo_full_process.py with the
# exception that it uses the multiprocessing module to parallelize the
# calculation of the combiner positions. This can be useful for large plants
# where the calculation of the combiner positions is the bottleneck.
# Testing on my laptop showed a roughly 4x speedup for the Sample Plant 1 data
# (2:30 per iteration serially vs about 0:40 in the multithreaded case).


def main():
    """
    Function to wrap the calculations. We can't do multithreaded processing
    without putting it inside a function like this and using:
      if __name__ == "__main__":
        main()
    """
    # #############
    # # READ DATA #
    # #############

    datafile = "data/sample_plant_1.h5"

    dfs = {}
    cmvs = {}

    # Load the plant layout
    pos_utm = pd.read_hdf(datafile, mode="r", key="utm").infer_objects()
    cmb_ids = pos_utm.index
    inv_ids = list(set([nm.split('-')[1] for nm in cmb_ids]))
    inv_ids.sort()

    # Load all the time windows. If starting from truly raw data (i.e. time series
    # where potential CMV windows are not already known, you could consider
    # following the methodological approach in automate_cmv_demo.py to determine
    # time periods when the CMV is likely to be useful.
    for key in ['a', 'b', 'c', 'd', 'e']:
        dfs[key] = pd.read_hdf(datafile, mode="r", key=f"data_{key}").infer_objects()

    # ################
    # # COMPUTE CMVs #
    # ################

    # Calculate CMVs for each time window
    for key in dfs.keys():
        cld_spd, cld_dir, dat = cmv.compute_cmv(dfs[key], pos_utm, method='jamaly', options={'minvelocity': 1})
        cmvs[key] = spatial.pol2rect(cld_spd, cld_dir)
    print(pd.DataFrame(cmvs.values(), index=cmvs.keys(), columns=['cld_x', 'cld_y']))

    # Determine suitable CMV pairs based on anglular separation
    min_ang_sep = 45
    cmv_pairs = list(itertools.combinations(cmvs.keys(), 2))
    for pair in cmv_pairs.copy():
        ang_between = np.rad2deg(
            np.arccos(np.dot(spatial.unit(cmvs[pair[0]]), spatial.unit(cmvs[pair[1]]))))
        if not (min_ang_sep < ang_between < (180 - min_ang_sep)):
            cmv_pairs.remove(pair)
    print(cmv_pairs)

    # Create the initial mapping. Initially, we assume all plan positions are
    # correct, and we'll remap them as we go. remap will represent the current
    # mapping of combiners to combiners. We'll start with a 1:1 mapping.
    # pos_map will represent the current plant layout used for this iteration.
    remap = [(ref, ref) for ref in cmb_ids]
    pos_map = pos_utm.copy()

    # Make lists to hold the values at each iteration, and store current vals
    positions = []  # A list of the plant layouts
    predictions = []  # A list of combiner position predictions
    remaps = []  # A list of the remapping indices
    positions.append(pos_map)
    remaps.append(remap)

    # ###########################
    # # ITERATE TIL CONVERGENCE #
    # ###########################

    # Loop until the data converges
    working = True
    while working:
        # In this loop, we'll compute the positions within the whole plant. This
        # is the level at which we will calculate perform the remappings, based
        # on the combiner positions for the whole plant. This loop iterates,
        # updating predictions every time we get a better idea of the true plant
        # layout.

        # ####################################################
        # # PERFORM LAYOUT CALCULATION USING MULTIPROCESSING #
        # ####################################################

        # Unlike the previous example, we'll delegate the actual calculation to
        # the multiprocessing module for speed improvements.

        # If you were concerned with producing clean code you might consider
        # wrapping this section in a function that would compute and return
        # mean_preds, because the multiprocessing loop gets a bit ugly.

        # Create the data holder for the whole plant's mean combiner preds
        mean_preds = pd.DataFrame(index=cmb_ids, columns=['E', 'N'], dtype=float)

        # Initialize the pool for parallel processing
        pool = Pool()
        chunksize = 1

        # We have to do some gymnastics to get this to work. The pool.imap
        # function can only accept a single argument, so we have to pass a
        # tuple of arguments to the helper function. This is why we have the
        # list comprehension here. Most of the arguments are the same for each
        # combiner, so we can just repeat them. The only true iteration
        # variable is cmb_id.
        for cmb_id, cmb_preds in pool.imap(multipool_helper,
                                           [(cmb_id, pos_map, dfs, cmv_pairs, cmvs) for cmb_id in cmb_ids],
                                           chunksize=chunksize):
            # Use the returned data from the helper function to update the
            # mean_preds object
            mean_preds.loc[cmb_id] = cmb_preds.mean()

        # We'll then continue to use the data exactly as before.

        # ######################
        # # REASSIGN POSITIONS #
        # ######################

        # We now have mean_preds containing a predicted mean position for each
        # combiner. We can now reassign these positions to the closest combiner
        # in the plant layout. We limit these reassignments by inverter for
        # this plant, because errors are unlikely to span beyond the inverter.
        remap_new = []
        for inv in inv_ids:
            # Get the index subset matching the inverter
            refsub = [nm for nm in cmb_ids if inv in nm.split('-')[1]]
            # Compute the remapping for this inverter
            remap_inds, _ = field.assign_positions(pos_map.loc[refsub], mean_preds.loc[refsub])
            # Add it to the global remapping list
            for pair in remap_inds:
                remap_new.append(pair)
        # Cascade it against the previous iteration's remapping to convert
        # the remapping back to the original coordinate system.
        remap_new = field.cascade_remap(remap, remap_new)
        # Compute an updated plant layout based on the new remapping
        pos_upd = field.remap_positions(pos_utm, remap_new)

        # Get a list of which combiners have changed thier position since the
        # last iteration
        changed = [newmap[0] for newmap, oldmap in zip(remap_new, remap) if newmap != oldmap]
        print(f"Changed: {len(changed)}")

        # ###################################
        # # TEST FOR CONVERGENCE AND UPDATE #
        # ###################################

        # Check if we should continue. We stop if a remapping repeats or if we
        # reach a max number of iterations.
        if remap_new in remaps or len(positions) >= 10:
            working = False
        else:
            # Keep working, update our variables for the next iteration
            pos_map = pos_upd.copy()
            remap = remap_new

        # Append the new positions to our running list
        predictions.append(mean_preds.copy())
        positions.append(pos_upd.copy())
        remaps.append(remap_new)

    # ##################
    # # OUTPUT RESULTS #
    # ##################

    # The final entry in remaps contains the mapping between the original combiner
    # positions and the predicted correct positions. We could convert this to a
    # DataFrame and save it to a file if we wanted to using .to_csv() or .to_hdf()
    remap_df = pd.DataFrame(remap, columns=['id', 'copy location from']).set_index('id')
    print(remap_df)

    # Another useful item might be to look at changed, which contains a list of
    # combiners that changed position in the last iteration. When reaching the end
    # of the loop, this will contain all combiners that have failed to converge.
    print(f"Failed to converge: {changed}")

    # We can also plot the results from each iteration. Each figure shows the
    # initial predictions, the reassignment of those positions to the closest
    # combiner, and finally the updated re-calculated predictions based on the
    # position reassignments.

    for i, (init_pos, remap_pos, pred_pos, next_pred) in enumerate(zip(positions[:-1], positions[1:], predictions, predictions[1:]+[predictions[-1]])):
        # Just creating some helper variables for the plot
        initial = pd.DataFrame({'E': init_pos['E'].loc[cmb_ids], 'N': init_pos['N'].loc[cmb_ids], 'E-delay': pred_pos['E'], 'N-delay': pred_pos['N']}, index=cmb_ids, dtype=float)
        reassigned = pd.DataFrame({'E': remap_pos['E'].loc[cmb_ids], 'N': remap_pos['N'].loc[cmb_ids], 'E-delay': pred_pos['E'], 'N-delay': pred_pos['N']}, index=cmb_ids, dtype=float)
        recalculated = pd.DataFrame({'E': remap_pos['E'].loc[cmb_ids], 'N': remap_pos['N'].loc[cmb_ids], 'E-delay': next_pred['E'], 'N-delay': next_pred['N']}, index=cmb_ids, dtype=float)

        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(18, 6))

        ax0.set_title('Initial Predictions')
        ax0.scatter(pos_utm['E'], pos_utm['N'])
        ax0.plot(initial[['E', 'E-delay']].values.T, initial[['N', 'N-delay']].values.T, 'r-+')
        ax0.xaxis.set_label('E')
        ax0.yaxis.set_label('N')
        ax0.xaxis.set_ticklabels([])
        ax0.yaxis.set_ticklabels([])

        ax1.set_title('Reassign to Closest')
        ax1.scatter(pos_utm['E'], pos_utm['N'])
        ax1.plot(reassigned[['E', 'E-delay']].values.T, reassigned[['N', 'N-delay']].values.T, 'r-+')
        ax1.xaxis.set_label('E')
        ax1.yaxis.set_label('N')
        ax1.xaxis.set_ticklabels([])
        ax1.yaxis.set_ticklabels([])

        ax2.set_title('Recalculate Predictions')
        ax2.scatter(pos_utm['E'], pos_utm['N'])
        ax2.plot(recalculated[['E', 'E-delay']].values.T, recalculated[['N', 'N-delay']].values.T, 'r-+')
        ax2.xaxis.set_label('E')
        ax2.yaxis.set_label('N')
        ax2.xaxis.set_ticklabels([])
        ax2.yaxis.set_ticklabels([])

    plt.show()


def multipool_helper(args):
    """
    Helper function to allow for parallelization of the compute_predicted_position
    function. It computes the predicted position for a single combiner pair.
    Note the messy argument passing, due to the limitation that pool.map can
    only accept a single iterable as an argument.
    """
    # Can only accept a single argument, so we unpack the tuple
    cmb_id, pos_map, dfs, cmv_pairs, cmvs = args[0], args[1], args[2], args[3], args[4]

    # Data object for predictions for this combiner for each cmv pair
    cmb_preds = pd.DataFrame(index=[f'{p[0]}-{p[1]}' for p in cmv_pairs],
                             columns=['E', 'N'],
                             dtype=float)

    for pair in cmv_pairs:
        # In this loop, we compute the combiner position for an individual
        # pair of CMVs.
        cmb_pos, dat = field.compute_predicted_position(
            [dfs[pair[0]], dfs[pair[1]]],
            pos_map, cmb_id, [cmvs[pair[0]], cmvs[pair[1]]])

        # Store this single CMV pair's prediction
        cmb_preds.loc[f'{pair[0]}-{pair[1]}'] = cmb_pos

    return cmb_id, cmb_preds


if __name__ == "__main__":
    main()
