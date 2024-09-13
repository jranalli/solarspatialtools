import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from solarspatialtools import spatial, field

# #############
# # READ DATA #
# #############

# This example demonstrates how the automated remapping can be used to
# determine the best possible positions for each combiner in a plant layout.
# We'll skip over some of the details of explaining the field calculation,
# please refer to the field_demo.py for a more detailed explanation.

# Load the data for Sample Plant 1 and specify the CMVs
datafile = "data/sample_plant_1.h5"
ts_data_a = pd.read_hdf(datafile, mode="r", key="data_a").infer_objects()
ts_data_c = pd.read_hdf(datafile, mode="r", key="data_c").infer_objects()
cmv_a = spatial.pol2rect(10.7, 3.29)
cmv_c = spatial.pol2rect(3.14, 1.92)

# Load the plant layout
pos_utm = pd.read_hdf(datafile, mode="r", key="utm").infer_objects()

# We now have two hours of data, each with a different CMV. These two are known
# to have roughly perpendicular directions so the field analysis should work.

# Inverter 20 is known to have an incorrect layout, so we'll focus on those
# combiners for this demonstration
refs = [nm for nm in pos_utm.index if nm.split('-')[1] in ['20']]

# #############################
# # PERFORM FIELD CALCULATION #
# #############################

# The first step is to perform the field calculation for the combiners that
# we're interested in. This will give us the expected position of the combiners
# relative to these particular CMVs.

# Perform the calculation for each combiner and store the results
preds_r0 = pd.DataFrame(index=refs, columns=['E', 'N'], dtype=float)
for ref in refs:
    pos, _ = field.compute_predicted_position([ts_data_a, ts_data_c], pos_utm, ref, [cmv_a, cmv_c])
    preds_r0.loc[ref] = pos

# field.assign_positions will return a list of tuples that can be used to
# remap the combiner positions to the new, best-fit positions.
# The tuple is written as (Combiner X, Copy-Location-From)
remap_inds_r0, _ = field.assign_positions(pos_utm.loc[refs], preds_r0)
print(str(remap_inds_r0) + '\n')

# ############################
# # REPEAT FIELD CALCULATION #
# ############################

# Because the combiner position predictions depend on their original positions,
# we need to re-run the remapping process, with these remapped positions in
# place, to see to what extent the current prediction was dependent upon those
# positions. So we'll run as follows.

# Create a remapped version of the plant layout using the remapped indices
pos_utm_r1 = field.remap_positions(pos_utm, remap_inds_r0)

# Rerun the field calculation with the remapped positions
preds_r1 = pd.DataFrame(index=refs, columns=['E', 'N'], dtype=float)
for ref in refs:
    pos, _ = field.compute_predicted_position([ts_data_a, ts_data_c], pos_utm_r1, ref, [cmv_a, cmv_c])
    preds_r1.loc[ref] = pos

# Update the assignment of combiner positions based on this recalculation.
remap_inds_r1, _ = field.assign_positions(pos_utm_r1.loc[refs], preds_r1)

# Because this remap is based upon an already remapped coordinate system
# (a remap of a remap), we need to convert it back to the original pos_utm
# coordinates.
remap_inds_r1 = field.cascade_remap(remap_inds_r0, remap_inds_r1)
print(remap_inds_r1)

# Let's check and see if it's the same as the original remap?
print(f"Has the remap stayed the same after recalculating? {remap_inds_r1 == remap_inds_r0}\n")

# ##########################
# # REPEAT FOR CONVERGENCE #
# ##########################

# Since it hasn't yet converged, we'll go through it one more time

# Update the plant layout again and recalculate again.
pos_utm_r2 = field.remap_positions(pos_utm, remap_inds_r1)
preds_r2 = pd.DataFrame(index=refs, columns=['E', 'N'], dtype=float)
for ref in refs:
    pos, _ = field.compute_predicted_position([ts_data_a, ts_data_c], pos_utm_r2, ref, [cmv_a, cmv_c])
    preds_r2.loc[ref] = pos

# Redo the assignment and redo the coordinate system cascade
remap_inds_r2, _ = field.assign_positions(pos_utm_r2.loc[refs], preds_r2)
remap_inds_r2 = field.cascade_remap(remap_inds_r1, remap_inds_r2)
print(remap_inds_r2)
print(f"Has the remap stayed the same after recalculating? {remap_inds_r2 == remap_inds_r1}\n")

# Now we've converged, so we can look at what happened.

# ################
# # PLOT RESULTS #
# ################

# We did three rounds through the prediction calculations here. We'll show a
# plot of the results from each. Each figure shows the initial predictions,
# the reassignment of those positions to the closest combiner, and finally the
# updated re-calculated predictions based on the position reassignments.

# Just as a note, the third frame in each iteration is identical to the first
# frame in the next

for i, (init_pos, remap_pos, pred_pos, next_pred) in enumerate(zip([pos_utm, pos_utm_r1, pos_utm_r2], [pos_utm_r1, pos_utm_r2, pos_utm_r2], [preds_r0, preds_r1, preds_r2], [preds_r1, preds_r2, preds_r2])):
    # Just creating some helper variables for the plot
    initial = pd.DataFrame({'E': init_pos['E'].loc[refs], 'N': init_pos['N'].loc[refs], 'E-delay': pred_pos['E'], 'N-delay': pred_pos['N']}, index=refs, dtype=float)
    reassigned = pd.DataFrame({'E': remap_pos['E'].loc[refs], 'N': remap_pos['N'].loc[refs], 'E-delay': pred_pos['E'], 'N-delay': pred_pos['N']}, index=refs, dtype=float)
    recalculated = pd.DataFrame({'E': remap_pos['E'].loc[refs], 'N': remap_pos['N'].loc[refs], 'E-delay': next_pred['E'], 'N-delay': next_pred['N']}, index=refs, dtype=float)

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

    # Plot some arrows to show the CMV
    for cmv, color in zip([cmv_a, cmv_c], ['green', 'blue']):
        velvec = np.array(spatial.unit(cmv)) * 100
        ax0.arrow(-200, -150, velvec[0], velvec[1],
                  length_includes_head=True, width=7, head_width=20, color=color)
        ax1.arrow(-200, -150, velvec[0], velvec[1],
                  length_includes_head=True, width=7, head_width=20, color=color)
        ax2.arrow(-200, -150, velvec[0], velvec[1],
                  length_includes_head=True, width=7, head_width=20, color=color)
    # Some plot config
    plt.axis('equal')
    fig.suptitle(f'Iteration {i+1} Predictions')
    plt.tight_layout()

plt.show()



