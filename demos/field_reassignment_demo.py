import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from solartoolbox import spatial, field

# #############
# # READ DATA #
# #############

# This is the name of the default datafile
# It contains the definition of the plant layout as well as
# two time periods A) and B) that represent two different CMV periods
datafile = "data/sample_plant_1.h5"

# Input the CMVs, see cmv_demo.py for examples of how to calculate these
cmv_a = spatial.pol2rect(10.7, 3.29)
cmv_c = spatial.pol2rect(3.14, 1.92)

# pos_utm is a pandas DataFrame. The index is the combiner ID, and the columns
# 'E' and 'N' specify the combiner position East and North in a UTM-like
# geographic projection
pos_utm = pd.read_hdf(datafile, mode="r", key="utm").infer_objects()

# df_a contains the individual time series for each combiner. The index is
# the time (with an arbitrary offset, so it begins at 00:00:00). The columns
# are keyed by the combiner ID. df_a and df_b each represents a single hour of
# data with a well defined CMV.
df_a = pd.read_hdf(datafile, mode="r", key="data_a").infer_objects()
df_c = pd.read_hdf(datafile, mode="r", key="data_c").infer_objects()


# #############################
# # PERFORM FIELD CALCULATION #
# #############################

# Create a data frame to hold the predicted data
# Columns E and N will be the expected original position of the combiner
# Columns E-delay and N-delay will be the predicted position coordinates

# Inverter 20 is known to have an incorrect layout, so we'll extract just those
# combiners for this demonstration
refs = [nm for nm in pos_utm.index if nm.split('-')[1] in ['20']]

# Create a data frame to hold the predictions
pos_pred_orig = pd.DataFrame(index=refs, columns=['E', 'N', 'E-delay', 'N-delay'], dtype=float)
preds = pd.DataFrame(index=refs, columns=['E', 'N'], dtype=float)
for ref in refs:

    # compute_predicted_position performs the entire calculation
    # it returns the expected aggregate position of the combiner (pos)
    pos, _ = field.compute_predicted_position(
        [df_a, df_c], pos_utm, ref,[cmv_a, cmv_c])

    # Add this combiner's calculated values to the output DataFrame
    pos_pred_orig.loc[ref] = [pos_utm.loc[ref]['E'], pos_utm.loc[ref]['N'], pos[0], pos[1]]
    preds.loc[ref] = pos

# remap_indices, _ = field.assign_positions(pos_pred_orig[['E', 'N']], pos_pred_orig[['E-delay', 'N-delay']])
remap_indices, _ = field.assign_positions(pos_utm.loc[refs], preds)
print(remap_indices)

# We've changed the source locations for these combiners, but we'll keep their predictions
pos_pred_remap = field.remap_data(pos_pred_orig[['E', 'N']], remap_indices)
pos_pred_remap['E-delay'] = pos_pred_orig['E-delay']
pos_pred_remap['N-delay'] = pos_pred_orig['N-delay']

# Because the combiner position predictions depend on their original positions,
# we need to re-run the remapping process to see whether the current prediction
# is dependent upon those positions. So we'll run as follows.
pos_utm_remap1 = field.remap_data(pos_utm, remap_indices)
pos_pred_repeat1 = pd.DataFrame(index=refs, columns=['E', 'N', 'E-delay', 'N-delay'], dtype=float)
for ref in refs:

    # compute_predicted_position performs the entire calculation
    # it returns the expected aggregate position of the combiner (pos)
    pos, _ = field.compute_predicted_position(
        [df_a, df_c],  # The dataframes with the two one hour periods
        pos_utm_remap1,  # the dataframe specifying the combiner positions
        ref,  # the position within pos_utm to calculate about
        [cmv_a, cmv_c],  # The two individual CMVs for the DFs
        mode='preavg',  # Mode for downselecting the comparison points
        ndownsel=8)  # Num points to use for downselecting

    # Add this combiner's calculated values to the output DataFrame
    pos_pred_repeat1.loc[ref] = [pos_utm_remap1.loc[ref]['E'], pos_utm_remap1.loc[ref]['N'], pos[0], pos[1]]

remap_indices_repeat1, _ = field.assign_positions(pos_pred_repeat1[['E', 'N']], pos_pred_repeat1[['E-delay', 'N-delay']])
# Because this remap is based upon an already remapped coordinate system (a remap of a remap), we need to
# use the field.cascade_remap function to convert it back to the original pos_utm coordinates.
remap_indices_repeat1 = field.cascade_remap(remap_indices, remap_indices_repeat1)
print(remap_indices_repeat1)

pos_pred_remap_repeat1 = field.remap_data(pos_utm.loc[refs], remap_indices_repeat1)
pos_pred_remap_repeat1['E-delay'] = pos_pred_repeat1['E-delay']
pos_pred_remap_repeat1['N-delay'] = pos_pred_repeat1['N-delay']


pos_utm_remap2 = field.remap_data(pos_utm, remap_indices_repeat1)
pos_pred_repeat2 = pd.DataFrame(index=refs, columns=['E', 'N', 'E-delay', 'N-delay'], dtype=float)
for ref in refs:

    # compute_predicted_position performs the entire calculation
    # it returns the expected aggregate position of the combiner (pos)
    pos, _ = field.compute_predicted_position(
        [df_a, df_c],  # The dataframes with the two one hour periods
        pos_utm_remap1,  # the dataframe specifying the combiner positions
        ref,  # the position within pos_utm to calculate about
        [cmv_a, cmv_c],  # The two individual CMVs for the DFs
        mode='preavg',  # Mode for downselecting the comparison points
        ndownsel=8)  # Num points to use for downselecting

    # Add this combiner's calculated values to the output DataFrame
    pos_pred_repeat2.loc[ref] = [pos_utm_remap2.loc[ref]['E'], pos_utm_remap2.loc[ref]['N'], pos[0], pos[1]]

remap_indices_repeat2, _ = field.assign_positions(pos_pred_repeat2[['E', 'N']], pos_pred_repeat2[['E-delay', 'N-delay']])
remap_indices_repeat2 = field.cascade_remap(remap_indices_repeat1, remap_indices_repeat2)
print(remap_indices_repeat1 == remap_indices_repeat2)

pos_pred_remap_repeat2 = field.remap_data(pos_utm.loc[refs], remap_indices_repeat2)
pos_pred_remap_repeat2['E-delay'] = pos_pred_repeat2['E-delay']
pos_pred_remap_repeat2['N-delay'] = pos_pred_repeat2['N-delay']


# ################
# # PLOT RESULTS #
# ################

# Create a figure to show results
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
# Plot all default positions
ax0.set_title('Initial Calculation')
ax0.scatter(pos_utm['E'], pos_utm['N'])
ax0.plot(pos_pred_orig[['E', 'E-delay']].values.T, pos_pred_orig[['N', 'N-delay']].values.T, 'r-+')
ax0.xaxis.set_label('E')
ax0.yaxis.set_label('N')
ax0.xaxis.set_ticklabels([])
ax0.yaxis.set_ticklabels([])
ax1.set_title('Reassigned Positions')
ax1.scatter(pos_utm['E'], pos_utm['N'])
ax1.plot(pos_pred_remap[['E', 'E-delay']].values.T, pos_pred_remap[['N', 'N-delay']].values.T, 'r-+')
ax1.xaxis.set_label('E')
ax1.yaxis.set_label('N')
ax1.xaxis.set_ticklabels([])
ax1.yaxis.set_ticklabels([])
plt.tight_layout()

# Plot some arrows to show the CMV
for cmv, color in zip([cmv_a, cmv_c], ['green', 'blue']):
    velvec = np.array(spatial.unit(cmv)) * 100
    ax0.arrow(-200, -150, velvec[0], velvec[1],
              length_includes_head=True, width=7, head_width=20, color=color)
    ax1.arrow(-200, -150, velvec[0], velvec[1],
              length_includes_head=True, width=7, head_width=20, color=color)

# Some plot config
plt.axis('equal')
fig.suptitle(f'Iteration 1 Predictions')
plt.tight_layout()

# Create a figure to show results
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
# Plot all default positions
ax0.set_title('Initial Calculation')
ax0.scatter(pos_utm['E'], pos_utm['N'])
ax0.plot(pos_pred_repeat1[['E', 'E-delay']].values.T, pos_pred_repeat1[['N', 'N-delay']].values.T, 'r-+')
ax0.xaxis.set_label('E')
ax0.yaxis.set_label('N')
ax0.xaxis.set_ticklabels([])
ax0.yaxis.set_ticklabels([])
ax1.set_title('Reassigned Positions')
ax1.scatter(pos_utm['E'], pos_utm['N'])
ax1.plot(pos_pred_remap_repeat1[['E', 'E-delay']].values.T, pos_pred_remap_repeat1[['N', 'N-delay']].values.T, 'r-+')
ax1.xaxis.set_label('E')
ax1.yaxis.set_label('N')
ax1.xaxis.set_ticklabels([])
ax1.yaxis.set_ticklabels([])
plt.tight_layout()

# Plot some arrows to show the CMV
for cmv, color in zip([cmv_a, cmv_c], ['green', 'blue']):
    velvec = np.array(spatial.unit(cmv)) * 100
    ax0.arrow(-200, -150, velvec[0], velvec[1],
              length_includes_head=True, width=7, head_width=20, color=color)
    ax1.arrow(-200, -150, velvec[0], velvec[1],
              length_includes_head=True, width=7, head_width=20, color=color)

# Some plot config
plt.axis('equal')
fig.suptitle(f'Iteration 2 Predictions')
plt.tight_layout()


# Create a figure to show results
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 6))
# Plot all default positions
ax0.set_title('Initial Calculation')
ax0.scatter(pos_utm['E'], pos_utm['N'])
ax0.plot(pos_pred_repeat2[['E', 'E-delay']].values.T, pos_pred_repeat2[['N', 'N-delay']].values.T, 'r-+')
ax0.xaxis.set_label('E')
ax0.yaxis.set_label('N')
ax0.xaxis.set_ticklabels([])
ax0.yaxis.set_ticklabels([])
ax1.set_title('Reassigned Positions')
ax1.scatter(pos_utm['E'], pos_utm['N'])
ax1.plot(pos_pred_remap_repeat2[['E', 'E-delay']].values.T, pos_pred_remap_repeat2[['N', 'N-delay']].values.T, 'r-+')
ax1.xaxis.set_label('E')
ax1.yaxis.set_label('N')
ax1.xaxis.set_ticklabels([])
ax1.yaxis.set_ticklabels([])
plt.tight_layout()

# Plot some arrows to show the CMV
for cmv, color in zip([cmv_a, cmv_c], ['green', 'blue']):
    velvec = np.array(spatial.unit(cmv)) * 100
    ax0.arrow(-200, -150, velvec[0], velvec[1],
              length_includes_head=True, width=7, head_width=20, color=color)
    ax1.arrow(-200, -150, velvec[0], velvec[1],
              length_includes_head=True, width=7, head_width=20, color=color)

# Some plot config
plt.axis('equal')
fig.suptitle(f'Iteration 3 Predictions')
plt.tight_layout()


plt.show()



