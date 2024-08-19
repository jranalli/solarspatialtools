import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from solarspatialtools import spatial, field

# #############
# # READ DATA #
# #############

# This is the name of the default datafile
# It contains the definition of the plant layout as well as
# two time periods A) and B) that represent two different CMV periods
datafile = "data/sample_plant_2.h5"

# Input the CMVs, see cmv_demo.py for examples of how to calculate these. It's
# important to note that the CMVs as calculated using the CMV module would be
# dependant on accurate field positions. So it's important in this step that
# the CMVs either come from an independent source, or that the errors in the
# field positions are isolated incidents, and the location data for the plant
# as a whole is somewhat accurate. Plants that were completely scrambled would
# likely require special treatment that we haven't tested the method on.
cmv_a = spatial.pol2rect(9.52, 0.62)
cmv_b = spatial.pol2rect(8.47, 2.17)

# pos_utm is a pandas DataFrame. The index is the combiner ID, and the columns
# 'E' and 'N' specify the combiner position East and North in a UTM-like
# geographic projection
pos_utm = pd.read_hdf(datafile, mode="r", key="utm")

# df_a contains the individual time series for each combiner. The index is
# the time (with an arbitrary offset, so it begins at 00:00:00). The columns
# are keyed by the combiner ID. df_a and df_b each represents a single hour of
# data with a well defined CMV.
ts_data_a = pd.read_hdf(datafile, mode="r", key="data_a")
ts_data_b = pd.read_hdf(datafile, mode="r", key="data_b")


# #############################
# # PERFORM FIELD CALCULATION #
# #############################

# Create a data frame to hold the predicted data
# Columns E and N will be the expected original position of the combiner
# Columns com-E and com-N will be the predicted position coordinates
df = pd.DataFrame(index=pos_utm.index, columns=['E', 'N', 'com-E', 'com-N'])

# Calculation is performed on each reference individually. Loop to do so.
# This may take several minutes, so the indexing uses a subset to save time
# for demo purposes.
for ref in pos_utm.index[46:62]:

    # compute_predicted_position performs the entire calculation
    # it returns the expected aggregate position of the combiner (pos)
    pos, _ = field.compute_predicted_position(
        [ts_data_a, ts_data_b],  # The dataframes with the two one hour periods
        pos_utm,  # the dataframe specifying the combiner positions
        ref,  # the position within pos_utm to calculate about
        [cmv_a, cmv_b],  # The two individual CMVs for the DFs
        mode='preavg',  # Mode for downselecting the comparison points
        ndownsel=8)  # Num points to use for downselecting

    # Add this combiner's calculated values to the output DataFrame
    df.loc[ref] = [pos_utm.loc[ref]['E'], pos_utm.loc[ref]['N'], pos[0], pos[1]]


# ################
# # PLOT RESULTS #
# ################

# Create a figure to show results
plt.figure(figsize=(6, 6))
# Plot all default positions
plt.scatter(pos_utm['E'], pos_utm['N'])
# For each row, plot a red line showing the predicted position relative to the
# plans.
for row in df.iterrows():
    r = row[1]
    plt.plot([r['E'], r['com-E']], [r['N'], r['com-N']], 'r-+')

# Plot some arrows to show the CMV
for cmv, color in zip([cmv_a, cmv_b],['green','blue']):
    velvec = np.array(spatial.unit(cmv)) * 100
    plt.arrow(-100, 600, velvec[0], velvec[1],
              length_includes_head=True, width=7, head_width=20, color=color)

# Some plot config
plt.axis('equal')
plt.xlabel('E')
plt.ylabel('N')
plt.title(f'All Predicted Positions')
axes = plt.gca()
axes.xaxis.set_ticklabels([])
axes.yaxis.set_ticklabels([])
plt.tight_layout()

plt.show()

