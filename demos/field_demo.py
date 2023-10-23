import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from solartoolbox import spatial, field


datafile = "data/sample_field.h5"

cmv_a = spatial.pol2rect(9.515372409962954, 0.6196401023982533)
cmv_b = spatial.pol2rect(8.486245566393151, 2.177365219628146)

pos_utm = pd.read_hdf(datafile, mode="r", key="latlon")
df_a = pd.read_hdf(datafile, mode="r", key="data_a")
df_b = pd.read_hdf(datafile, mode="r", key="data_b")

plt.figure(figsize=(8, 6))
plt.scatter(pos_utm['E'], pos_utm['N'])

df = pd.DataFrame(index=pos_utm.index, columns=['E', 'N', 'com-E', 'com-N'])

for refs in pos_utm.index[5:10]:
    com, pos, _ = field.compute_predicted_position(
        [df_a, df_b],
        pos_utm,
        refs,
        [cmv_a, cmv_b],
        mode='preavg',
        ndownsel=8)

    df.loc[refs] = [pos_utm.loc[refs]['E'], pos_utm.loc[refs]['N'], com[0], com[1]]

    # Show the quality of the current reference's predicted pos
    plt.plot([pos_utm['E'][refs], com[0]], [pos_utm['N'][refs], com[1]], 'r-+')

plt.axis('equal')

vscale = 100
arrow_origin = [-100, 600]
for cmv, color in zip([cmv_a, cmv_b],['green','blue']):
    velvec = np.array(spatial.unit(cmv)) * vscale
    plt.arrow(arrow_origin[0], arrow_origin[1],
          velvec[0], velvec[1],
          length_includes_head=True,
          width=7,
          head_width=20,
          color=color)
plt.xlabel('E')
plt.ylabel('N')
plt.title(f'All Predicted Positions')
axes = plt.gca()
axes.xaxis.set_ticklabels([])
axes.yaxis.set_ticklabels([])

plt.show()

