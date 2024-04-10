import itertools

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from solartoolbox import cmv, spatial, field

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x):
        return x

def main():
    datafile = "data/sample_plant_1.h5"

    dfs = {}
    cmvs = {}

    pos_utm = pd.read_hdf(datafile, mode="r", key="utm").infer_objects()
    refs = pos_utm.index

    for key in ['a', 'b', 'c', 'd', 'e']:
        dfs[key] = pd.read_hdf(datafile, mode="r", key=f"data_{key}").infer_objects()

    # Calculate CMVs
    for key in dfs.keys():
        cld_spd, cld_dir, dat = cmv.compute_cmv(dfs[key], pos_utm, method='jamaly', options={'minvelocity': 1})
        cmvs[key] = spatial.pol2rect(cld_spd, cld_dir)


    print(pd.DataFrame(cmvs.values(), index=cmvs.keys(), columns=['cld_x', 'cld_y']))

    # Determine suitable CMV pairs
    min_ang_sep = 45
    cmv_pairs = list(itertools.combinations(cmvs.keys(), 2))
    for pair in cmv_pairs.copy():
        ang_between = np.rad2deg(
            np.arccos(np.dot(spatial.unit(cmvs[pair[0]]), spatial.unit(cmvs[pair[1]]))))
        if not (min_ang_sep < ang_between < (180 - min_ang_sep)):
            cmv_pairs.remove(pair)
    print(cmv_pairs)

    positions_initial = []
    positions_remapped = []
    predictions_initial = []
    predictions_recalc = []

    remaps = []

    working = True
    remap = [(ref, ref) for ref in refs]
    pos_remap = pos_utm.copy()

    # get the unique inverter names out of the index.
    invs = list(set([nm.split('-')[1] for nm in refs]))
    invs.sort()

    while working:
        positions_initial.append(pos_remap.copy())

        mean_preds = pd.DataFrame(index=refs, columns=['E', 'N'], dtype=float)

        for ref in tqdm(refs):
            cmb_preds = pd.DataFrame(index=[f'{p[0]}-{p[1]}' for p in cmv_pairs], columns=['E', 'N'], dtype=float)

            for pair in cmv_pairs:
                com, dat = field.compute_predicted_position(
                    [dfs[pair[0]], dfs[pair[1]]],
                    pos_remap,
                    ref,
                    [cmvs[pair[0]], cmvs[pair[1]]],
                    mode='preavg',
                    ndownsel=8,
                    delay_method="multi")

                cmb_preds.loc[f'{pair[0]}-{pair[1]}'] = com

            mean_preds.loc[ref] = cmb_preds.mean()

        predictions_initial.append(mean_preds.copy())


        # Compute remaps on an inverter-by-inverter basis
        remap_new = {ref: None for ref in refs}
        for inv in invs:
            refsub = [nm for nm in refs if inv in nm.split('-')[1]]
            remap_inds, _ = field.assign_positions(pos_remap.loc[refsub], mean_preds.loc[refsub])
            for (x,y) in remap_inds:
                remap_new[x] = y

        # Convert to tuple format
        remap_new = [(a,b) for a,b in remap_new.items()]

        remap_new = field.cascade_remap(remap, remap_new)
        pos_upd = field.remap_positions(pos_utm, remap_new)
        positions_remapped.append(pos_upd.copy())
        pos_remap = pos_upd.copy()

        # Check which combiners have changed
        changed = [newmap[0] for newmap, oldmap in zip(remap_new, remap) if newmap != oldmap]
        print(f"Changed: {len(changed)}")

        if remap_new in remaps or len(positions_remapped) >= 10:
            working = False
            remaps.append(remap_new)
        else:
            remap = remap_new
            remaps.append(remap)

    for i, (init_pos, remap_pos, pred_pos, next_pred) in enumerate(zip(positions_initial, positions_remapped, predictions_initial, predictions_initial[1:]+[predictions_initial[-1]])):
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

    plt.show()

if __name__ == "__main__":
    main()
