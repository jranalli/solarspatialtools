import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from solartoolbox import stats, spatial, cmv

from scipy.stats import linregress

def main():
    fn = 'data/hope_melpitz_10s.h5'
    avg_interval = '1h'
    day_start = '2013-09-08 00:00:00'
    day_stop = '2013-09-12 00:00:00'

    pos = pd.read_hdf(fn, mode="r", key="latlon")
    pos_utm = spatial.latlon2utm(pos['lat'], pos['lon'])
    ts = pd.read_hdf(fn, mode="r", key="data")


    ts = ts[np.logical_and(day_start<ts.index, ts.index<day_stop)]

    vs = ts.resample(avg_interval).apply(
                    lambda x: stats.variability_score(x[ts.columns]))

    vs = vs.median(axis=1).sort_values(ascending=False)
    vs = vs.iloc[0:20]
    print(vs)


    ngood_min = 50
    rval_min = 0.85

    cmvs = pd.DataFrame(columns=["cld_spd", "cld_dir_rad", "df_p95", "ngood", "rval", "stderr", "error_index"])
    cmvs_flags = []
    for date in vs.index:
        hour = pd.date_range(date, date + pd.to_timedelta('1h'), freq='10s')
        hour = ts.loc[hour]

        hourlymax = np.mean(hour.quantile(0.95))
        kt = hour / hourlymax

        cld_spd, cld_dir, dat = cmv.compute_cmv(kt, pos_utm, method='jamaly', options={'minvelocity': 1})
        cmvs_flags.append(dat.flag)
        s, i, r, p, se = linregress(dat.pair_lag[dat.pair_flag == cmv.Flag.GOOD], dat.pair_dists[dat.pair_flag == cmv.Flag.GOOD])
        dat.ngood = np.sum(dat.pair_flag == cmv.Flag.GOOD)
        dat.rval = r
        dat.stderr = se
        cmvs.loc[date] = [cld_spd, cld_dir, hourlymax, dat.ngood, dat.rval, dat.stderr, np.abs(dat.method_data["error_index"])]

    pd.options.display.max_columns = None
    pd.options.display.max_rows = None
    print(cmvs)

    indices = cmv.optimum_subset(*spatial.pol2rect(cmvs.cld_spd, cmvs.cld_dir_rad))
    print(indices)
    print(cmvs.iloc[indices])




if __name__ == "__main__":
    main()