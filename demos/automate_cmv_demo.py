import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from solartoolbox import stats, spatial, cmv

from scipy.stats import linregress
from scipy.optimize import linear_sum_assignment, shgo

def main():
    fn = 'data/hope-melpitz-qcinterp.h5'
    avg_interval = '1h'
    day_start = '2013-09-08 00:00:00'
    day_stop = '2013-09-12 00:00:00'

    pos = pd.read_hdf(fn, mode="r", key="latlon")
    pos_utm = spatial.latlon2utm(pos['lat'], pos['lon'])
    ts = pd.read_hdf(fn, mode="r", key="data")


    ts = ts[np.logical_and(day_start<ts.index, ts.index<day_stop)]
    ts = ts.resample('10s').mean()

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

    subcmvs = optimize_spread(cmvs)
    print(subcmvs)


def optimize_spread(cmvs, n=10):
    """
    Chooses a diverse set of vectors given the full set of vectors. Operates
    in 2 quadrants only subject to the assumption that anti-parallels are also
    undesirable.

    Parameters
    ----------
    cmvs : pd.DataFrame
        The full set of CMVs.

    n : int
        The number of vectors to select

    Returns
    -------
    cmvs_subset : pd.DataFrame
        The subset of CMVs that are most diverse
    """

    # Compute unit vectors representing the CMVs
    cmvx, cmvy = spatial.pol2rect(cmvs.cld_spd, cmvs.cld_dir_rad)
    cmvx /= cmvs.cld_spd
    cmvy /= cmvs.cld_spd
    cmv_vecs = np.array([cmvx, cmvy]).T

    def calc_cost(ang_0):
        """
        Compute the cost function associated with using each CMV as a member of
        the set relative to the ideal set of equally spaced vectors.

        Parameters
        ----------
        ang_0 : the rotation angle of the ideal vectors in radians

        Returns
        -------
        cost : the total cost of the assignment
        indices : the indices of the CMVs in the optimal assignment
        """
        # Compute unit vectors equally distributed around 180 deg.
        ideal_angs = np.arange(0, n)/n * np.pi + ang_0
        ideal_vecs = np.array([np.cos(ideal_angs), np.sin(ideal_angs)]).T

        # Compute cost as dot products between each CMV and each ideal vector
        # Absolute value used because both parallel and anti-parallel are bad
        # for our case.
        dots = -np.array([np.abs(spatial.dot(cmv_vecs.T, ideal_vec))
                          for ideal_vec in ideal_vecs])

        # Compute the optimal assignment of CMVs to maximize alignment with the
        # ideal vectors. Relative cost could be used to compare multiple zero
        # angles.
        r_ind, c_ind = linear_sum_assignment(dots)
        cost = dots[r_ind, c_ind].sum()
        return cost, c_ind

    def cost_wrapper(ang_0):
        """
        The minimizer only works on a single output, so in this case we wrap it
        """
        return calc_cost(ang_0)[0]  # return only the cost

    # The bounds of the optimization are limited by the spacing between ideal
    # vectors.
    bounds = [(0, np.pi/n)]

    #  What we're optimizing here is a rotation angle for the set of ideal vecs
    y = shgo(cost_wrapper, bounds)

    # Use the optimized angle to figure out which CMVs to use
    final_cost, indices = calc_cost(y.x[0])
    cmvs_subset = cmvs.copy().iloc[indices]

    return cmvs_subset


if __name__ == "__main__":
    main()