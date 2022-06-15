import matplotlib
import numpy as np


def annotate_fig(axis):
    """
    Helper to annotate a figure axis with some significant frequency timescales
    Parameters
    ----------
    axis an axis object. Get via ax = plt.gca()

    Returns
    -------
    void
    """

    annfreqs = [1 / (60*60*24), 1 / (60*60*24), np.nan,
                1 / (60 * 60 * 12), 1 / (60 * 60 * 12), np.nan,
                1 / (60 * 60), 1 / (60 * 60), np.nan,
                1 / (30 * 60), 1 / (30 * 60), np.nan,
                1 / (10 * 60), 1 / (10 * 60), np.nan,
                1 / 60, 1 / 60, np.nan,
                1 / 30, 1 / 30, np.nan,
                1 / 15, 1 / 15, np.nan,
                1 / 5, 1 / 5, np.nan]
    specfreqnames = ["1d", "12h", "1h", "30m", "10m", "1m", "30s", "15s", "5s"]
    if axis.get_xscale == matplotlib.scale.LogScale and \
            axis.get_yscale == matplotlib.scale.LinearScale:
        axis.semilogx(annfreqs,
                      np.tile([0, 0.05, 0], len(annfreqs) // 3),
                      'k-',
                      linewidth=0.5)
    elif axis.get_xscale == matplotlib.scale.LogScale and \
            axis.get_yscale == matplotlib.scale.LogScale:
        axis.loglog(annfreqs,
                    np.tile([0, 0.05, 0], len(annfreqs) // 3),
                    'k-',
                    linewidth=0.5)
    else:
        axis.plot(annfreqs,
                  np.tile([0, 0.05, 0], len(annfreqs) // 3),
                  'k-',
                  linewidth=0.5)
    for (lbl, xpos) in zip(specfreqnames, annfreqs[::3]):  # Annotate
        axis.annotate(lbl,
                      (xpos, 0.05),
                      size=8,
                      rotation='vertical',
                      ha='center',
                      va='bottom')
