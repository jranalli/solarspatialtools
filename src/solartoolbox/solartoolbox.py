import pvlib
import numpy as np
import pandas as pd


def clearsky_index(ghi, clearsky_ghi, max_clearsky_index=2.0):
    """
    Wrapper for pvlib.irradiance.clearsky_index. Original doesn't handle some
    of the datatypes that we use here cleanly.

    The clearsky index is the ratio of global to clearsky global irradiance.
    Negative and non-finite clearsky index values will be truncated to zero.

    Parameters
    ----------
    ghi : numeric, pandas.Series, pandas.DataFrame
        Global horizontal irradiance in W/m^2.

    clearsky_ghi : numeric
        Modeled clearsky GHI

    max_clearsky_index : numeric, default 2.0
        Maximum value of the clearsky index. The default, 2.0, allows
        for over-irradiance events typically seen in sub-hourly data.

    Returns
    -------
    clearsky_index : numeric
        Clearsky index
    """

    if isinstance(ghi, pd.DataFrame):  # apply column by column
        return ghi.apply(lambda x:
                         pvlib.irradiance.clearsky_index(x, clearsky_ghi, 2),
                         axis=0)
    else:
        return pvlib.irradiance.clearsky_index(ghi, clearsky_ghi,
                                           max_clearsky_index)
