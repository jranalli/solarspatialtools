import numpy as np
from scipy.interpolate import RegularGridInterpolator

import matplotlib.pyplot as plt
from scipy.ndimage import sobel, uniform_filter

from solarspatialtools.stats import variability_score


def _random_at_scale(rand_size, final_size, plot=False):
    """
    Generate a 2D array of random. Generate initially at the size of rand_size
    and then linearly interpolate to the size of final_size.

    Parameters
    ----------
    rand_size : tuple
        The size of the random array to generate (rows, cols)
    final_size : tuple
        The size of the final array to return (rows, cols)

    Returns
    -------
    np.ndarray
        A 2D array of random values
    """

    # Generate random values at the scale of rand_size
    random = np.random.rand(rand_size[0], rand_size[1])

    # Linearly interpolate to the final size
    x = np.linspace(0, 1, rand_size[0])
    y = np.linspace(0, 1, rand_size[1])

    xnew = np.linspace(0, 1, final_size[0])
    ynew = np.linspace(0, 1, final_size[1])

    interp_f = RegularGridInterpolator((x, y), random, method='linear')
    Xnew, Ynew = np.meshgrid(xnew, ynew, indexing='ij')
    random_new = interp_f((Xnew, Ynew))



    if plot:
        # generate side by side subplots to compare
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(random, extent=(0, rand_size[1], 0, rand_size[0]))
        axs[0].set_title('Original Random')
        axs[1].imshow(random_new, extent=(0, final_size[1], 0, final_size[0]))
        axs[1].set_title('Interpolated Random')
        plt.show()

    return random, random_new


def _calc_vs_weights(scales, vs):
    """
    Calculate the weight for each scale

    Parameters
    ----------
    scales : int
        The space scale
    vs : float
        The variability score

    Returns
    -------

    """
    VS1 = -np.log(1-vs/180)/0.6
    weight = scales ** (1/VS1)
    return weight / np.sum(weight)


def _calc_wavelet_weights(waves):
    """
    Calculate the weights for each wavelet

    Parameters
    ----------
    waves : np.ndarray
        The wavelet coefficients

    Returns
    -------

    """
    scales = np.nanmean(waves**2, axis=1)
    return scales / np.sum(scales)



def space_to_time(pixres=1, cloud_speed=50):
    pixjump = cloud_speed / pixres
    # n space
    # dx space
    # velocity
    # dt = dx / velocity
    # max space = n * dx
    # max time = max space / velocity


def stacked_field(vs, size, weights=None, scales=(1, 2, 3, 4, 5, 6, 7), plot=False):

    field = np.zeros(size, dtype=float)

    if weights is None:
        weights = _calc_vs_weights(scales, vs)
    else:
        assert len(weights) == len(scales)

    for scale, weight in zip(scales, weights):
        prop = 2**(-scale+1)  # proportion for this scale
        _, i_field = _random_at_scale((int(size[0]*prop), int(size[1]*prop)), size)
        field += i_field * weight

    # Scale it zero to 1??
    field = (field - np.min(field))
    field = field / np.max(field)
    assert np.min(field) == 0
    assert np.max(field) == 1

    if plot:
        # Plot the field
        plt.imshow(field, extent=(0, size[1], 0, size[0]))
        plt.show()

    return field

def _clip_field(field, kt=0.5, plot=False):
    # Zero where clouds, 1 where clear

    # clipping needs to be based on pixel fraction, which thus needs to be
    # done on quantile because the field has a normal distribution
    quant = np.quantile(field, kt)

    # Find that quantile and cap it
    field_out = np.ones_like(field)
    field_out[field > quant] = 0

    assert (np.isclose(kt, np.sum(field_out) / field.size, rtol=1e-3))

    if plot:
        plt.imshow(field_out, extent=(0, field.shape[1], 0, field.shape[0]))
        plt.show()

    return field_out

def _find_edges(size, plot=False):
    edges = np.abs(sobel(out_field))
    smoothed = uniform_filter(edges, size=size)

    # We want to binarize it
    smoothed[smoothed < 1e-5] = 0  # Zero out the small floating point values
    # Calculate a threshold based on quantile, because otherwise we get the whole clouds
    baseline = np.quantile(smoothed[smoothed>0], 0.5)
    smoothed = smoothed > baseline

    if plot:
        # Compare the edges and uniform filtered edges side by side
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(edges, extent=(0, ysiz, 0, xsiz))
        axs[0].set_title('Edges')
        axs[1].imshow(smoothed, extent=(0, ysiz, 0, xsiz))
        axs[1].set_title('Uniform Filtered Edges')
        plt.show()

    return edges, smoothed

def shift_mean_lave(field, ktmean, max_overshoot=1.4, ktmin=0.2, min_quant=0.005, max_quant=0.995, plot=True):

    # ##### Shift values of kt to range from 0.2 - 1

    # Calc the "max" and "min", excluding clear values
    field_max = np.quantile(field[field < 1], max_quant)
    field_min = np.quantile(field[field < 1], min_quant)

    # Scale it between ktmin and max_overshoot
    field_out = (field - field_min) / (field_max - field_min) * (1-ktmin) + ktmin

    # # Clip limits to sensible boundaries
    field_out[field_out > 1] = 1
    field_out[field_out < 0] = 0

    # ##### Apply multiplier to shift mean to ktmean #####

    # Rescale the mean
    tgtsum = np.prod(np.shape(field_out)) * ktmean  # Mean scaled over whole field
    diff_sum = tgtsum - np.sum(field_out == 1)  # Shifting to exclude fully clear values
    tgt_mean = diff_sum / np.sum(field_out < 1)  # Recalculating the expected mean of the cloudy-only aareas
    current_cloud_mean = np.mean(field_out[field_out < 1]) # Actual cloud mean

    if diff_sum > 0:
        field_out[field_out!=1] = tgt_mean / current_cloud_mean * field_out[field_out!=1]

    # print(diff_sum)
    # print(current_cloud_mean)
    print(f"Desired Mean: {ktmean}, actual global mean {np.mean(field_out)}.")


    if plot:
        plt.hist(field_out[field_out<1].flatten(), bins=100)
        plt.show()

        # plot field and field_out side by side
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(field, extent=(0, ysiz, 0, xsiz))
        axs[0].set_title('Original Field')
        axs[1].imshow(field_out, extent=(0, ysiz, 0, xsiz))
        axs[1].set_title('Shifted Field')
        plt.show()
    return field_out


def lave_scaling_exact(field, ktmean, max_overshoot=1.4, ktmin=0.2, min_quant=0.005, max_quant=0.995, plot=True):

    # ##### Shift values of kt to range from 0.2 - 1

    # Calc the "max" and "min", excluding clear values
    field_min = np.quantile(field[field < 1], .99)

    # Scale it between ktmin and max_overshoot
    clouds3 = 1 - field*0.8/field_min


    # # Clip limits to sensible boundaries
    clouds3[clouds3 > 1] = 1
    clouds3[clouds3 < 0] = 0

    # ##### Apply multiplier to shift mean to ktmean #####
    mn = np.mean(clouds3)
    minmn = np.min(clouds3)/mn
    maxmn = np.max(clouds3/mn-minmn)

    ce = 1+ (clouds3/mn-minmn)/maxmn*(1.4-1)

    # Rescale the mean
    tgtsum = np.prod(np.shape(field_out)) * ktmean  # Mean scaled over whole field
    diff_sum = tgtsum - np.sum(field_out == 1)  # Shifting to exclude fully clear values
    tgt_mean = diff_sum / np.sum(field_out < 1)  # Recalculating the expected mean of the cloudy-only aareas
    current_cloud_mean = np.mean(field_out[field_out < 1]) # Actual cloud mean

    if diff_sum > 0:
        field_out[field_out!=1] = tgt_mean / current_cloud_mean * field_out[field_out!=1]

    # print(diff_sum)
    # print(current_cloud_mean)
    print(f"Desired Mean: {ktmean}, actual global mean {np.mean(field_out)}.")


    if plot:
        plt.hist(field_out[field_out<1].flatten(), bins=100)
        plt.show()

        # plot field and field_out side by side
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(field, extent=(0, ysiz, 0, xsiz))
        axs[0].set_title('Original Field')
        axs[1].imshow(field_out, extent=(0, ysiz, 0, xsiz))
        axs[1].set_title('Shifted Field')
        plt.show()
    return field_out


def get_settings_from_timeseries(kt_ts, plot=True):
    # Get the mean and standard deviation of the time series
    ktmean = np.mean(kt_ts)
    ktstd = np.std(kt_ts)
    ktmax = np.max(kt_ts)
    ktmin = np.min(kt_ts)

    # Get the fraction of clear sky
    frac_clear = np.sum(kt_ts > 0.95) / np.prod(np.shape(kt_ts))

    vs = variability_score(kt) * 1e4

    # Compute the wavelet weights
    # should be the mean(wavelet squared) for all modes except the steady mode
    waves, tmscales = pvlib.scaling._compute_wavelet(kt_ts)

    if plot:
        # create a plot where each of the timeseries in waves is aligned vertically in an individual subplot
        fig, axs = plt.subplots(len(waves), 1, figsize=(10, 2 * len(waves)), sharex=True)
        for i, wave in enumerate(waves):
            axs[i].plot(wave)
            axs[i].set_title(f'Wavelet {i+1}')
        plt.show()

    waves = waves[:-1, :]  # remove the steady mode
    tmscales = [i+1 for i, _ in enumerate(tmscales[:-1])]
    weights = _calc_wavelet_weights(waves)

    return ktmean, ktstd, ktmin, ktmax, frac_clear, vs, weights, tmscales




if __name__ == '__main__':

    import pandas as pd
    import pvlib

    datafn = "../../demos/data/hope_melpitz_1s.h5"
    twin = pd.date_range('2013-09-08 9:15:00', '2013-09-08 10:15:00', freq='1s', tz='UTC')
    data = pd.read_hdf(datafn, mode="r", key="data")
    data = data[40]
    plt.plot(data)
    plt.show()

    pos = pd.read_hdf(datafn, mode="r", key="latlon")
    loc = pvlib.location.Location(np.mean(pos['lat']), np.mean(pos['lon']))
    cs_ghi = loc.get_clearsky(data.index, model='simplified_solis')['ghi']
    cs_ghi = 1000/max(cs_ghi) * cs_ghi
    kt = pvlib.irradiance.clearsky_index(data, cs_ghi, 2)

    plt.plot(data)
    plt.plot(cs_ghi)
    plt.show()

    plt.plot(kt)
    plt.show()

    ktmean, ktstd, ktmin, ktmax, frac_clear, vs, weights, scales = get_settings_from_timeseries(kt, plot=False)

    print(f"Clear Fraction is: {frac_clear}")

    np.random.seed(42)  # seed it for repeatability

    xsiz = 2**12
    ysiz = 2**14

    cfield = stacked_field(vs, (xsiz, ysiz), weights, scales)

    mask_field = stacked_field(vs, (xsiz, ysiz), weights, scales)
    mask_field = _clip_field(mask_field, frac_clear, plot=False)

    # Clear Everywhere
    out_field = np.ones_like(cfield)
    # Where it's cloudy, mask in the clouds
    out_field[mask_field == 0] = cfield[mask_field == 0]

    # plt.imshow(out_field, extent=(0, ysiz, 0, xsiz))
    # plt.show()

    edges, smoothed = _find_edges(3)

    # field_final = shift_mean_lave(out_field, ktmean)
    lave_scaling_exact(out_field, ktmean)

    plt.plot(field_final[1,:])
    plt.show()

    # assert np.all(r == rnew)