import numpy as np
from scipy.interpolate import RegularGridInterpolator
# from scipy.ndimage import map_coordinates

import matplotlib.pyplot as plt
from scipy.ndimage import sobel, uniform_filter

from solarspatialtools.stats import variability_score


def _random_at_scale(rand_size, final_size, plot=False):
    """
    Generate a 2D array of random at an initial size and linearly interpolate up to a larger final size.

    Parameters
    ----------
    rand_size : tuple
        The size of the random field
    final_size : tuple
        The size of the final interpolated field
    plot : bool
        Whether to plot the results

    Returns
    -------
    random : np.ndarray
        The original random field
    random_new : np.ndarray
        The interpolated random field
    """

    # Generate random values at the scale of rand_size
    random = np.random.rand(rand_size[0], rand_size[1])

    # Linearly interpolate to the final size
    x = np.linspace(0, 1, rand_size[0])
    y = np.linspace(0, 1, rand_size[1])

    xnew = np.linspace(0, 1, final_size[0])
    ynew = np.linspace(0, 1, final_size[1])

    # # Latest Recommended Scipy Method
    interp_f = RegularGridInterpolator((x, y), random, method='linear')
    Xnew, Ynew = np.meshgrid(xnew, ynew, indexing='ij')
    random_new = interp_f((Xnew, Ynew))

    # # # Alternate Scipy Method
    # interp_f = RectBivariateSpline(x, y, random, kx=1, ky=1)
    # # interp_ft = lambda xnew, ynew: interp_f(xnew, ynew).T
    # random_new = interp_f(xnew, ynew)

    # # # Different potentially faster Scipy Method
    # Xnew, Ynew = np.meshgrid(xnew, ynew, indexing='ij')
    # Xnew = Xnew*len(x)
    # Ynew = Ynew*len(y)
    # random_new = map_coordinates(random, [Xnew.ravel(), Ynew.ravel()], order=1, mode='nearest').reshape(final_size)

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
    Calculate the weights for each scale based on the variability score

    Parameters
    ----------
    scales : np.ndarray
        The scales of the wavelets
    vs : float
        The variability score

    Returns
    -------
    weights : np.ndarray
        The weights for each scale
    """

    if vs < 0:
        raise ValueError("VS must be greater than 0.")

    VS1 = -np.log(1-vs/180)/0.6
    weight = scales ** (1/VS1)
    return weight / np.sum(weight)


def _calc_wavelet_weights(waves):
    """
    Calculate the weights for each wavelet scale based on the wavelet values.

    Parameters
    ----------
    waves : np.ndarray
        The wavelet values. Should be the timeseries for each mode. Consider
        generating from `pvlib.scaling._compute_wavelet`.

    Returns
    -------
    weights : np.ndarray
        The weights for each wavelet scale
    """
    scales = np.nanmean(waves**2, axis=1)
    return scales / np.sum(scales)


def _stack_random_field(weights, scales, size, normalize=False, plot=False):
    """
    Generate a field of random clouds at different scales and weights based on Perlin Noise.
    This runs relatively slowly for large arrays.

    Parameters
    ----------
    weights : np.ndarray
        The weights for each scale. See _calc_wavelet_weights or _calc_vs_weights.
        Should add to unity.
    scales : np.ndarray
        The scales for each weight. Should be the same length as weights.
    size : tuple
        The size of the field
    normalize : bool
        Whether to normalize the field to 0-1
    plot : bool
        Whether to plot the field

    Returns
    -------
    field : np.ndarray
        The field of random clouds
    """

    if len(weights) != len(scales):
        raise ValueError("Number of weights must match scales.")

    # Calculate a field for each scale, and add them together in a weighted manner to form the field
    field = np.zeros(size, dtype=float)

    for scale, weight in zip(scales, weights):
        prop = 2.0**(-scale+1)  # proportion for this scale
        xsz = np.max([int(size[0]*prop),2])  # min of 2 so that we never go past the limit on interp
        ysz = np.max([int(size[1]*prop),2])
        _, i_field = _random_at_scale((xsz, ysz), size)
        field += i_field * weight

    # Optionally Scale it zero to 1
    if normalize:
        field = (field - np.min(field))
        field = field / np.max(field)
        assert np.min(field) == 0
        assert np.max(field) == 1

    if plot:
        # Plot the field
        plt.imshow(field, extent=(0, size[1], 0, size[0]))
        plt.show()

    return field

def _calc_clear_mask(field, clear_frac=0.5, plot=False):
    """
    Find the value in the field that will produce an X% clear sky mask. The
    mask is 1 where clear, and 0 where cloudy.

    Parameters
    ----------
    field : np.ndarray
        The field of clouds
    clear_frac : float
        The fraction of the field that should be clear
    plot : bool
        Whether to plot the field

    Returns
    -------
    field_mask : np.ndarray
        The clear sky mask. Zero indicates cloudy, one indicates clear
    """
    if clear_frac > 1 or clear_frac < 0:
        raise ValueError("Clear fraction must be between 0 and 1.")

    # Zero where clouds, 1 where clear

    if clear_frac == 0:
        field_mask = np.zeros_like(field)

    else:
        # clipping needs to be based on pixel fraction, which thus needs to be
        # done on quantile because the field has a normal distribution
        quant = np.quantile(field, clear_frac)

        # Find that quantile and cap it
        field_mask = np.ones_like(field)
        field_mask[field > quant] = 0

    # Test to make sure that we're close to matching the desired fraction
    assert (np.isclose(clear_frac, np.sum(field_mask) / field.size, rtol=1e-3))

    if plot:
        plt.imshow(field_mask, extent=(0, field.shape[1], 0, field.shape[0]))
        plt.show()

    return field_mask

def _find_edges(base_mask, size, plot=False):
    """
    Find the edges of a mask using a Sobel filter and then broadens it with a uniform filter.

    Parameters
    ----------
    base_mask : np.ndarray
        The mask to find the edges of
    size : int
        The size of the uniform filter (effectively doubles the size of the filter.

    Returns
    -------
    edges : np.ndarray
        The edges of the mask
    smoothed_binary : np.ndarray
        The smoothed binary mask
    """

    # This gets us roughly 50% overlapping with mask and 50% outside
    edges = np.abs(sobel(base_mask,0)**2+sobel(base_mask,1)**2)
    smoothed = uniform_filter(edges, size=size)

    # We want to binarize it
    smoothed[smoothed < 1e-5] = 0  # Zero out the small floating point values
    # Calculate a threshold based on quantile, because otherwise we get the whole clouds
    smoothed_binary = smoothed > 0

    if plot:
        xsiz, ysiz = base_mask.shape
        # Compare the edges and uniform filtered edges side by side
        fig, axs = plt.subplots(1, 3, figsize=(10, 5))
        axs[0].imshow(edges, extent=(0, ysiz, 0, xsiz))
        axs[0].set_title('Edges')
        axs[1].imshow(smoothed_binary, extent=(0, ysiz, 0, xsiz))
        axs[1].set_title('Uniform Filtered Edges')
        axs[2].imshow(base_mask, extent=(0, ysiz, 0, xsiz))
        axs[2].set_title('Original Mask')
        plt.show()

    return edges, smoothed_binary


def _scale_field_lave(field, clear_mask, edge_mask, ktmean, ktmax=1.4, kt1pct=0.2, max_quant=0.99, plot=False):
    """
    Scale a field of clouds to match a desired mean and distribution of kt values.

    Parameters
    ----------
    field : np.ndarray
        The field of clouds
    clear_mask : np.ndarray
        The clear sky mask. Zero indicates cloudy, one indicates clear
    edge_mask : np.ndarray
        The edge mask. Zero indicates not an edge, one indicates an edge
    ktmean : float
        The desired mean of the kt values
    ktmax : float
        The maximum of the kt values
    kt1pct : float
        The 1st percentile of the kt values
    max_quant : float
        The quantile to use for the maximum of the field
    plot : bool
        Whether to plot the results

    Returns
    -------
    clouds5 : np.ndarray
        The scaled field of clouds
    """


    # ##### Shift values of kt to range from 0.2 - 1

    # Calc the "max" and "min", excluding clear values
    field_max = np.quantile(field[clear_mask == 0], max_quant)

    # Create a flipped version of the distribution that scales between slightly below kt1pct and bascially (1-field_min)
    # I think the intent here would be to make it vary between kt1pct and 1, but that's not quite what it does.
    clouds3 = 1 - field*(1-kt1pct)/field_max

    # # Clip limits to sensible boundaries
    clouds3[clouds3 > 1] = 1
    clouds3[clouds3 < 0] = 0

    # ##### Apply multiplier to shift mean to ktmean #####
    mean_c3 = np.mean(clouds3)
    nmin_c3 = np.min(clouds3)/mean_c3
    nrange_c3 = np.max(clouds3)/mean_c3-nmin_c3
    ce = 1+ (clouds3/mean_c3-nmin_c3)/nrange_c3*(ktmax-1)

    # Rescale one more time to make the mean of clouds3 match the ktmean from the timeseries
    try:
        cloud_mask = np.bitwise_or(clear_mask>0, edge_mask) == 0  # Where is it neither clear nor edge
    except TypeError:
        cloud_mask = np.bitwise_or(clear_mask>0, edge_mask > 0) == 0  # Where is it neither clear nor edge
    tgtsum = field.size * ktmean  # Mean scaled over whole field
    diff_sum = tgtsum - np.sum(clear_mask) - np.sum(ce[np.bitwise_and(edge_mask > 0, clear_mask==0)])  # Shifting target to exclude fully clear values and the cloud enhancement
    tgt_cloud_mean = diff_sum / np.sum(cloud_mask)  # Find average required in areas where it's neither cloud nor edge
    current_cloud_mean = np.mean(clouds3[cloud_mask]) # Actual cloud mean in field

    if diff_sum > 0:
        clouds4 = tgt_cloud_mean / current_cloud_mean * clouds3
    else:
        clouds4 = clouds3.copy()

    clouds5 = clouds4.copy()

    # Edges then clear means that the clearsky overrides the edge enhancement
    clouds5[edge_mask > 0] = ce[edge_mask > 0]
    clouds5[clear_mask > 0] = 1

    if plot:
        plt.hist(ce.flatten(), bins=100)
        plt.hist(clouds3.flatten(), bins=100, alpha=0.5)
        plt.hist(clouds4.flatten(), bins=100, alpha=0.5)
        plt.hist(clouds5.flatten(), bins=100, alpha=0.5)
        plt.hist(field.flatten(), bins=100, alpha=0.5)
        plt.legend(["Cloud Enhancement", "1st Scaled Cloud Distribution", "2nd Scaled Cloud Distribution", "Fully Remapped Distribution",
                    "Original Field Distribution"])

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(field, extent=(0, field.shape[0], 0, field.shape[1]))
        axs[0].set_title('Original Field')
        axs[1].imshow(clouds5, extent=(0, field.shape[0], 0, field.shape[1]))
        axs[1].set_title('Shifted Field')
        plt.show()

    return clouds5


def get_timeseries_stats(kt_ts, clear_threshold=0.95, plot=True):
    """
    Get the properties of a time series of kt values.

    Parameters
    ----------
    kt_ts : pandas.Series
        The time series of kt values
    clear_threshold : float
        The threshold in kt for what is considered clear
    plot : bool
        Whether to plot the wavelets

    Returns
    -------
    ktmean : float
        The mean of the time series
    kt_1_pct : float
        The 1st percentile of the time series
    ktmax : float
        The maximum of the time series
    frac_clear : float
        The fraction of clear sky
    vs : float
        The variability score
    weights : np.ndarray
        The wavelet weights
    scales : list
        The timescales of the wavelets
    """
    import pvlib

    # Get the mean and standard deviation of the time series
    ktmean = np.mean(kt_ts)  # represents mean of kt
    ktstd = np.std(kt_ts)
    ktmax = np.max(kt_ts)  # represents peak cloud enhancement
    ktmin = np.min(kt_ts)

    kt_1_pct = np.nanquantile(kt_ts, 0.01)  # represents "lowest" kt

    # Get the fraction of clear sky with a threshold
    frac_clear = np.sum(kt_ts > clear_threshold) / kt_ts.size

    vs = variability_score(kt_ts) * 1e4

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
    scales = [i+1 for i, _ in enumerate(tmscales[:-1])]
    weights = _calc_wavelet_weights(waves)

    return ktmean, kt_1_pct, ktmax, frac_clear, vs, weights, scales


def space_to_time(pixres=1, cloud_speed=50):

    # Existing code uses 1 pixel per meter, and divides by cloud speed to get X size
    # does 3600 pixels, but assumes the clouds move by one full timestep.

    # ECEF coordinate system (Earth Centered Earth Fixed) is used to convert lat/lon to x/y.
    # This seems really weird. https://en.wikipedia.org/wiki/Geographic_coordinate_conversion

    # [X, Y] = geod2ecef(Lat_to_Sim, Lon_to_Sim, zeros(size(Lat_to_Sim)));
    # ynorm1 = Y - mean(Y);
    # xnorm1 = X - mean(X);
    # ynorm=round((ynorm1-min(ynorm1))./Cloud_Spd(qq))+1;
    # xnorm=round((xnorm1-min(xnorm1))./Cloud_Spd(qq))+1;
    # Xsize=60*60+max(xnorm);
    # Ysize=max(ynorm);
    # Xsize and Ysize are the pixel sizes generated

    # Extracting a time series has us loop through the entire size of X pixels, and choose a window 3600 pixels wide, and multiply by GHI_cs
    # GHI_syn(i,hour(datetime2(GHI_timestamp))==h1)=clouds_new{h1}(ynorm(i),xnorm(i):xnorm(i)+3600-1)'.*GHI_clrsky(hour(datetime2(GHI_timestamp))==h1);

    pixjump = cloud_speed / pixres
    # n space
    # dx space
    # velocity
    # dt = dx / velocity
    # max space = n * dx
    # max time = max space / velocity



def cloudfield_timeseries(weights, scales, size, frac_clear, ktmean, ktmax, kt1pct):
    """
    Generate a time series of cloud fields based on the properties of a time series of kt values.

    Parameters
    ----------
    weights : np.ndarray
        The wavelet weights at each scale
    scales : list
        The scales of the wavelets, should be integer values interpreted as 2**(scale-1) seconds
    size : tuple
        The size of the field to generate, x by y
    frac_clear : float
        The fraction of clear sky
    ktmean : float
        The mean of the kt values
    ktmax : float
        The maximum of the kt values
    kt1pct : float
        The 1st percentile of the kt values

    Returns
    -------
    field_final : np.ndarray
        The final field of simulated clouds
    """
    cfield = _stack_random_field(weights, scales, size)
    clear_mask = _stack_random_field(weights, scales, size)
    clear_mask = _calc_clear_mask(clear_mask, frac_clear)  # 0 is cloudy, 1 is clear

    edges, smoothed = _find_edges(clear_mask, 3)

    field_final = _scale_field_lave(cfield, clear_mask, smoothed, ktmean, ktmax, kt1pct, plot=True)
    return field_final


if __name__ == '__main__':

    import pandas as pd
    import pvlib
    from solarspatialtools import irradiance
    from solarspatialtools import cmv
    from solarspatialtools import spatial
    np.random.seed(42)  # seed it for repeatability


    # #### Load Timeseries Data

    datafn = "../../../demos/data/hope_melpitz_1s.h5"
    twin = pd.date_range('2013-09-08 9:15:00', '2013-09-08 10:15:00', freq='1s', tz='UTC')
    data = pd.read_hdf(datafn, mode="r", key="data")
    data_i = data[40]
    # plt.plot(data)
    # plt.show()

    # Get the time series for a single sensor and convert it to a clear sky index.
    # Record some statistics about it.
    pos = pd.read_hdf(datafn, mode="r", key="latlon")
    loc = pvlib.location.Location(np.mean(pos['lat']), np.mean(pos['lon']))
    cs_ghi = loc.get_clearsky(data_i.index, model='simplified_solis')['ghi']
    cs_ghi = 1000/max(cs_ghi) * cs_ghi  # Rescale (possible scaling on
    kt = pvlib.irradiance.clearsky_index(data_i, cs_ghi, 2)

    ktmean, kt1pct, ktmax, frac_clear, vs, weights, scales = get_timeseries_stats(
        kt, plot=False)

    # Get the Cloud Motion Vector for the Timeseries
    pos_utm = pd.read_hdf(datafn, mode="r", key="utm")
    kt_all = irradiance.clearsky_index(data, cs_ghi, 2)
    cld_spd, cld_dir, _ = cmv.compute_cmv(kt_all, pos_utm, reference_id=None, method='jamaly')
    cld_vec_rect = spatial.pol2rect(cld_spd, cld_dir)

    print(f"Cld Speed  {cld_spd:8.2f}, Cld Dir {np.rad2deg(cld_dir):8.2f}")

    # Rotate the sensor positions by -cld dir to position the incoming clouds
    # toward the upwind side of the plant. Shift to zero out the minimum value.
    rot = spatial.rotate_vector((pos_utm['E'], pos_utm['N']), theta=-cld_dir)
    pos_utm_rot = pd.DataFrame({'X': rot[0] - np.min(rot[0]),
                                'Y': rot[1] - np.min(rot[1])},
                               index=pos_utm.index)

    # # plot the original field and the rotated field side by side in two subplots
    # fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    # axs[0].scatter(pos_utm['E'], pos_utm['N'])
    # axs[0].set_title('Original Field')
    # axs[0].quiver(pos_utm['E'][40], pos_utm['N'][40], 200 * cld_vec_rect[0], 200 * cld_vec_rect[1], scale=10, scale_units='xy')
    # axs[0].set_xlabel('East')
    # axs[0].set_ylabel('North')
    # axs[1].scatter(pos_utm_rot['X'], pos_utm_rot['Y'])
    # axs[1].quiver(pos_utm_rot['X'][40], pos_utm_rot['Y'][40], 200*cld_spd, 0, scale=10, scale_units='xy')
    # axs[1].set_title('Rotated Field')
    # axs[0].set_xlabel('CMV Direction')
    # axs[0].set_ylabel('CMV Dir + 90 deg')
    # axs[0].set_aspect('equal')
    # axs[1].set_aspect('equal')
    # plt.show()


    # #### Generate the Simulated Cloud Field

    # Calculate the size of the field
    x_extent = np.abs(np.max(pos_utm_rot['X']) - np.min(pos_utm_rot['X']))
    y_extent = np.abs(np.max(pos_utm_rot['Y']) - np.min(pos_utm_rot['Y']))
    t_extent = (np.max(twin) - np.min(twin)).total_seconds()
    dt = (twin[1] - twin[0]).total_seconds()

    # Convert space to time
    spatial_time_x = x_extent / cld_spd
    spatial_time_y = y_extent / cld_spd

    # This now represents the time to space relationship in seconds, so each pixel of the field represents a 1 second step.
    # Our steps in X represent 1 second forward or backward in EITHER along-cloud space or time
    # Our steps in Y represent 1 "cloud second" left or right perpendicular to the motion axis
    # We actually have to oversize things a bit because if the field is too small, we can't
    # halve its size a sufficient number of times.
    # TODO rethink this one on the large scales side, are we interpolating for no reason?
    xt_size = np.max([int(np.ceil(spatial_time_x + t_extent)), 2**len(scales)])
    # yt_size = np.max([int(np.ceil(spatial_time_y)), 2**len(scales)])
    # xt_size = int(np.ceil(spatial_time_x + t_extent))
    yt_size = int(np.ceil(spatial_time_y))

    # Calculate the randomized field
    field_final = cloudfield_timeseries(weights, scales, (xt_size, yt_size), frac_clear, ktmean, ktmax, kt1pct)

    # Plot a timeseries
    plt.plot(field_final[1,:])
    plt.show()


    # Convert space to time to extract time series
    xpos = pos_utm_rot['X'] - np.min(pos_utm_rot['X'])
    ypos = pos_utm_rot['Y'] - np.min(pos_utm_rot['Y'])
    xpos_temporal = xpos / cld_spd
    ypos_temporal = ypos / cld_spd

    sim_kt = pd.DataFrame(index=twin, columns=pos_utm_rot.index)
    for sensor in pos_utm_rot.index:
        x = int(xpos_temporal[sensor])
        y = int(ypos_temporal[sensor])
        sim_kt[sensor] = field_final[x:x+int(t_extent)+1, y]

    plt.plot(sim_kt[[40,42]])
    plt.show()



    # Compare Hist of CS Index
    plt.hist(kt, bins=50)
    plt.hist(field_final[:,1], bins=50, alpha=0.5)

    # Ramp Rate
    plt.figure()
    plt.hist(np.diff(kt), bins=50)
    plt.hist(np.diff(field_final[:,1]), bins=200, alpha=0.5)
    plt.show()
