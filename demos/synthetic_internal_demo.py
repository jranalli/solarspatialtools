import numpy as np
import matplotlib.pyplot as plt
from solarspatialtools.synthirrad.cloudfield import _stack_random_field, _calc_clear_mask, _find_edges, _scale_field_lave, _random_at_scale, _scale_field_basic


# It's worth looking in more detail at the internal processes of the cloud generation methodology to better understand
# what's happening. This all follows the method developed by Lave et al [1].
# [1] Matthew Lave, Matthew J. Reno, Robert J. Broderick, "Creation and Value of Synthetic High-Frequency Solar Inputs
# for Distribution System QSTS Simulations," 2017 IEEE 44th Photovoltaic Specialist Conference (PVSC), Washington, DC,
# USA, 2017, pp. 3031-3033, doi: https://dx.doi.org/10.1109/PVSC.2017.8366378.

# Cloud field relies on generating various scales of random noise and adding them together. The job of the function
# `_random_at_scale` is to generate a random field at a given scale and then interpolate it to a higher resolution. This
# function will be called at each level of the wavelet decomposition to generate the cloud field with different scaling
# factors.

np.random.seed(42)
layer, interp_layer = _random_at_scale((20, 20), (500, 500), plot=True)
layer, interp_layer = _random_at_scale((60, 60), (500, 500), plot=True)

# We'll now show two examples of `_stack_random_field` that will combine multiple such layers, stacked field across the
# multiple scales/weights. The first example will show the cloud field generated with a bias toward small scales (high
# variability) while the second will show a cloud field generated with predominantly large scales (low variability).
internal_size = (500, 500)
fine_weights = np.array([1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1])
coarse_weights = np.flipud(fine_weights)
scales = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

internal_cfield = _stack_random_field(fine_weights, scales, internal_size, plot=True)
internal_cfield = _stack_random_field(coarse_weights, scales, internal_size, plot=True)

# Directly using these fields isn't realistic, because they doesn't produce roughly normally distributed noise, rather
# than a cloudy/clear dichotomy like we would see in a real sky. So we will create an additional random field that will
# be used to create a mask. The reson we don't just mask the field directly is that we want to ensure that the
# variability isn't self-correlated to the shape of the clouds. First we will create a new random field, and then we
# will specify a fraction of clear sky conditions that we wish to have. Because the stacked field is not a uniform
# distribution, we will set a threshold level based on quantile within the field, as implemented by `_calc_clear_mask`.

frac_clear = 0.15
internal_clear_mask = _stack_random_field(coarse_weights, scales, internal_size)
internal_clear_mask = _calc_clear_mask(internal_clear_mask, frac_clear, plot=True)  # 0 is cloudy, 1 is clear


# Additionally, we will find edges in the clear mask. This will be used to create an additional edge mask that will be
# used to apply cloud enhancement at the edges of the cloudy regions. The `edgesmoothing` parameter will control an
# amount of smoothing that is applied to the edge mask, which helps to decide how wide the cloud edges should be. Edge
# smoothing essentially performs a dilation operation on the edge mask.
internal_edgesmoothing = 3
edges, smoothed = _find_edges(internal_clear_mask, internal_edgesmoothing, plot=True)


# We do some scaling to make the final field have the desired statistical properties. This happens in several steps.
# 1) Scale the field to be distributed from from kt1pct to the 99th percentile of the field, since we never see real
#    values of kt approach zero. Clip this distribution to [0, 1] just in case there are entries outside the bounds.
#    This transforms our field to represent the clear sky index ranging from the 1pct to 1.0, representing values of the
#    clear sky index.
# 2) Calculate a copy of this field and scale it to range from 1 to ktmax. This will be used to represent the cloud
#    enhancement.
# 3) Select only the regions of the final mask that will be cloudy (i.e. exclude the values of the field that are clear
#    sky (and thus have a value of 1) and the edge enhanced regions). Calculate a scaling factor such that the mean of
#    the cloudy region will lead to a global field mean of ktmean.
# 4) Apply the cloud enhancement field values to the regions assigned as edges.
# 5) Assign a value of 1 to the clear sky regions (note that the order of 4 & 5 ensures that the cloud enhancement stops
#    sharply at the cloud edges).

# The distribution will basically have three sub regions: clear sky (always 1), cloud enhancement edges (range from 1 to
# ktmax) and cloudy regions (range is scaled to produce desired ktmean). The overall mean of the field will be ktmean,
# and the maximum value will be close to ktmax. Note that kt1pct may not be respected in the final distribution, but
# instead follows the original author's implementation of the method.

ktmean = 0.6
kt1pct = 0.2
ktmax = 1.5

internal_field_final = _scale_field_lave(internal_cfield, internal_clear_mask, smoothed, ktmean, ktmax, kt1pct)
print(f"ktmean: {np.mean(internal_field_final):8.2f}")
print(f"kt1pct: {np.min(internal_field_final):8.2f}")
print(f"ktmax: {np.max(internal_field_final):8.2f}")

plt.hist(internal_field_final.flatten()[internal_field_final.flatten()<1], bins=100, alpha=0.5)
plt.hist(internal_field_final.flatten()[internal_field_final.flatten()==1], bins=100, alpha=0.5)
plt.hist(internal_field_final.flatten()[internal_field_final.flatten()>1], bins=100, alpha=0.5)
plt.legend(["Cloudy Area", "Clear Sky", "Edge Enhancement"])
plt.ylabel('Frequency')
plt.xlabel('kt')

plt.figure()
plt.imshow(internal_field_final.T, aspect='equal', cmap='viridis')
plt.colorbar()
plt.show()

# There's also an alternate scaling method available that contains a few additional options, including the ability to
# have the clearsky values reflect a distribution rather than uniform 1 values. This is implemented in the function
# `_scale_field_basic`.

# We compare using flipdistr=True on the basic method, because the lave method uses the flipping.
internal_field_final = _scale_field_lave(internal_cfield, internal_clear_mask, smoothed, 0.6, 1.2, 0.2)
internal_field_final2 = _scale_field_basic(internal_cfield, internal_clear_mask, smoothed, 0.6, 1.2, 0.2, flipdistr=True, cs_smoothing=0.8)

# Display a comparative histogram
plt.hist(internal_field_final.flatten(), bins=100, alpha=0.5, label='Original Method')
plt.hist(internal_field_final2.flatten(), bins=100, alpha=0.5, label='Basic Method')
plt.legend()
plt.show()

# Display the fields and the time series
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
axs[0,0].imshow(internal_field_final, extent=(0, internal_field_final.shape[1], 0, internal_field_final.shape[0]))
axs[0,0].set_title('Original Scaling')
axs[0,1].imshow(internal_field_final2, extent=(0, internal_field_final2.shape[1], 0, internal_field_final2.shape[0]))
axs[0,1].set_title('Modified Scaling')
# plot a time series from each
axs[1,0].plot(internal_field_final[0, :], label='Original Method')
axs[1,1].plot(internal_field_final2[0, :], label='Basic Method')
plt.show()

# compare the statistics of max and min from the two methods
print(f"Original Method: Max {internal_field_final.max():.2f}, Min {np.quantile(internal_field_final.flatten(), 0.01):.2f}, Mean {internal_field_final.mean():.2f}")
print(f"Basic Method: Max {internal_field_final2.max():.2f}, Min {np.quantile(internal_field_final2.flatten(), 0.01):.2f}, Mean {internal_field_final2.mean():.2f}")
