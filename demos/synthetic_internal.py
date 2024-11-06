import numpy as np
import matplotlib.pyplot as plt
from solarspatialtools.synthirrad.cloudfield import _stack_random_field, _calc_clear_mask, _find_edges, _scale_field

np.random.seed(42)
internal_size = (500, 500)
weights = np.flipud(np.array([1, 1, 1, 1, 1, 0.1, 0.1, 0.1, 0.1, 0.1]))
weights/=weights.sum()
scales = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

internal_cfield = _stack_random_field(weights, scales, internal_size, plot=False)

frac_clear = 0.15
internal_clear_mask = _stack_random_field(weights, scales, internal_size)
internal_clear_mask = _calc_clear_mask(internal_clear_mask, frac_clear, plot=False)

internal_edgesmoothing = 3
edges, smoothed = _find_edges(internal_clear_mask, internal_edgesmoothing, binarize_threshold=0, plot=False)

internal_field_final = _scale_field(internal_cfield, internal_clear_mask, smoothed, 0.5, 1.2, 0.2, method='original', plot=True)
internal_field_final2 = _scale_field(internal_cfield, internal_clear_mask, smoothed, 0.5, 1.2, 0.2, method='basic', plot=True)

# compare the statistics of max and min from the two methods

print(f"Original Method: Max {internal_field_final.max():.2f}, Min {np.quantile(internal_field_final.flatten(), 0.01):.2f}, Mean {internal_field_final.mean():.2f}")
print(f"Basic Method: Max {internal_field_final2.max():.2f}, Min {np.quantile(internal_field_final2.flatten(), 0.01):.2f}, Mean {internal_field_final2.mean():.2f}")
