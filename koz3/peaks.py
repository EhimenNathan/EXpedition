"""scipy-only equivalent of skimage.feature.peak_local_max(image, min_distance, threshold_abs, exclude_border=False)
(the platform image has no scikit-image). Same steps: maximum-filter peak mask above the threshold, intensity-sorted,
then peaks closer than min_distance (Chebyshev) to a stronger kept peak are dropped."""
import numpy as np
from scipy import ndimage as ndi
from scipy.spatial import cKDTree


def peak_local_max(image, min_distance=1, threshold_abs=None, exclude_border=False):
    size = 2 * min_distance + 1
    mx = ndi.maximum_filter(image, size=size, mode="nearest")
    mask = image == mx
    thr = image.min() if threshold_abs is None else threshold_abs
    mask &= image > thr
    coords = np.column_stack(np.nonzero(mask))
    if len(coords) == 0:
        return np.zeros((0, 2), np.intp)
    order = np.argsort(-image[coords[:, 0], coords[:, 1]], kind="stable")
    coords = coords[order]
    if len(coords) > 1:
        tree = cKDTree(coords)
        keep = np.ones(len(coords), bool)
        for i, nb in enumerate(tree.query_ball_point(coords, r=min_distance, p=np.inf)):
            if keep[i]:
                for j in nb:
                    if j > i:
                        keep[j] = False
        coords = coords[keep]
    return coords
