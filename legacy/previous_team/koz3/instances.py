"""Probability maps -> scored polygons.
Class masks give the extent; the centre heatmap gives one seed per object, so touching kurgans are split by a
marker-controlled watershed (the archaeologists label each mound separately)."""
import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from rasterio import features
from rasterio.transform import Affine
from shapely.geometry import shape

MIN_AREA = {"курганы_целые": 10.0, "курганы_поврежденные": 10.0, "городища": 1000.0, "фортификации": 20.0, "архитектуры": 4.0}
# hillforts: smallest described is ~40x50 m (class description); fragments below 1000 m2 are noise
SPLIT = {"курганы_целые", "курганы_поврежденные"}  # classes whose touching instances are separated


def extract(prob, heat, transform, classes, thr=0.5, peak_thr=0.3, min_dist_m=2.0, gsd=0.25):
    """prob: (C,H,W) class probabilities; heat: (H,W) centre heatmap or None. Returns list of (cls, polygon, score)."""
    out = []
    T = transform if isinstance(transform, Affine) else Affine(*transform)
    for ci, c in enumerate(classes):
        p = prob[ci]
        m = p >= thr
        if not m.any():
            continue
        if c in SPLIT and heat is not None:
            pk = peak_local_max(heat * m, min_distance=max(1, int(min_dist_m / gsd)), threshold_abs=peak_thr, labels=ndi.label(m)[0])
            markers = np.zeros(m.shape, np.int32)
            for k, (r, cc) in enumerate(pk, 1): markers[r, cc] = k
            lab = watershed(-heat, markers, mask=m) if len(pk) else ndi.label(m)[0]
            # components with no peak keep their own label
            rest = ndi.label(m & (lab == 0))[0]; lab = np.where(rest > 0, rest + lab.max(), lab)
        else:
            lab = ndi.label(m)[0]
        if lab.max() == 0:
            continue
        idx = np.arange(1, lab.max() + 1)
        score_max = ndi.maximum(p, lab, idx); score_mean = ndi.mean(p, lab, idx)
        for geom, v in features.shapes(lab.astype(np.int32), mask=lab > 0, transform=T):
            poly = shape(geom)
            if poly.area < MIN_AREA.get(c, 1.0):
                continue
            k = int(v) - 1
            out.append((c, poly.buffer(0), float(0.5 * score_max[k] + 0.5 * score_mean[k])))
    return out
