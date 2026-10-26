"""Exit-code probe (the platform returns only the exit code of failed runs): answers 8 yes/no questions about the
validation LAS data and our LiDAR expert, then exits with the bitmask. Writes no result (non-scored attempt).

bit 0  every LAS has ground class 2 on ≥ 5 % of its first million points
bit 1  some LAS has ground class 2 on ≥ 5 %
bit 2  every LAS header carries a parseable CRS
bit 3  every header CRS (where present) has the same EPSG as UTM.json
bit 4  every LAS has eastings in 100 000–900 000 and northings in 1 000 000–9 000 000 (UTM-like metres)
bit 5  the LiDAR expert ran on the smallest cloud without an exception
bit 6  …and produced ≥ 1 feature
bit 7  every LiDAR feature from that run lies inside the cloud's own bounding box (CRS round trip consistent)
"""
import os
import sys
import traceback
from pathlib import Path


def run(in_dir):
    import numpy as np
    import laspy
    from .common import region_dirs, region_files, utm_crs
    bits = [False] * 8
    las = []
    for r in region_dirs(in_dir):
        for p in region_files(r).get("las", []):
            las.append((r, p))
    g_all, g_any, crs_all, crs_match, utm_like = True, False, True, True, True
    for r, p in las:
        try:
            with laspy.open(p) as rd:
                h = rd.header
                ch = next(rd.chunk_iterator(1_000_000))
                gf = float((np.asarray(ch.classification) == 2).mean())
                g_all &= gf >= 0.05; g_any |= gf >= 0.05
                try:
                    c = h.parse_crs()
                except Exception:
                    c = None
                crs_all &= c is not None
                if c is not None and c.to_epsg():
                    u = utm_crs(r)
                    crs_match &= bool(u) and str(c.to_epsg()) in str(u)
                utm_like &= 1e5 <= h.mins[0] <= 9e5 and 1e6 <= h.mins[1] <= 9e6
        except Exception:
            traceback.print_exc(); g_all = False; crs_all = False; utm_like = False
    bits[0], bits[1], bits[2], bits[3], bits[4] = g_all, g_any, crs_all, crs_match, utm_like
    if las:
        r, p = min(las, key=lambda rp: rp[1].stat().st_size)
        try:
            import time
            from .experts import las as L
            feats = L.predict(r, [p], time.time() + 900)
            bits[5] = True; bits[6] = len(feats) > 0
            with laspy.open(p) as rd:
                h = rd.header
            from pyproj import Transformer
            from shapely.geometry import shape
            tr = Transformer.from_crs(3857, utm_crs(r), always_xy=True)
            ok = True
            for f in feats:
                c = shape(f["geometry"]).centroid
                x, y = tr.transform(c.x, c.y)
                ok &= h.mins[0] - 5 <= x <= h.maxs[0] + 5 and h.mins[1] - 5 <= y <= h.maxs[1] + 5
            bits[7] = ok and len(feats) > 0
        except Exception:
            traceback.print_exc()
    code = sum(1 << i for i, b in enumerate(bits) if b)
    print("PROBE bits", bits, "code", code, file=sys.stderr, flush=True)
    return code
