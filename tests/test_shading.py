"""koz3.shading: a synthetic sun-lit mound points to the sun; a pit points away; a scene with a common azimuth has R≈1."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from koz3.shading import dipole, agreement  # noqa: E402


def lit(h, az_deg, elev_deg=35.0):
    """Lambertian shading of height field h (rows southwards) for sun azimuth az (clockwise from north)."""
    gy, gx = np.gradient(h)                          # gy along rows (south), gx along cols (east)
    nx, ny, nz = -gx, gy, np.ones_like(h)            # normal in (east, north, up)
    a, e = np.radians(az_deg), np.radians(elev_deg)
    s = np.array([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)])
    return (nx * s[0] + ny * s[1] + nz * s[2]) / np.sqrt(nx ** 2 + ny ** 2 + nz ** 2)


if __name__ == "__main__":
    yy, xx = np.mgrid[0:200, 0:400]
    centres = [(100, 60), (100, 160), (100, 260)]
    h = sum(3.0 * np.exp(-((yy - r) ** 2 + (xx - c) ** 2) / (2 * 8.0 ** 2)) for r, c in centres)
    h -= 3.0 * np.exp(-((yy - 100) ** 2 + (xx - 350) ** 2) / (2 * 8.0 ** 2))      # a pit
    img = lit(h, az_deg=135.0)                                                    # sun in the south-east
    v = dipole(img, [100] * 4, [60, 160, 260, 350], 16)
    ang = np.degrees(np.arctan2(v[:, 0], v[:, 1])) % 360                          # azimuth of each dipole
    assert all(abs(a - 135) < 15 for a in ang[:3]), ang
    assert abs(((ang[3] - 315 + 180) % 360) - 180) < 15, ang                       # pit: opposite direction
    cos, R, _ = agreement(v, np.array([0.9, 0.9, 0.9, 0.9]), s_min=0.5)
    assert (cos[:3] > 0.9).all() and cos[3] < 0 and (R[:3] > 0.3).all(), (cos, R)
    print("PASS: mounds point to the sun (135°), the pit points away; LOO agreement separates them",
          np.round(ang, 1), np.round(cos, 2))
