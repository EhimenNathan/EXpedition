"""CPU-only historical-aerial style augmentation and train-only style audit.

This does not claim detection improvement. The optical transform is stochastic
but geometry-preserving, so image boxes remain valid. All style parameters are
chosen from training sites of a geographic fold; held-out Ae pixels/labels are
never used for the profile choice.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from pyproj import Geod, Transformer

ROOT = Path('/home/nathan/Expedition Competition')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from koz3.sp_train import load_chips

GEOD = Geod(ellps='WGS84')


def lonlat_center(row):
    a, b, x0, d, e, y0 = row['transform']
    xx, yy = x0 + a*row['w']/2 + b*row['h']/2, y0 + d*row['w']/2 + e*row['h']/2
    return Transformer.from_crs(row['crs'], 4326, always_xy=True).transform(xx, yy)


def crop_center(im, x, y, side=256):
    """Fixed-size crop; caller retains the affine box transform if training."""
    h, w = im.shape[:2]
    x0, y0 = int(round(x - side / 2)), int(round(y - side / 2))
    if x0 < 0 or y0 < 0 or x0 + side > w or y0 + side > h:
        return None
    out = np.zeros((side, side, 3), np.uint8)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(w, x0 + side), min(h, y0 + side)
    if xa < xb and ya < yb:
        out[ya - y0:yb - y0, xa - x0:xb - x0] = im[ya:yb, xa:xb]
    return out


def shad(bgr, rng, target_p02, target_p98, strength=1.0):
    """Photometric film simulation; input/output dimensions and boxes agree."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    p02, p98 = np.percentile(gray, (2, 98))
    desired_p02 = np.clip(target_p02 + rng.normal(0, .025), 0, 1)
    desired_p98 = np.clip(target_p98 + rng.normal(0, .025), 0, 1)
    gain = np.clip((desired_p98-desired_p02) / max(p98-p02, .08), .5, 2.0)
    gray = np.clip((gray-p02)*gain + desired_p02, 0, 1)
    # Film tone and exposure, with identity branch elsewhere in training.
    gamma = rng.uniform(.75, 1.4)
    black = rng.uniform(0, .06) * strength
    gray = np.clip((gray - black) / max(1 - black, .5), 0, 1) ** gamma
    if rng.random() < .9 * strength:
        sigma = rng.uniform(.8, 2.0) * strength
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=max(.1, sigma))
    if rng.random() < .3 * strength:
        length = int(rng.integers(3, 12)) | 1
        angle = rng.uniform(0, np.pi)
        ker = np.zeros((length, length), np.float32)
        cx = (length - 1) / 2
        dx, dy = cx * np.cos(angle), cx * np.sin(angle)
        cv2.line(ker, (int(cx-dx), int(cx-dy)), (int(cx+dx), int(cx+dy)), 1, 1)
        ker /= max(ker.sum(), 1)
        gray = cv2.filter2D(gray, -1, ker)
    h, w = gray.shape
    yy, xx = np.mgrid[:h, :w]
    radius = ((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2
    gray *= 1 - rng.uniform(.02, .22) * strength * np.clip(radius, 0, 1)
    gray += rng.normal(0, rng.uniform(.002, .012) * strength, gray.shape).astype(np.float32)
    # Sparse scratches/dust are restrained: dense artifacts would teach FPs.
    layer = np.zeros_like(gray)
    if rng.random() < .3 * strength:
        for _ in range(int(rng.integers(1, 4))):
            x = int(rng.integers(0, w))
            cv2.line(layer, (x, 0), (int(np.clip(x + rng.normal(0, 8), 0, w-1)), h-1),
                     float(rng.choice([-.25, .25])), 1)
    gray = np.clip(gray + layer, 0, 1)
    # Match only robust global film tone. The spatial texture remains free to
    # vary and needs a detector test before this can be called useful.
    q02, q98 = np.percentile(gray, (2, 98))
    scale = np.clip((desired_p98-desired_p02)/max(q98-q02, .05), .5, 2.0)
    gray = np.clip((gray-q02)*scale + desired_p02, 0, 1)
    quality = int(rng.integers(55, 96))
    ok, enc = cv2.imencode('.jpg', (gray * 255).astype(np.uint8),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    assert ok
    return cv2.imdecode(enc, cv2.IMREAD_GRAYSCALE)


def signature(gray):
    g = gray.astype(np.float32)
    p2, p98 = np.percentile(g, (2, 98))
    dx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    dy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.hypot(dx, dy)
    lap = cv2.Laplacian(g, cv2.CV_32F)
    return np.array([p2, p98, p98-p2, np.median(grad), np.percentile(grad, 90),
                     np.median(np.abs(lap))], np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--n', type=int, default=200)
    args = ap.parse_args()
    out = HERE / 'shad_audit'
    out.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261025 + args.fold)
    all_ae = json.loads((ROOT / 'data/ae_prep/ann.json').read_text())
    ae = [r for r in all_ae if int(r['fold']) != args.fold]
    ae_sites = {r['region'] for r in ae}
    held_ae_sites = {r['region'] for r in all_ae
                     if int(r['fold']) == args.fold}
    assert ae_sites.isdisjoint(held_ae_sites)
    hold = [r for r in all_ae if int(r['fold']) == args.fold]
    hold_ll = np.array([lonlat_center(r) for r in hold], dtype=float)
    sp0 = [r for r in load_chips('SpOr') if int(r['fold']) != args.fold and r['objs']]
    sp = []
    for r in sp0:
        if r['region'] in held_ae_sites: continue
        lon, lat = lonlat_center(r)
        _, _, dist = GEOD.inv(np.full(len(hold_ll), lon), np.full(len(hold_ll), lat),
                              hold_ll[:, 0], hold_ll[:, 1])
        if len(dist) and np.min(dist) < 25_000: continue
        sp.append(r)
    assert {r['region'] for r in sp}.isdisjoint(held_ae_sites)
    rng.shuffle(sp)
    rng.shuffle(ae)
    ae_sig, raw_sig, shad_sig = [], [], []
    pairs = []
    for r in ae:
        im = cv2.imread(str(ROOT / 'data/ae_prep/chips' / r['file']), cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        h, w = im.shape
        if r['boxes']:
            box = r['boxes'][int(rng.integers(len(r['boxes'])))]
            cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
        else:
            cx, cy = w/2, h/2
        c = crop_center(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), cx, cy)
        if c is None: continue
        ae_sig.append(signature(c[..., 0]))
        if len(ae_sig) >= args.n: break
    assert len(ae_sig) >= 20
    ae_med = np.median(ae_sig, 0)
    for r in sp:
        im = cv2.imread(str(ROOT / 'data/spor_prep/chips' / r['file']))
        if im is None: continue
        obj = r['objs'][int(rng.integers(len(r['objs'])))]
        p = obj[1].representative_point()
        c = crop_center(im, p.x, p.y)
        if c is None: continue
        raw = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
        film = shad(c, rng, ae_med[0]/255.0, ae_med[1]/255.0)
        raw_sig.append(signature(raw)); shad_sig.append(signature(film))
        if len(pairs) < 12: pairs.append((raw, film))
        if len(shad_sig) >= args.n: break
    assert len(ae_sig) >= 20 and len(shad_sig) >= 20
    raw_med = np.median(raw_sig, 0)
    film_med = np.median(shad_sig, 0)
    names = ['p02', 'p98', 'spread', 'grad_median', 'grad_p90', 'lap_abs_median']
    report = {'fold': args.fold, 'train_ae_chips': len(ae_sig), 'train_sp_object_chips': len(shad_sig),
              'train_ae_sites': len(ae_sites), 'train_sp_sites': len({r['region'] for r in sp}),
              'sp_chips_excluded_by_site_or_25km': len(sp0)-len(sp),
              'metrics': {k: {'ae': float(a), 'sat_gray': float(b), 'shad': float(c)}
                          for k, a, b, c in zip(names, ae_med, raw_med, film_med)}}
    (out / f'fold{args.fold}_summary.json').write_text(json.dumps(report, indent=2))
    rows = [np.hstack([a, b]) for a, b in pairs]
    cv2.imwrite(str(out / f'fold{args.fold}_preview.jpg'), np.vstack(rows))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
