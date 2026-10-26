"""Render model-ready synthetic film chips from train-only SpOr polygons.

Requires a separate site-disjoint detector experiment before deployment. The
image transform preserves pixel geometry; all polygons intersecting a crop are
written as boxes. Test folds and a true geodesic 25 km buffer are excluded.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from pyproj import Geod, Transformer
from shapely.geometry import box as geom_box

ROOT = Path('/home/nathan/Expedition Competition')
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from koz3.sp_train import load_chips
from shad_cpu import crop_center, shad, signature, lonlat_center

CLASSES = ['kurgany_povrezhdennye', 'kurgany_tselye', 'gorodishcha',
           'arkhitektury', 'fortifikatsii']


def train_ae_tone(fold, rng):
    ae = json.loads((ROOT / 'data/ae_prep/ann.json').read_text())
    train = [r for r in ae if r['fold'] != fold]
    hold = [r for r in ae if r['fold'] == fold]
    assert {r['region'] for r in train}.isdisjoint({r['region'] for r in hold})
    rng.shuffle(train)
    sig = []
    for r in train:
        im = cv2.imread(str(ROOT / 'data/ae_prep/chips' / r['file']), cv2.IMREAD_GRAYSCALE)
        if im is None: continue
        if r['boxes']:
            b = r['boxes'][int(rng.integers(len(r['boxes'])))]
            cx, cy = (b[0]+b[2])/2, (b[1]+b[3])/2
        else: cx, cy = r['w']/2, r['h']/2
        c = crop_center(cv2.cvtColor(im, cv2.COLOR_GRAY2BGR), cx, cy)
        if c is not None: sig.append(signature(c[..., 0]))
        if len(sig) >= 200: break
    assert len(sig) >= 30
    med = np.median(sig, axis=0)
    return float(med[0]/255), float(med[1]/255), hold


def allowed_sp(fold, hold):
    geod = Geod(ellps='WGS84')
    hold_sites = {r['region'] for r in hold}
    hll = np.asarray([lonlat_center(r) for r in hold])
    records = [r for r in load_chips('SpOr') if int(r['fold']) != fold and r['objs']]
    keep = []
    rejected = Counter()
    for r in records:
        if r['region'] in hold_sites:
            rejected['same_heldout_site'] += 1; continue
        lon, lat = lonlat_center(r)
        _, _, dist = geod.inv(np.full(len(hll), lon), np.full(len(hll), lat), hll[:, 0], hll[:, 1])
        if len(dist) and float(np.min(dist)) < 25_000:
            rejected['within_true_25km'] += 1; continue
        # Avoid enlarging low-resolution screenshot pixels into fake detail.
        gsd = float(abs(r['transform'][0]))
        if not .25 <= gsd <= .8:
            rejected['gsd_outside_0.25_0.8'] += 1; continue
        keep.append(r)
    return keep, rejected


def annotations(chip, cx, cy):
    side = 256
    x0, y0 = int(round(cx-side/2)), int(round(cy-side/2))
    win = geom_box(x0, y0, x0+side, y0+side)
    labels = []
    for cls, poly in chip['objs']:
        if poly.is_empty or not poly.intersects(win): continue
        clip = poly.intersection(win)
        if clip.is_empty or clip.area < .75 * poly.area: continue
        bx0, by0, bx1, by1 = clip.bounds
        bx0, by0, bx1, by1 = bx0-x0, by0-y0, bx1-x0, by1-y0
        if bx1-bx0 < 3 or by1-by0 < 3: continue
        labels.append((int(cls), (bx0+bx1)/(2*side), (by0+by1)/(2*side),
                       (bx1-bx0)/side, (by1-by0)/side))
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True, choices=range(4))
    ap.add_argument('--max-images', type=int, default=64)
    args = ap.parse_args()
    rng = np.random.default_rng(20261025 + args.fold)
    p02, p98, hold = train_ae_tone(args.fold, rng)
    chips, excluded = allowed_sp(args.fold, hold)
    bysite = defaultdict(list)
    for r in chips:
        rng.shuffle(r['objs'])
        for cls, poly in r['objs']:
            p = poly.representative_point()
            bysite[r['region']].append((r, p.x, p.y, poly))
    for val in bysite.values(): rng.shuffle(val)
    out = HERE / 'shad_dataset' / f'fold{args.fold}_strict'
    images, labels_dir = out/'images', out/'labels'
    images.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    selected, class_counts = [], Counter()
    while len(selected) < args.max_images and any(bysite.values()):
        for site in sorted(bysite):
            if not bysite[site]: continue
            r, cx, cy, primary_poly = bysite[site].pop()
            x0, y0 = int(round(cx-128)), int(round(cy-128))
            window = geom_box(x0, y0, x0+256, y0+256)
            if primary_poly.is_empty or primary_poly.intersection(window).area < .75*primary_poly.area:
                continue
            im = cv2.imread(str(ROOT/'data/spor_prep/chips'/r['file']))
            if im is None: continue
            crop = crop_center(im, cx, cy)
            if crop is None: continue
            labs = annotations(r, cx, cy)
            if not labs: continue
            film = shad(crop, rng, p02, p98)
            stem = f'{len(selected):05d}'
            cv2.imwrite(str(images/f'{stem}.jpg'), film, [cv2.IMWRITE_JPEG_QUALITY, 95])
            (labels_dir/f'{stem}.txt').write_text(''.join(
                f'{cls} {x:.7f} {y:.7f} {w:.7f} {h:.7f}\n' for cls,x,y,w,h in labs))
            selected.append({'file': stem+'.jpg', 'source_file': r['file'],
                             'source_region': site, 'classes': [int(v[0]) for v in labs]})
            class_counts.update(int(v[0]) for v in labs)
            if len(selected) >= args.max_images: break
    report = {'fold': args.fold, 'images': len(selected), 'sites': len({z['source_region'] for z in selected}),
              'heldout_ae_sites': len({r['region'] for r in hold}),
              'source_chips_after_buffer': len(chips), 'source_chips_excluded': dict(excluded),
              'train_ae_tone_p02_p98': [p02, p98],
              'class_counts': {CLASSES[k]: v for k,v in sorted(class_counts.items())},
              'records': selected}
    (out/'manifest.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k != 'records'}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
