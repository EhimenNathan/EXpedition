"""Prepare a geographically held-out satellite SegFormer pilot.

This reads the competition's existing local chips and polygon labels without
altering them. Fold 0 is validation only; all generated files stay in Codex Work.
The optional smoke mode writes a small, inspectable sample before a full export.
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from rasterio.features import rasterize
from shapely.affinity import translate

ROOT = Path('/home/nathan/Expedition Competition')
sys.path.insert(0, str(ROOT))
from koz3.sp_train import load_chips  # reads the canonical labels and fold assignment

SOURCE = ROOT / 'data/spor_prep/chips'
OUT = ROOT / 'Codex Work/nvidia_pilot/sp_segformer_f0'
SIZE = 512
STRIDE = 384


def starts(length):
    if length <= SIZE:
        return [0]
    return sorted(set(range(0, length - SIZE + 1, STRIDE)) | {length - SIZE})


def export(smoke):
    out = OUT.with_name('sp_segformer_f0_smoke') if smoke else OUT
    chips = load_chips('SpOr')
    train_sites = {c['region'] for c in chips if c['fold'] != 0}
    held_sites = {c['region'] for c in chips if c['fold'] == 0}
    assert train_sites.isdisjoint(held_sites), 'geographic fold contamination'
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    if smoke:
        # Exercise both splits and several source sites without exporting the dataset.
        chips = [next(c for c in chips if c['fold'] != 0),
                 next(c for c in chips if c['fold'] == 0)]
    limit = 16 if smoke else None
    for c in chips:
        if limit is not None and len(manifest) >= limit:
            break
        split = 'val' if c['fold'] == 0 else 'train'
        im = cv2.imread(str(SOURCE / c['file']), cv2.IMREAD_COLOR)
        if im is None:
            raise FileNotFoundError(c['file'])
        assert im.shape[:2] == (c['h'], c['w'])
        for y0 in starts(c['h']):
            for x0 in starts(c['w']):
                if limit is not None and len(manifest) >= limit:
                    break
                crop = im[y0:y0 + SIZE, x0:x0 + SIZE]
                valid = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) > 0
                polygons = [(translate(g, xoff=-x0, yoff=-y0), int(cl) + 1)
                            for cl, g in c['objs'] if not g.is_empty]
                mask = rasterize(polygons, out_shape=crop.shape[:2], fill=0,
                                 dtype=np.uint8) if polygons else np.zeros(crop.shape[:2], np.uint8)
                mask[~valid] = 255
                name = f"{len(manifest):06d}"
                img_dir = out / 'images' / split
                mask_dir = out / 'masks' / split
                img_dir.mkdir(parents=True, exist_ok=True)
                mask_dir.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(img_dir / f'{name}.png'), crop)
                cv2.imwrite(str(mask_dir / f'{name}.png'), mask)
                counts = np.bincount(mask[mask != 255], minlength=6)
                manifest.append({'name': name, 'split': split, 'site': c['region'],
                                 'source': c['file'], 'x0': x0, 'y0': y0,
                                 'size': [int(crop.shape[1]), int(crop.shape[0])],
                                 'class_pixels': counts.tolist(),
                                 'valid_fraction': round(float(valid.mean()), 4)})
    (out / 'manifest.json').write_text(json.dumps({
        'fold': 0, 'smoke': smoke, 'size': SIZE, 'stride': STRIDE,
        'class_ids': {'background': 0, 'damaged_kurgan': 1, 'intact_kurgan': 2,
                      'hillfort': 3, 'architecture': 4, 'fortification': 5,
                      'ignore': 255},
        'train_sites': sorted(train_sites), 'heldout_sites': sorted(held_sites),
        'tiles': manifest}, ensure_ascii=False, indent=2))
    print('chips', len(chips), 'train sites', len(train_sites), 'heldout sites', len(held_sites),
          'exported tiles', len(manifest), 'positive tiles', sum(any(v > 0 for v in x['class_pixels'][1:]) for x in manifest))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    export(parser.parse_args().smoke)
