"""CPU-only, training-fold-only audit for aerial-film contrast recipes.

No held-out labels or pixels influence recipe selection. This script measures
image properties; it does not claim an object-detection score.
"""
import json
from pathlib import Path

import cv2
import numpy as np


ROOT = Path('/home/nathan/Expedition Competition')
SOURCE = ROOT / 'data/ae_prep'
OUT = ROOT / 'Codex Work/nvidia_pilot'


def percentiles(image):
    valid = image > 0
    if valid.sum() < 100:
        return 0., 0., 0., valid
    p2, p50, p98 = np.percentile(image[valid], (2, 50, 98))
    return float(p2), float(p50), float(p98), valid


def bounded_contrast(image):
    p2, p50, p98, valid = percentiles(image)
    if valid.sum() < 100:
        return image
    gain = min(2.5, 220. / max(p98 - p2, 1.))
    out = np.clip((image.astype(np.float32) - p50) * gain + 128., 0, 255).astype(np.uint8)
    out[~valid] = 0
    return out


def mild_clahe(image):
    _, _, _, valid = percentiles(image)
    out = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(image)
    out[~valid] = 0
    return out


def metrics(image):
    p2, _, p98, valid = percentiles(image)
    px = image[valid]
    if not len(px):
        return {'spread': 0., 'clipped': 0., 'high_freq': 0.}
    lap = cv2.Laplacian(image, cv2.CV_32F)
    return {'spread': round(p98-p2, 3),
            'clipped': round(float(np.mean((px <= 2) | (px >= 253))), 5),
            'high_freq': round(float(np.median(np.abs(lap[valid]))), 3)}


def main():
    rows = json.loads((SOURCE / 'ann.json').read_text())
    train = [r for r in rows if r['fold'] != 0]
    result = []
    for r in train:
        im = cv2.imread(str(SOURCE / 'chips' / r['file']), cv2.IMREAD_GRAYSCALE)
        if im is None:
            raise FileNotFoundError(r['file'])
        p2, _, p98, _ = percentiles(im)
        # Quality tier is fixed before enhancement and evaluated only on training sites.
        spread = p98 - p2
        tier = 'low' if spread < 80 else ('high' if spread >= 160 else 'medium')
        variants = {'raw': im,
                    'bounded': bounded_contrast(im),
                    'clahe': mild_clahe(im)}
        result.append({'file': r['file'], 'site': r['region'], 'tier': tier,
                       'boxes': len(r['boxes']),
                       'variants': {k: metrics(v) for k, v in variants.items()}})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'preprocess_train_fold0_audit.json').write_text(
        json.dumps(result, ensure_ascii=False, indent=2))
    summary = {}
    for tier in ('low', 'medium', 'high'):
        subset = [r for r in result if r['tier'] == tier]
        summary[tier] = {'chips': len(subset), 'positive_chips': sum(r['boxes'] > 0 for r in subset)}
        for variant in ('raw', 'bounded', 'clahe'):
            summary[tier][variant] = {key: round(float(np.median([
                r['variants'][variant][key] for r in subset])), 3)
                for key in ('spread', 'clipped', 'high_freq')} if subset else {}
    (OUT / 'preprocess_train_fold0_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
