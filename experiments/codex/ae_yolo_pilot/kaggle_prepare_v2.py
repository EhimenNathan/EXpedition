"""Package the fixed fold-0 Ae pilot for a private Kaggle GPU run.

Held-out boxes/classes never leave the local machine. Kaggle receives only
held-out pixels and georeferencing needed to return predictions for local
official scoring. This script writes only below Codex Work.
"""
import json
import shutil
from pathlib import Path

ROOT = Path('/home/nathan/Expedition Competition')
HERE = ROOT / 'Codex Work/ae_yolo_pilot'
SOURCE = ROOT / 'data/ae_prep'
PILOT = HERE / 'fold0_v2'
OUT = HERE / 'kaggle_fold0_v2_dataset'


def main():
    rows = json.loads((SOURCE / 'ann.json').read_text())
    train_sites = set(json.loads((PILOT / 'manifest.json').read_text())['train_sites'])
    held = [r for r in rows if r['fold'] == 0]
    held_sites = {r['region'] for r in held}
    assert train_sites.isdisjoint(held_sites)

    for kind in ('images', 'labels'):
        src = PILOT / 'train' / kind
        dst = OUT / 'train' / kind
        dst.mkdir(parents=True, exist_ok=True)
        for f in sorted(src.glob('*')):
            assert f.is_file() and not f.is_symlink(), f
            target = dst / f.name
            if not target.exists() or target.stat().st_size != f.stat().st_size:
                shutil.copy2(f, target)
    val_dir = OUT / 'heldout_images'
    val_dir.mkdir(parents=True, exist_ok=True)
    meta = []
    for i, r in enumerate(held):
        name = f'{i:05d}.jpg'
        src = SOURCE / 'chips' / r['file']
        target = val_dir / name
        assert src.is_file(), src
        if not target.exists() or target.stat().st_size != src.stat().st_size:
            shutil.copy2(src, target)
        meta.append({k: r[k] for k in ('region', 'scene', 'transform', 'crs', 'w', 'h')}
                    | {'image': name})
    (OUT / 'heldout_metadata.json').write_text(json.dumps(meta, ensure_ascii=False))
    (OUT / 'dataset-metadata.json').write_text(json.dumps({
        'title': 'Private KOZ3 Ae fold0 Codex v2',
        'id': 'solapeolusola/codex-koz3-ae-f0-v2',
        'licenses': [{'name': 'other'}],
        'description': 'Private organizer-approved competition data for geographically held-out pilot. No heldout labels included.'
    }, indent=2))
    assert len(list((OUT / 'train/images').glob('*.jpg'))) == 510
    assert len(list((OUT / 'train/labels').glob('*.txt'))) == 510
    assert len(meta) == 298
    assert not any(('boxes' in r or 'classes' in r) for r in meta)
    print('prepared', OUT, 'train=510 heldout=298 no heldout labels')


if __name__ == '__main__':
    main()
