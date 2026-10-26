"""Package the pre-audited fold-0 LiDAR-to-Ae pseudo-label pilot for private Kaggle.

Reuses the existing private fold-0 image dataset; uploads only a training-site
annotation overlay. Held-out fold-0 boxes and classes are never uploaded.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / 'kaggle_ae_dfine_next/fold0'
OUT = ROOT / 'kaggle_pseudo_f0'
DATA = OUT / 'dataset'
KERNEL = OUT / 'kernel'
SLUG = 'codex-koz3-ae-pseudo-f0'
OVERLAY = 'codex-koz3-ae-li-pseudo-f0'
BASE_SLUG = 'codex-koz3-ae-dfine-f0-s15'


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    KERNEL.mkdir(parents=True, exist_ok=True)
    pseudo = json.loads((ROOT / 'li_ae_pseudo/ann_train_fold0.json').read_text())
    original = json.loads((BASE / 'stage/ann.json').read_text())
    held = [r for r in original if r['fold'] == 0]
    assert len(held) == 298 and all(not r['boxes'] and not r['classes'] for r in held)
    assert len(pseudo) == len(original) - len(held)
    assert all(r['fold'] != 0 for r in pseudo)
    assert {r['region'] for r in pseudo}.isdisjoint({r['region'] for r in held})
    base_train_boxes = sum(len(r['boxes']) for r in original if r['fold'] != 0)
    pseudo_train_boxes = sum(len(r['boxes']) for r in pseudo)
    assert pseudo_train_boxes - base_train_boxes == 151
    (DATA / 'ann_train_fold0.json').write_text(json.dumps(pseudo, ensure_ascii=False))
    (DATA / 'dataset-metadata.json').write_text(json.dumps({
        'title': 'Codex KOZ3 Ae LiDAR pseudo labels fold0 private',
        'id': f'solapeolusola/{OVERLAY}',
        'licenses': [{'name': 'other'}]
    }, indent=2))

    src = (BASE / 'kernel/codex-koz3-ae-dfine-ft-f0-s15.py').read_text()
    src = src.replace('import base64\n', 'import base64\nimport json\n')
    src = src.replace("SLUG='codex-koz3-ae-dfine-f0-s15'", f"SLUG='{BASE_SLUG}'\nPSEUDO_SLUG='{OVERLAY}'")
    src = src.replace("ROOT=Path('/kaggle/working/ae_dfine_f0_s15')",
                      "ROOT=Path('/kaggle/working/ae_dfine_pseudo_f0')")
    needle = "assert DATASET is not None,f'dataset not mounted; candidates={candidates}'"
    repl = needle + "\nPSEUDO=next((p for p in [INPUT/PSEUDO_SLUG, INPUT/'datasets/solapeolusola'/PSEUDO_SLUG] + list(INPUT.glob(f'**/{PSEUDO_SLUG}')) if (p/'ann_train_fold0.json').is_file()),None)\nassert PSEUDO is not None,'private training-only pseudo overlay not mounted'"
    assert needle in src
    src = src.replace(needle, repl)
    needle = "(ROOT/'data/ae_prep').symlink_to(DATASET/'data/ae_prep',target_is_directory=True)"
    repl = """(ROOT/'data/ae_prep').mkdir(exist_ok=True)
(ROOT/'data/ae_prep/chips').symlink_to(DATASET/'data/ae_prep/chips',target_is_directory=True)
train_ann=json.loads((PSEUDO/'ann_train_fold0.json').read_text())
held_ann=[r for r in json.loads((DATASET/'data/ae_prep/ann.json').read_text()) if r['fold']==0]
assert len(held_ann)==298 and all(not r['boxes'] and not r['classes'] for r in held_ann)
assert all(r['fold']!=0 for r in train_ann)
assert {r['region'] for r in train_ann}.isdisjoint({r['region'] for r in held_ann})
(ROOT/'data/ae_prep/ann.json').write_text(json.dumps(train_ann+held_ann,ensure_ascii=False))
print('TRAIN_BOXES',sum(len(r['boxes']) for r in train_ann),'HELDOUT_LABELS',sum(len(r['boxes']) for r in held_ann),flush=True)"""
    assert needle in src
    src = src.replace(needle, repl)
    src = src.replace("print('TRAINING_ONLY_FOLD0_SOURCE; HELDOUT_LABELS_ABSENT',flush=True)",
                      "print('TRAINING_ONLY_FOLD0_LIDAR_PSEUDO; HELDOUT_LABELS_ABSENT',flush=True)")
    src = src.replace("for script,tag in [('ae_dfine_scale.py','scale15')]:",
                      "for script,tag in [('ae_dfine_finetune.py','pseudo')]:")
    src = src.replace("model_root=ROOT/'Codex Work/ae_yolo_pilot'/('dfine_ft' if tag=='plain' else 'dfine_scale')",
                      "model_root=ROOT/'Codex Work/ae_yolo_pilot/dfine_ft'")
    src = src.replace("name='codex_ae_ft' if tag=='plain' else 'codex_ae_s15'",
                      "name='codex_ae_ft'")
    src = src.replace("print('AE_DFINE_F0_DONE',flush=True)",
                      "print('AE_DFINE_PSEUDO_F0_DONE',flush=True)")
    assert 'PSEUDO_SLUG' in src and "for script,tag in [('ae_dfine_finetune.py','pseudo')]:" in src
    code_file = f'{SLUG}.py'
    (KERNEL / code_file).write_text(src)
    (KERNEL / 'kernel-metadata.json').write_text(json.dumps({
        'id': f'solapeolusola/{SLUG}', 'title': SLUG,
        'code_file': code_file, 'language': 'python', 'kernel_type': 'script',
        'is_private': True, 'enable_gpu': True, 'enable_tpu': False,
        'enable_internet': True, 'keywords': ['gpu'],
        'dataset_sources': [f'solapeolusola/{BASE_SLUG}', f'solapeolusola/{OVERLAY}'],
        'kernel_sources': [], 'competition_sources': [], 'model_sources': [],
        'machine_shape': 'NvidiaTeslaT4'
    }, indent=2))
    print(json.dumps({'train_sites': len({r['region'] for r in pseudo}),
                      'heldout_sites': len({r['region'] for r in held}),
                      'base_train_boxes': base_train_boxes,
                      'pseudo_train_boxes': pseudo_train_boxes,
                      'heldout_boxes_uploaded': 0,
                      'source': f'solapeolusola/{BASE_SLUG}',
                      'overlay': f'solapeolusola/{OVERLAY}',
                      'kernel': f'solapeolusola/{SLUG}'}, indent=2))


if __name__ == '__main__':
    main()
