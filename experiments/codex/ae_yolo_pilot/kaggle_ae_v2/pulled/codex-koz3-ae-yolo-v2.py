"""Fixed fold-0 aerial YOLO pilot; held-out labels are never mounted.

The private input dataset has training images/labels and held-out pixels only.
All predictions are exported for scoring locally with the official metric.
"""
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path('/kaggle/working/ae_v2')
DATASET = Path('/kaggle/input/codex-koz3-ae-f0-v2')
EPOCHS = 80
HIGH_QUALITY_SPREAD = 160.0


def normalize_bgr(im):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    valid = gray > 0
    if valid.sum() < 100:
        return im
    p2, p50, p98 = np.percentile(gray[valid], (2, 50, 98))
    if p98 - p2 >= HIGH_QUALITY_SPREAD:
        return im
    gain = min(2.5, 220.0 / max(p98 - p2, 1.0))
    adjusted = np.clip((gray.astype(np.float32) - p50) * gain + 128.0, 0, 255).astype(np.uint8)
    adjusted[~valid] = 0
    return cv2.cvtColor(adjusted, cv2.COLOR_GRAY2BGR)


def main():
    assert DATASET.is_dir(), DATASET
    ROOT.mkdir(parents=True, exist_ok=True)
    shutil.copytree(DATASET / 'train', ROOT / 'train', dirs_exist_ok=True)
    train_images = list((ROOT / 'train/images').glob('*.jpg'))
    train_labels = list((ROOT / 'train/labels').glob('*.txt'))
    held = json.loads((DATASET / 'heldout_metadata.json').read_text())
    assert len(train_images) == len(train_labels) == 510
    assert len(held) == 298
    assert all('boxes' not in r and 'classes' not in r for r in held)

    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'ultralytics>=8.4.32,<8.5'], check=True)
    from ultralytics import YOLO

    yaml = ROOT / 'data.yaml'
    yaml.write_text('path: ' + str(ROOT / 'train') + '\ntrain: images\nval: images\n'
                    'names:\n  0: kurgany_povrezhdennye\n  1: kurgany_tselye\n'
                    '  2: gorodishcha\n  3: arkhitektury\n  4: fortifikatsii\n')
    model = YOLO('yolo11m.pt')

    def fail_on_nonfinite_batch(trainer):
        loss = trainer.loss
        if loss is not None and not bool(torch.isfinite(loss.detach()).all()):
            raise FloatingPointError(f'Non-finite YOLO loss at epoch {trainer.epoch}')

    model.add_callback('on_train_batch_end', fail_on_nonfinite_batch)
    model.train(data=str(yaml), epochs=EPOCHS, imgsz=1024, batch=2, device=0, workers=4,
                project=str(ROOT), name='train_run', exist_ok=True, seed=20261024,
                hsv_h=0., hsv_s=0., hsv_v=.15, degrees=0., fliplr=.5, flipud=.5,
                mosaic=0., amp=False, val=False, plots=False, save=True,
                save_period=10, verbose=True, patience=EPOCHS + 1)
    last = ROOT / 'train_run/weights/last.pt'
    assert last.is_file()
    net = YOLO(str(last))
    assert all(bool(torch.isfinite(t).all()) for t in net.model.state_dict().values()
               if t.is_floating_point()), 'nonfinite trained weights'

    out = Path('/kaggle/working/ae_yolo_fold0_raw.csv')
    count = 0
    with out.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('image', 'class_id', 'confidence', 'x0', 'y0', 'x1', 'y1'))
        for i in range(0, len(held), 4):
            batch_meta = held[i:i+4]
            images = [cv2.imread(str(DATASET / 'heldout_images' / r['image']), cv2.IMREAD_COLOR)
                      for r in batch_meta]
            assert all(im is not None for im in images)
            images = [normalize_bgr(im) for im in images]
            predictions = net.predict(images, imgsz=1024, conf=.05, iou=.5,
                                      device=0, batch=4, verbose=False)
            for r, pred in zip(batch_meta, predictions):
                for box, sc, cls in zip(pred.boxes.xyxy.cpu().numpy(),
                                        pred.boxes.conf.cpu().numpy(),
                                        pred.boxes.cls.cpu().numpy().astype(int)):
                    writer.writerow((r['image'], int(cls), float(sc), *[float(x) for x in box]))
                    count += 1
            if i % 40 == 0:
                print('OOF inference', min(i+4,len(held)), '/',len(held),'raw',count,flush=True)
    print('EXPORTED_OOF', count, out, flush=True)


if __name__ == '__main__':
    main()
