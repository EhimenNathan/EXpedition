"""Independent aerial-film YOLO pilot with a fixed geographic fold-0 holdout.

Reads prepared competition chips and metadata in place; writes only under
Codex Work. Training uses no held-out chips or labels. The last fixed-epoch
checkpoint, rather than a checkpoint selected on the holdout, is exported.
"""
import json
import os
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from pyproj import Transformer
from rasterio.transform import Affine
from ultralytics import YOLO

ROOT = Path("/home/nathan/Expedition Competition")
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "data/ae_prep"
OUT = Path(os.environ.get("KOZ3_AE_OUT", HERE / "fold0"))
FOLD = 0
EPOCHS = int(os.environ.get("KOZ3_AE_EPOCHS", "80"))
CLASSES = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii"]
NORMALIZE = os.environ.get("KOZ3_AE_NORMALIZE", "0") == "1"
HIGH_QUALITY_SPREAD = 160.0  # fixed from the training-fold-only contrast audit
SITE_CAP = int(os.environ.get("KOZ3_AE_SITE_CAP", "0"))
sys.path.insert(0, str(ROOT))
from koz3.sp_fuse import fuse


def records():
    return json.loads((SOURCE / "ann.json").read_text())


def normalize_bgr(im):
    """Bounded per-chip film contrast adjustment; identical at train/inference."""
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


def prepare(rows):
    images = OUT / "train/images"
    labels = OUT / "train/labels"
    images.mkdir(parents=True, exist_ok=True)
    labels.mkdir(parents=True, exist_ok=True)
    neg_keep = float(os.environ.get("KOZ3_AE_NEG_KEEP", "1.0"))
    train = [r for r in rows if r["fold"] != FOLD and
             (r["boxes"] or zlib.crc32(r["file"].encode()) / 2**32 < neg_keep)]
    if SITE_CAP:
        # A separate candidate for testing site balance. Select using training
        # metadata only; preserve rare-class chips before uniform hash order.
        by_site = {}
        for r in train:
            by_site.setdefault(r["region"], []).append(r)
        train = []
        for site in sorted(by_site):
            def priority(r):
                rare = any(c in (2, 3) for c in r["classes"])
                return (not rare, not bool(r["boxes"]), zlib.crc32(r["file"].encode()))
            train.extend(sorted(by_site[site], key=priority)[:SITE_CAP])
    held = [r for r in rows if r["fold"] == FOLD]
    assert set(r["region"] for r in train).isdisjoint(set(r["region"] for r in held))
    (OUT / "manifest.json").write_text(json.dumps({"fold": FOLD, "train_chips": len(train),
                                                    "heldout_chips": len(held),
                                                    "contrast_normalization": NORMALIZE,
                                                    "identity_spread_ge": HIGH_QUALITY_SPREAD if NORMALIZE else None,
                                                    "site_cap": SITE_CAP or None,
                                                    "train_sites": sorted({r["region"] for r in train}),
                                                    "heldout_sites": sorted({r["region"] for r in held})}, indent=2,
                                                   ensure_ascii=False))
    for i, r in enumerate(train):
        stem = f"{i:05d}"
        dest = images / f"{stem}.jpg"
        if NORMALIZE:
            if dest.is_symlink():
                dest.unlink()
            im = cv2.imread(str(SOURCE / "chips" / r["file"]), cv2.IMREAD_COLOR)
            assert im is not None
            cv2.imwrite(str(dest), normalize_bgr(im), [cv2.IMWRITE_JPEG_QUALITY, 95])
        elif not dest.exists():
            dest.symlink_to(SOURCE / "chips" / r["file"])
        with (labels / f"{stem}.txt").open("w") as f:
            for (x0, y0, x1, y1), cls in zip(r["boxes"], r["classes"]):
                x0, x1 = sorted((max(0., min(r["w"], x0)), max(0., min(r["w"], x1))))
                y0, y1 = sorted((max(0., min(r["h"], y0)), max(0., min(r["h"], y1))))
                if x1 - x0 < 2 or y1 - y0 < 2:
                    continue
                f.write(f"{cls} {(x0+x1)/(2*r['w']):.7f} {(y0+y1)/(2*r['h']):.7f} "
                        f"{(x1-x0)/r['w']:.7f} {(y1-y0)/r['h']:.7f}\n")
    yaml = OUT / "data.yaml"
    yaml.write_text("path: " + str(OUT / "train") + "\ntrain: images\nval: images\n" +
                    "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    print("prepared", len(train), "training chips", len(held), "held-out chips", flush=True)
    return yaml, held


def train(yaml):
    pretrained = ROOT / "Codex Work/yolo11m.pt"
    model = YOLO(str(pretrained if pretrained.exists() else "yolo11m.pt"))

    def fail_on_nonfinite_batch(trainer):
        # The first pilot silently continued after its weights became NaN.
        # Stop this independent run immediately so it cannot export an invalid model.
        loss = trainer.loss
        if loss is not None and not bool(torch.isfinite(loss.detach()).all()):
            raise FloatingPointError(f"Non-finite YOLO loss at epoch {trainer.epoch}")

    model.add_callback("on_train_batch_end", fail_on_nonfinite_batch)
    model.train(data=str(yaml), epochs=EPOCHS, imgsz=1024, batch=4, device=0, workers=4,
                project=str(OUT), name="ae_fold0_run", exist_ok=True, seed=20261024,
                hsv_h=0., hsv_s=0., hsv_v=.15,
                degrees=float(os.environ.get("KOZ3_AE_DEGREES", "180")),
                fliplr=.5, flipud=.5, mosaic=float(os.environ.get("KOZ3_AE_MOSAIC", "1")),
                amp=os.environ.get("KOZ3_AE_AMP", "1") == "1",
                val=False, plots=False, save=True, save_period=10,
                verbose=True, patience=EPOCHS + 1)
    last = OUT / "ae_fold0_run/weights/last.pt"
    assert last.exists()
    trained = YOLO(str(last)).model
    assert all(bool(torch.isfinite(t).all()) for t in trained.state_dict().values()
               if t.is_floating_point()), "Trained checkpoint contains non-finite weights"
    return last


def export(held, weights):
    model = YOLO(str(weights))
    rows = []
    for i in range(0, len(held), 4):
        batch_meta = held[i:i + 4]
        batch = [cv2.imread(str(SOURCE / "chips" / r["file"]), cv2.IMREAD_COLOR) for r in batch_meta]
        assert all(im is not None for im in batch)
        if NORMALIZE:
            batch = [normalize_bgr(im) for im in batch]
        outputs = model.predict(batch, imgsz=1024, conf=.05, iou=.5, device=0, batch=4, verbose=False)
        for r, pred in zip(batch_meta, outputs):
            T = Affine(*r["transform"])
            tr = Transformer.from_crs(r["crs"], 3857, always_xy=True)
            for box, sc, cls in zip(pred.boxes.xyxy.cpu().numpy(), pred.boxes.conf.cpu().numpy(),
                                    pred.boxes.cls.cpu().numpy().astype(int)):
                x0, y0, x1, y1 = box
                gx, gy = T * ((x0+x1)/2, (y0+y1)/2)
                mx, my = tr.transform(gx, gy)
                rows.append((r["region"], CLASSES[cls], float(sc), float(mx), float(my)))
        if (i + 4) % 40 == 0:
            print("inferred", min(i + 4, len(held)), "/", len(held), "chips", len(rows), "raw", flush=True)
    raw = pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y"])
    raw.to_parquet(OUT / "oof_raw.parquet")
    merged = fuse([raw], np.ones(1), match_m=4.0) if len(raw) else raw
    merged.to_parquet(OUT / "oof_merged.parquet")
    print("exported", len(raw), "raw ->", len(merged), "merged", flush=True)


def main():
    yaml, held = prepare(records())
    if os.environ.get("KOZ3_AE_PREP_ONLY") == "1":
        return
    weights = train(yaml)
    export(held, weights)


if __name__ == "__main__":
    main()
