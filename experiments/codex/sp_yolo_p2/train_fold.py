#!/usr/bin/env python3
"""Geographic OOF pilot for a true P2 satellite detector.

The existing satellite chips/folds are reused verbatim.  A YOLO11m backbone is
loaded into a P2/P3/P4/P5 head, trained without validation-driven checkpoint
selection, then every held-out chip is exported in EPSG:3857 for exact scoring.
"""
import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import Affine
from ultralytics import YOLO


ROOT = Path("/home/nathan/Expedition Competition/Codex Work/dinov3_wt")
SRC = ROOT / "data/spor_prep"
HERE = ROOT / "experiments/codex/sp_yolo_p2"
CLS = ["kurgany_povrezhdennye", "kurgany_tselye", "gorodishcha", "arkhitektury", "fortifikatsii"]


def prepare(fold: int, out: Path):
    rows = json.loads((SRC / "ann.json").read_text())
    # Match the deployed D-FINE geographic folds exactly.  The original
    # sp_prep folds predate that ensemble and are different.
    fp = ROOT / "kaggle_import/out/koz3-sp-det/sp_runs/sp_dfine_l/folds.json"
    ext = {k.rsplit("__", 1)[0]: int(v) for k, v in json.loads(fp.read_text()).items()}
    for r in rows:
        if r["region"] in ext:
            r["fold"] = ext[r["region"]]
    train = [r for r in rows if int(r["fold"]) != fold]
    held = [r for r in rows if int(r["fold"]) == fold]
    image_dir, label_dir = out / "dataset/images", out / "dataset/labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    for i, r in enumerate(train):
        stem = f"{i:05d}"
        dst = image_dir / f"{stem}.jpg"
        if not dst.exists():
            os.link(SRC / "chips" / r["file"], dst)
        with (label_dir / f"{stem}.txt").open("w") as f:
            for (x0, y0, x1, y1), cls in zip(r["boxes"], r["classes"]):
                x0, x1 = sorted((np.clip(x0, 0, r["w"]), np.clip(x1, 0, r["w"])))
                y0, y1 = sorted((np.clip(y0, 0, r["h"]), np.clip(y1, 0, r["h"])))
                if x1 - x0 >= 2 and y1 - y0 >= 2:
                    f.write(f"{cls} {(x0+x1)/(2*r['w']):.8f} {(y0+y1)/(2*r['h']):.8f} "
                            f"{(x1-x0)/r['w']:.8f} {(y1-y0)/r['h']:.8f}\n")
    yaml = out / "data.yaml"
    yaml.write_text("path: " + str(out / "dataset") + "\ntrain: images\nval: images\nnames:\n" +
                    "".join(f"  {i}: {name}\n" for i, name in enumerate(CLS)))
    (out / "manifest.json").write_text(json.dumps({
        "heldout_fold": fold, "train_chips": len(train), "heldout_chips": len(held),
        "train_regions": sorted({r["region"] for r in train}),
        "heldout_regions": sorted({r["region"] for r in held}),
    }, ensure_ascii=False, indent=2))
    return yaml, held


def export_oof(net: YOLO, held, fold: int, out: Path):
    rows = []
    for start in range(0, len(held), 2):
        meta = held[start:start + 2]
        images = [cv2.imread(str(SRC / "chips" / r["file"])) for r in meta]
        pred = net.predict(images, imgsz=1024, conf=0.01, iou=0.5, max_det=1000,
                           device=0, batch=2, verbose=False)
        for r, p in zip(meta, pred):
            transform = Affine(*r["transform"])
            project = Transformer.from_crs(r["crs"], 3857, always_xy=True)
            scene = r["file"].rsplit("__", 1)[0]
            for box, score, cls in zip(p.boxes.xyxy.cpu().numpy(), p.boxes.conf.cpu().numpy(),
                                       p.boxes.cls.cpu().numpy().astype(int)):
                gx, gy = transform * ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                x, y = project.transform(gx, gy)
                rows.append((r["region"], scene, fold, CLS[cls], float(score), x, y))
        if start % 20 == 0:
            print("infer", start, "/", len(held), "candidates", len(rows), flush=True)
    df = pd.DataFrame(rows, columns=["region", "scene", "fold", "cls", "score", "x", "y"])
    df.to_parquet(out / "oof_candidates.parquet")
    print("saved", len(df), "candidates", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=50)
    args = ap.parse_args()
    out = HERE / f"fold{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    yaml, held = prepare(args.fold, out)
    model = YOLO(str(HERE / "yolo11m-p2.yaml")).load(str(ROOT.parent / "yolo11m.pt"))
    model.train(data=str(yaml), epochs=args.epochs, imgsz=1024, batch=2, device=0, workers=4,
                project=str(out), name="run", exist_ok=True, seed=20261029 + args.fold,
                hsv_h=0.015, hsv_s=0.35, hsv_v=0.25, degrees=180, translate=0.1,
                scale=0.35, fliplr=0.5, flipud=0.5, mosaic=1.0, close_mosaic=10,
                mixup=0.1, amp=True, val=False, plots=False, save=True, verbose=True,
                patience=args.epochs + 1)
    weights = out / "run/weights/last.pt"
    export_oof(YOLO(str(weights)), held, args.fold, out)


if __name__ == "__main__":
    main()
