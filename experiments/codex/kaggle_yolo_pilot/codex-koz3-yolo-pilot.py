"""Private KOZ-3 satellite YOLO pilot: fixed-epoch fold-0 training and OOF export.

Reads only the existing private koz3-sp-prep Kaggle kernel source. Its held-out
fold is never used by the trainer or to choose a checkpoint. Output predictions
are scored locally with the organiser's exact metric before any further run.
"""
import glob
import json
import os
import subprocess
import sys
import tarfile
import zlib
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from pyproj import Transformer
from rasterio.transform import Affine
from shapely import wkb


ROOT = Path(os.environ.get("KOZ3_CODE", "/kaggle/code_root"))
INPUT = Path(os.environ.get("KOZ3_INPUT", "/kaggle/input"))
OUT = Path(os.environ.get("KOZ3_OUT", "/kaggle/working/yolo_pilot"))
FOLD = int(os.environ.get("KOZ3_FOLD", "0"))
EPOCHS = int(os.environ.get("KOZ3_EPOCHS", "40"))
SMOKE = os.environ.get("KOZ3_SMOKE") == "1"
CHIP, OVERLAP = 640, 160
CLASSES = ["kurgany_tselye", "kurgany_povrezhdennye", "gorodishcha", "fortifikatsii", "arkhitektury"]
RU = {"курганы_целые": 0, "курганы_поврежденные": 1, "городища": 2, "фортификации": 3, "архитектуры": 4}


def setup():
    global ROOT
    if not (ROOT / "docs/site_groups.csv").exists():
        direct_root = INPUT / "koz3-code"
        if (direct_root / "docs/site_groups.csv").exists():
            ROOT = direct_root
            code = []
        else:
            code = glob.glob(str(INPUT / "**/koz3_code.tar.gz"), recursive=True)
        if code:
            ROOT.mkdir(parents=True, exist_ok=True)
            with tarfile.open(code[0]) as tar:
                tar.extractall(ROOT)
        elif not (ROOT / "docs/site_groups.csv").exists():
            # Kaggle may mount this private dataset already extracted.
            modules = glob.glob(str(INPUT / "**/koz3/sp_io.py"), recursive=True)
            if not modules:
                raise FileNotFoundError("private koz3-code dataset source is missing")
            ROOT = Path(modules[0]).parents[1]  # read-only source; no copy needed
    sp_override = os.environ.get("KOZ3_SP")
    if sp_override:
        sp = Path(sp_override)
    else:
        direct_sp = INPUT / "koz3-sp-prep/sp"
        if (direct_sp / "scenes.csv").exists():
            sp = direct_sp
        else:
            matches = glob.glob(str(INPUT / "**/sp/scenes.csv"), recursive=True)
            if not matches:
                raise FileNotFoundError("private koz3-sp-prep kernel source is missing")
            sp = Path(matches[0]).parent
    OUT.mkdir(parents=True, exist_ok=True)
    return sp


def scene_info(sp):
    scenes = pd.read_csv(sp / "scenes.csv")
    scenes = scenes[scenes.valid_frac > .01].reset_index(drop=True)
    labels = pd.read_parquet(sp / "labels.parquet")
    groups = pd.read_csv(ROOT / "docs/site_groups.csv").set_index("site")["g20"].to_dict()
    rows = {}
    for r in scenes.itertuples():
        T = Affine(*json.loads(r.T))
        boxes = []
        for cls, blob in zip(labels[labels.scene == r.scene].cls, labels[labels.scene == r.scene].wkb):
            g = wkb.loads(blob)
            x0, y0, x1, y1 = g.bounds
            u0, v0 = (~T) * (x0, y1)
            u1, v1 = (~T) * (x1, y0)
            boxes.append((RU[cls], min(u0, u1), min(v0, v1), max(u0, u1), max(v0, v1)))
        rows[r.scene] = dict(site=r.site, crs=r.crs, T=T, H=int(r.H), W=int(r.W),
                             boxes=np.array(boxes, np.float32).reshape(-1, 5),
                             group=groups.get(r.site, zlib.crc32(r.site.encode()) % 997))
    # Same deterministic geographic fold assignment as the existing D-FINE.
    load, group_load, fold_of_group = [0] * 4, {}, {}
    for d in rows.values():
        group_load[d["group"]] = group_load.get(d["group"], 0) + len(d["boxes"]) + 1
    for group, n in sorted(group_load.items(), key=lambda kv: -kv[1]):
        f = int(np.argmin(load))
        fold_of_group[group] = f
        load[f] += n
    for d in rows.values():
        d["fold"] = fold_of_group[d["group"]]
    return rows


def tiles(h, w):
    step = CHIP - OVERLAP
    return [(y, x) for y in range(0, max(h - OVERLAP, 1), step)
            for x in range(0, max(w - OVERLAP, 1), step)]


def read_rgb_chip(image, y, x):
    h, w = image.shape[1:]
    crop = np.zeros((4, CHIP, CHIP), np.uint8)
    part = np.asarray(image[:, y:min(y + CHIP, h), x:min(x + CHIP, w)])
    crop[:, :part.shape[1], :part.shape[2]] = part
    crop[:3, crop[3] == 0] = 0
    return crop[:3].transpose(1, 2, 0)


def chip_and_boxes(image, boxes, y, x):
    rgb = read_rgb_chip(image, y, x)
    b = boxes.copy()
    b[:, [1, 3]] -= x
    b[:, [2, 4]] -= y
    original = np.maximum(1e-6, (b[:, 3] - b[:, 1]) * (b[:, 4] - b[:, 2]))
    clipped = b.copy()
    clipped[:, 1:] = np.clip(clipped[:, 1:], 0, CHIP)
    width = clipped[:, 3] - clipped[:, 1]
    height = clipped[:, 4] - clipped[:, 2]
    keep = (width * height >= .5 * original) & (width >= 2) & (height >= 2)
    return rgb, clipped[keep]


def export_training(sp, scenes):
    image_dir = OUT / "train/images"
    label_dir = OUT / "train/labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)
    count, with_object = 0, 0
    train_scenes = [name for name, d in scenes.items() if d["fold"] != FOLD]
    held_scenes = [name for name, d in scenes.items() if d["fold"] == FOLD]
    assert set(train_scenes).isdisjoint(held_scenes)
    (OUT / "fold_manifest.json").write_text(json.dumps({"heldout_fold": FOLD, "train_scenes": train_scenes,
                                                         "heldout_scenes": held_scenes,
                                                         "folds": {name: d["fold"] for name, d in scenes.items()}}, indent=2))
    if SMOKE:
        train_scenes = train_scenes[:4]
    for i, name in enumerate(train_scenes, 1):
        d = scenes[name]
        image = np.load(sp / "img" / (name + ".npy"), mmap_mode="r")
        for y, x in tiles(d["H"], d["W"]):
            rgb, boxes = chip_and_boxes(image, d["boxes"], y, x)
            # Keep all positives and one quarter of negative tiles.
            if len(boxes) == 0 and zlib.crc32(f"{name}/{y}/{x}".encode()) % 4:
                continue
            stem = f"{i:03d}_{y:05d}_{x:05d}"
            cv2.imwrite(str(image_dir / (stem + ".jpg")), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(label_dir / (stem + ".txt"), "w") as stream:
                for cls, x0, y0, x1, y1 in boxes:
                    stream.write(f"{int(cls)} {(x0+x1)/(2*CHIP):.7f} {(y0+y1)/(2*CHIP):.7f} "
                                 f"{(x1-x0)/CHIP:.7f} {(y1-y0)/CHIP:.7f}\n")
            count += 1
            with_object += bool(len(boxes))
        if i % 20 == 0:
            print(f"export {i}/{len(train_scenes)} scenes, {count} tiles", flush=True)
    yaml = OUT / "data.yaml"
    yaml.write_text("path: " + str(OUT / "train") + "\ntrain: images\nval: images\n" +
                    "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASSES)))
    print(f"exported {count} train tiles, {with_object} with objects", flush=True)
    return yaml


def train(yaml):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics>=8.3,<9"], check=True)
    from ultralytics import YOLO
    model = YOLO("yolo11m.pt")
    model.train(data=str(yaml), epochs=EPOCHS, imgsz=CHIP, batch=8, device=0, workers=4,
                project=str(OUT), name="fold" + str(FOLD), exist_ok=True, seed=20261023,
                val=False, plots=False, save=True, verbose=True, patience=EPOCHS + 1)
    weights = OUT / f"fold{FOLD}/weights/last.pt"
    assert weights.exists()
    return weights


def export_oof(sp, scenes, weights):
    import torch
    import torchvision
    from ultralytics import YOLO
    model = YOLO(str(weights))
    rows = []
    held = [(name, d) for name, d in scenes.items() if d["fold"] == FOLD]
    for si, (name, d) in enumerate(held, 1):
        image = np.load(sp / "img" / (name + ".npy"), mmap_mode="r")
        boxes, scores, classes = [], [], []
        ts = tiles(d["H"], d["W"])
        for start in range(0, len(ts), 8):
            batch_xy = ts[start:start + 8]
            # Ultralytics treats numpy inputs as OpenCV BGR, then converts to RGB.
            # Training JPEGs were encoded from BGR; match that path at inference.
            batch = [cv2.cvtColor(read_rgb_chip(image, y, x), cv2.COLOR_RGB2BGR)
                     for y, x in batch_xy]
            outputs = model.predict(batch, imgsz=CHIP, conf=.05, iou=.5,
                                    device=os.environ.get("KOZ3_DEVICE", "0"), verbose=False, batch=8)
            for (y, x), out in zip(batch_xy, outputs):
                if len(out.boxes) == 0:
                    continue
                b = out.boxes.xyxy.cpu().numpy()
                b[:, [0, 2]] += x
                b[:, [1, 3]] += y
                boxes.append(torch.from_numpy(b))
                scores.append(out.boxes.conf.cpu())
                classes.append(out.boxes.cls.cpu().long())
        if boxes:
            B, S, C = torch.cat(boxes), torch.cat(scores), torch.cat(classes)
            keep = torchvision.ops.batched_nms(B, S, C, .5)
            tr = Transformer.from_crs(d["crs"], 3857, always_xy=True)
            for j in keep.tolist():
                x0, y0, x1, y1 = B[j].tolist()
                gx, gy = d["T"] * ((x0+x1)/2, (y0+y1)/2)
                mx, my = tr.transform(gx, gy)
                rows.append((d["site"], name, FOLD, CLASSES[int(C[j])], float(S[j]), mx, my))
        if si % 10 == 0:
            print(f"inferred {si}/{len(held)} held-out scenes, {len(rows)} detections", flush=True)
    path = OUT / f"oof_yolo_fold{FOLD}.parquet"
    pd.DataFrame(rows, columns=["region", "scene", "fold", "cls", "score", "x", "y"]).to_parquet(path)
    print(f"OOF predictions {path}: {len(rows)}", flush=True)


def main():
    sp = setup()
    scenes = scene_info(sp)
    yaml = export_training(sp, scenes)
    if SMOKE:
        print("SMOKE preparation complete; no GPU training or inference", flush=True)
        return
    weights = train(yaml)
    export_oof(sp, scenes, weights)


if __name__ == "__main__":
    main()
