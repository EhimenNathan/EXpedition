"""CPU-only frozen-image-feature study for satellite kurgan scene presence.

Each region is labelled from its training annotation, and image embeddings
contain no filenames, coordinates or labels. The classifier is fitted only on
other geographic folds for every reported OOF probability.
"""
import json
import sys
from pathlib import Path

import cv2
import geopandas as gpd
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from torchvision.models import efficientnet_b0

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from sp_assemble_oof import region_folds

IMAGES = ROOT / "kaggle_import/code/work/sp/img"
WEIGHTS = Path("/home/nathan/.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth")
CACHE = Path(__file__).with_name("sp_kurgan_scene_features.npz")


def embed():
    scenes = pd.read_csv(ROOT / "kaggle_import/code/work/sp/scenes.csv")
    model = efficientnet_b0(weights=None)
    model.load_state_dict(torch.load(WEIGHTS, map_location="cpu", weights_only=True))
    model.classifier = torch.nn.Identity()
    model.eval()
    torch.set_num_threads(4)
    mean = np.array([.485, .456, .406], np.float32)[:, None, None]
    std = np.array([.229, .224, .225], np.float32)[:, None, None]
    rows = []
    for i, r in enumerate(scenes.itertuples(), 1):
        a = np.load(IMAGES / (r.scene + ".npy"), mmap_mode="r")
        step = max(1, int(np.ceil(max(a.shape[1:]) / 512)))
        small = np.asarray(a[:, ::step, ::step])
        rgb = small[:3].transpose(1, 2, 0)
        rgb = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_AREA)
        x = ((rgb.astype(np.float32).transpose(2, 0, 1) / 255 - mean) / std).copy()
        with torch.no_grad():
            z = model(torch.from_numpy(x)[None]).numpy()[0]
        rows.append((r.scene, r.site, z))
        if i % 25 == 0:
            print(f"embedded {i}/{len(scenes)} scenes", flush=True)
    np.savez_compressed(CACHE, scene=np.array([r[0] for r in rows]), region=np.array([r[1] for r in rows]),
                        x=np.stack([r[2] for r in rows]))


def main():
    if not CACHE.exists():
        embed()
    dat = np.load(CACHE)
    D = pd.DataFrame({"region": dat["region"], "i": np.arange(len(dat["region"]))})
    RF = region_folds()
    regions = sorted(set(D.region) & set(RF))
    X = []
    for region in regions:
        a = dat["x"][D[D.region == region].i.to_numpy()]
        X.append(np.r_[a.mean(0), a.max(0)])
    X = np.stack(X)
    L = gpd.read_parquet(ROOT / "data/labels_3857.parquet")
    y = np.array([int(((L.region == r) & (L["mod"] == "SpOr") & L.class_name.isin(("kurgany_povrezhdennye", "kurgany_tselye"))).any()) for r in regions])
    fold = np.array([RF[r] for r in regions])
    p = np.zeros(len(regions))
    for f in sorted(set(fold)):
        tr, va = fold != f, fold == f
        pipe = make_pipeline(StandardScaler(), PCA(n_components=24, random_state=20261023),
                             LogisticRegression(C=.1, max_iter=1000))
        pipe.fit(X[tr], y[tr])
        p[va] = pipe.predict_proba(X[va])[:, 1]
    result = {"regions": len(regions), "positive_regions": int(y.sum()),
              "auc": float(roc_auc_score(y, p)), "average_precision": float(average_precision_score(y, p)),
              "folds": {str(f): {"n": int((fold == f).sum()), "positive": int(y[fold == f].sum()),
                                 "auc": float(roc_auc_score(y[fold == f], p[fold == f])) if len(set(y[fold == f])) == 2 else None}
                        for f in sorted(set(fold))}}
    pd.DataFrame({"region": regions, "fold": fold, "target": y, "p": p}).to_parquet(Path(__file__).with_name("sp_kurgan_scene_oof.parquet"), index=False)
    Path(__file__).with_suffix(".json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
