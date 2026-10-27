"""Prospective geographic OOF test for a half-resolution LiDAR inference pass.

Large, clean mounds at Erentuna are outside the dominant training pixel scale.  This test uses the existing
fold-specific weights only: inputs are downsampled 2x, inferred, and predictions are resized back to the native grid.
The half-scale probabilities are blended with the deployed OOF blend.  Alpha is selected on the other geographic
folds; acceptance requires an honest gain and no outer-fold regression.  Decoded features are checkpointed after
every region, so the run can resume without storing another multi-gigabyte probability cache.

  python scripts/li_halfscale_oof.py
"""
import gc
import os
import pickle
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))
ALPHAS = (0.0, 0.1, 0.2, 0.33, 0.5)
OUT = os.path.join(ROOT, "work/li/halfscale_features.pkl")


def resize_chw(x, shape):
    h, w = shape
    return np.stack([cv2.resize(np.asarray(a, np.float32), (w, h), interpolation=cv2.INTER_AREA) for a in x])


def upsample_chw(x, shape):
    h, w = shape
    return np.stack([cv2.resize(a, (w, h), interpolation=cv2.INTER_LINEAR) for a in x])


def main():
    import li_compare_oof as lc
    from koz3 import hm
    from koz3.labels import to_gt_geojson
    from koz3.li_train import CLS, N_HM, N_MK, decode_site_v2, labels, load_model, site_items
    from koz3.nested import _add_all, fold_counts, nested, weighted

    L = labels(); items = {it["region"]: it for it in site_items(L)}
    # Exact deployed OOF probability blend.  Build one float16 map per site, releasing each source cache promptly.
    paths = ("work/li/oof_ens.pkl.zst", "work/li/oof_aug.pkl.zst", "work/li/oof_augx.pkl.zst")
    base = None
    for path in paths:
        q = lc._pk(path)
        if base is None:
            base = {r: q[r].astype(np.float32) / len(paths) for r in q}
        else:
            for r in base:
                base[r] += q[r].astype(np.float32) / len(paths)
        del q; gc.collect()
    base = {r: p.astype(np.float16) for r, p in base.items()}
    done = pickle.load(open(OUT, "rb")) if os.path.exists(OUT) else {}
    folds = sorted({it["fold"] for it in items.values() if it["fold"] >= 0})
    prior = {"R": 200.0, "K": 3, "snb": 0.2}

    for fold in folds:
        todo = [it for it in items.values() if it["fold"] == fold and it["region"] not in done]
        if not todo:
            continue
        models = [load_model(os.path.join(ROOT, f"work/li/li{tag}_f{fold}.pt")) for tag in ("", "_s1")]
        for it in todo:
            r = it["region"]; x = np.asarray(it["x"], np.float32); H, W = x.shape[1:]
            small = resize_chw(x, (max(32, H // 2), max(32, W // 2)))
            ps = [hm.predict_tiled(m, small, N_HM + N_MK, tta=True) for m in models]
            scale = upsample_chw(np.mean(ps, axis=0), (H, W))
            rows = {}
            for a in ALPHAS:
                p = base[r].astype(np.float32) if a == 0 else (1 - a) * base[r].astype(np.float32) + a * scale
                rows[a] = decode_site_v2(p, it["valid"], dict(it["meta"], region=r), thr=0.15, dual=0.05,
                                         comp_thr=0.5, prior=prior)
            done[r] = rows
            with open(OUT + ".tmp", "wb") as f:
                pickle.dump(done, f, protocol=pickle.HIGHEST_PROTOCOL); f.flush(); os.fsync(f.fileno())
            os.replace(OUT + ".tmp", OUT)
            print(f"halfscale fold {fold} {r}: " + " ".join(f"a={a}:{len(rows[a])}" for a in ALPHAS), flush=True)
            del x, small, ps, scale; gc.collect(); torch.cuda.empty_cache()
        del models; gc.collect(); torch.cuda.empty_cache()

    region_fold = {r: items[r]["fold"] for r in done}; gt = {
        f: to_gt_geojson(L[L.region.isin([r for r in done if region_fold[r] == f])]) for f in folds
    }
    C = fold_counts(lambda a, f: {"type": "FeatureCollection", "features": sum(
        (done[r][a] for r in done if region_fold[r] == f), [])}, ALPHAS, folds, gt, CLS)
    for a in ALPHAS:
        print(f"alpha={a:<4} pooled {weighted(_add_all(C[a].values())):.4f} folds " +
              " ".join(f"{weighted(C[a][f]):.4f}" for f in folds))
    h, ch, o, os_, _ = nested(C)
    baseline = weighted(_add_all(C[0.0].values()))
    worst = min(weighted(C[ch[f]][f]) - weighted(C[0.0][f]) for f in folds)
    print(f"baseline {baseline:.4f} | nested honest {h:.4f} ({h-baseline:+.4f}) oracle {o:.4f} @ {os_}")
    print(f"chosen {ch}; worst outer-fold change {worst:+.4f}")


if __name__ == "__main__":
    main()
