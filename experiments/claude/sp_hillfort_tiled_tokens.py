"""Hillfort localiser, fixed-GSD view: every satellite scene resampled to GSD m/px (default 2 m → 32 m tokens), tiled in
512 px windows (sp_dense/extract tiling: 24-token stride, 4-token margins), full 1024-d DINOv3-SAT tokens stitched into
one grid per scene. Unlike the whole-scene view, a 5 km scene keeps the same token size as a 1 km screenshot, so large
scenes no longer shrink a 150 m hillfort into 1–2 tokens. Output format = sp_hillfort_dinov3_tokens.py (tok, y, pos, sid,
grid, meta), so sp_hillfort_dinov3_eval.py can evaluate / ensemble it.
  python experiments/claude/sp_hillfort_tiled_tokens.py [out=work/hf_dinov3/tokens_gsd2.npz] [gsd=2]
"""
import importlib.util
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, ROOT)
_spec = importlib.util.spec_from_file_location("sp_dense_extract", os.path.join(HERE, "sp_dense", "extract.py"))
SP = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(SP)


def main(out="work/hf_dinov3/tokens_gsd2.npz", gsd=2.0):
    import geopandas as gpd
    from pyproj import Transformer
    from shapely import contains_xy
    gsd = float(gsd)
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[L["mod"] == "SpOr"]
    HF = {r: g.geometry.union_all() for r, g in L[L.class_name == "gorodishcha"].groupby("region")}
    bb = None; Tok, Y, POS, SID, GRID, META = [], [], [], [], [], []
    part_dir = os.path.join(ROOT, out[:-4] + "_parts"); os.makedirs(part_dir, exist_ok=True)   # per-scene parts: resumable
    for k, path in enumerate(SP.scenes()):
        region = os.path.basename(os.path.dirname(os.path.dirname(path)))
        pp = os.path.join(part_dir, f"{k:03d}.npz")
        if os.path.exists(pp):
            z = np.load(pp, allow_pickle=True)
            Tok.append(z["tok"]); Y.append(z["y"]); POS.append(z["pos"]); SID.append(np.full(len(z["y"]), k))
            GRID.append(tuple(z["grid"])); META.append(tuple(z["meta"])); continue
        bb = bb or SP.Backbone()
        img, T, crs, (h, w) = SP.read_gsd(path, gsd)
        views = []
        for flip in (False, True):
            im = np.ascontiguousarray(img[:, ::-1]) if flip else img
            t = SP.scene_tokens(bb, im, lambda t: t).astype(np.float32)
            views.append(t[:, ::-1] if flip else t)
        g = np.mean(views, 0); vh, vw = int(np.ceil(h / SP.PATCH)), int(np.ceil(w / SP.PATCH))
        g = g[:vh, :vw]                                                     # drop padding tokens
        rr, cc = np.mgrid[0:vh, 0:vw]
        Tt = T * T.scale(SP.PATCH, SP.PATCH)
        X, Yp = Tt * (cc.ravel() + 0.5, rr.ravel() + 0.5)
        X3, Y3 = Transformer.from_crs(crs, 3857, always_xy=True).transform(np.asarray(X), np.asarray(Yp))
        y = contains_xy(HF[region], X3, Y3) if region in HF else np.zeros(vh * vw, bool)
        Tok.append(g.reshape(vh * vw, -1).astype(np.float16)); Y.append(y); SID.append(np.full(vh * vw, k)); GRID.append((vh, vw))
        POS.append(np.c_[(cc.ravel() + 0.5) / vw * 2 - 1, (rr.ravel() + 0.5) / vh * 2 - 1, X3, Y3])
        META.append((region, os.path.basename(path), gsd, h, w))
        np.savez(pp + ".tmp.npz", tok=Tok[-1], y=y, pos=POS[-1], grid=np.array((vh, vw)), meta=np.array(META[-1], dtype=object))
        os.replace(pp + ".tmp.npz", pp)
        print(f"{k + 1} {region[:24]} grid {vh}x{vw} hillfort tokens {int(y.sum())}", flush=True)
    np.savez(os.path.join(ROOT, out), tok=np.concatenate(Tok), y=np.concatenate(Y), pos=np.concatenate(POS),
             sid=np.concatenate(SID), grid=np.array(GRID), meta=np.array(META, dtype=object))
    print("saved", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
