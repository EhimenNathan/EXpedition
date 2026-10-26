"""LiDAR test-time self-training (TTT), honest OOF: for each held-out site, the fold model (never trained on it) predicts;
its CONFIDENT outputs become pseudo-labels (kurgan peaks ≥ tau_hi sized from the model's own mound mask; hillfort /
fortification mask components ≥ 0.6), confidently empty ground is background, everything uncertain gets zero loss
weight. A copy of the model is fine-tuned on that site alone (original losses, low lr), then predicts again. No labels of
the site are used; the configuration is fixed in advance (no selection on the scored sites).

  python scripts/li_ttt_oof.py [--sites 021,011,...] [--tag _aug] [--iters 300] [--tau 0.5]
"""
import argparse
import copy
import os
import pickle
import sys
import time

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)


def pseudo_item(it, p, tau_hi=0.5, tau_lo=0.08):
    """→ item dict for li_train.Sampler: pseudo objects (class, polygon in cache px) + loss-weight map as 'valid'."""
    from rasterio.features import shapes
    from scipy.ndimage import label as cc_label, binary_dilation
    from shapely.geometry import Point, shape as shp
    from koz3 import hm
    from koz3.li_train import N_HM
    valid = it["valid"]; objs = []
    mk = p[N_HM:]
    kmask = (mk[0] + mk[1]) >= 0.5
    lab, _ = cc_label(kmask)
    covered = np.zeros_like(valid)
    for ch, x, y, s in hm.decode(p[:1], valid, thr=tau_hi, nms=11):
        yi, xi = int(y), int(x)
        k = lab[yi, xi]
        rad = float(np.clip(np.sqrt((lab == k).sum() / np.pi) if k else 6.0, 4.0, 24.0))       # px (0.5 m)
        c = 0 if mk[0, yi, xi] >= mk[1, yi, xi] else 1
        objs.append((c, Point(x, y).buffer(rad, 16)))
    for c, ch, min_px in ((2, N_HM + 2, 600), (3, N_HM + 3, 30)):
        m = (p[ch] >= 0.6) & valid
        for g, v in shapes(m.astype(np.uint8), mask=m):
            poly = shp(g)
            if v and poly.area >= min_px:
                objs.append((c, poly))
    from rasterio.features import rasterize
    if objs:
        covered = rasterize([(g, 1) for _, g in objs], out_shape=valid.shape).astype(bool)
    uncertain = ((p[0] >= tau_lo) | (mk.max(0) >= 0.2)) & ~covered
    if os.environ.get("TTT_BUMPS") == "1":     # every physical bump ≥ 15 cm (10 m local relief) is uncertain, not background
        from scipy.ndimage import gaussian_filter
        uncertain |= (gaussian_filter(np.asarray(it["x"][2], np.float32), 1.5) >= 0.3) & ~covered
    uncertain = binary_dilation(uncertain, iterations=2) & ~covered
    weight = valid & ~uncertain
    return dict(it, objs=objs, valid=weight.astype(bool)), len(objs)


def adapt(model, item, iters=300, bs=8, lr=1e-4, seed=0):
    from koz3 import hm
    from koz3.li_train import Sampler, N_HM
    m = copy.deepcopy(model).train()
    S = Sampler([item], np.random.default_rng(seed))
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-2)
    for i in range(iters):
        img, t_hm, mk, val = [x.cuda() for x in S.batch(bs)]
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = m(img.to(memory_format=torch.channels_last))
        w = val[:, None]
        loss = hm.focal_hm(out[:, :N_HM], t_hm, w) + hm.bce_dice(out[:, N_HM:], mk, w)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    return m.eval()


def site_score(fs, gt):
    from koz3.score import score
    _, res = score({"type": "FeatureCollection", "features": fs}, gt)
    k = [res[c] for c in ("kurgany_povrezhdennye", "kurgany_tselye") if c in res]
    tp, fp, fn = sum(v["tp"] for v in k), sum(v["fp"] for v in k), sum(v["fn"] for v in k)
    return tp, fp, fn


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sites", default=""); ap.add_argument("--tag", default="_aug")
    ap.add_argument("--iters", type=int, default=300); ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--out", default="work/li/oof_ttt.pkl")
    a = ap.parse_args()
    from koz3 import hm
    from koz3.li_train import labels, site_items, load_model, decode_site_v2, N_HM, N_MK
    from koz3.labels import to_gt_geojson
    L = labels(); items = [it for it in site_items(L) if it["fold"] >= 0]
    if a.sites:
        items = [it for it in items if it["region"].split("_")[0] in a.sites.split(",")]
    models = {}
    out = pickle.load(open(os.path.join(ROOT, a.out), "rb")) if os.path.exists(os.path.join(ROOT, a.out)) else {}
    tot0, tot1 = np.zeros(3), np.zeros(3)
    for it in items:
        r = it["region"]; f = it["fold"]
        if f not in models:
            models[f] = load_model(os.path.join(ROOT, f"work/li/li{a.tag}_f{f}.pt"))
        x = np.asarray(it["x"], np.float32); t0 = time.time()
        p0 = hm.predict_tiled(models[f], x, N_HM + N_MK)
        item, n = pseudo_item(it, p0, a.tau)
        if n >= 3:
            p1 = hm.predict_tiled(adapt(models[f], item, a.iters), x, N_HM + N_MK)
        else:
            p1 = p0
        out[r] = (p0.astype(np.float16), p1.astype(np.float16))
        gt = to_gt_geojson(L[L.region == r]); meta = dict(it["meta"], region=r)
        s0 = site_score(decode_site_v2(p0, it["valid"], meta, thr=0.15, dual=0.05), gt)
        s1 = site_score(decode_site_v2(p1, it["valid"], meta, thr=0.15, dual=0.05), gt)
        tot0 += s0; tot1 += s1
        f1 = lambda t: 2 * t[0] / max(2 * t[0] + t[2] + 0.5 * t[1], 1)
        print(f"{r[:24]:24s} pseudo {n:4d}  kurgans before tp/fp/fn {s0} F1 {f1(s0):.3f} → after {s1} F1 {f1(s1):.3f}  {time.time() - t0:.0f}s", flush=True)
        pickle.dump(out, open(os.path.join(ROOT, a.out), "wb"))
    f1 = lambda t: 2 * t[0] / max(2 * t[0] + t[2] + 0.5 * t[1], 1)
    print(f"POOLED kurgan F1 before {f1(tot0):.4f} → after {f1(tot1):.4f}  ({tot0.astype(int)} → {tot1.astype(int)})")


if __name__ == "__main__":
    main()
