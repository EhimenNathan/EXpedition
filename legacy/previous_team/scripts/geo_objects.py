"""Geophysics objects (class object_poly) on magnetometry, resistivity (plan maps + profiles) and thermography maps.

    python scripts/geo_objects.py [data_root=data_geo]

Per unit: physics feature stack at object scales (0.3-2.4 m; background 5/15 m) + HSV colour channels (resistivity and
thermography maps are colour-coded) -> LightGBM pixel classifier trained leave-one-region-out on sampled pixels ->
probability map -> threshold -> connected components -> polygons (confidence = mean probability) -> official matching
(cov >= 0.5 of the smaller shape AND predicted centroid inside the labelled object), pooled per method.
Writes work/geo/objects_report.csv, work/geo/objects_oof/<unit>.npz (probability maps) and figures."""
import os, sys, time, json, argparse, warnings
import numpy as np, pandas as pd
from scipy import ndimage as ndi
import rasterio.features as RF
from rasterio import features
from shapely.geometry import shape
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from koz3 import geo_data as GD, geo_feats as GF
from koz3.official.fastmatch import soft_match, prf

ap = argparse.ArgumentParser(); ap.add_argument("root", nargs="?", default="data_geo")
ap.add_argument("--palette", type=int, default=1); ap.add_argument("--tag", default="v2")
ap.add_argument("--npos", type=int, default=3000); ap.add_argument("--nneg", type=int, default=6000)
args = ap.parse_args()
PALETTE = bool(args.palette)
GSD = {"MagGr": 0.2, "EtGr": 0.1, "TerFl": 0.3}
OBJ_SCALES = (0.3, 0.6, 1.2, 2.4)
METHODS = ["MagGr", "EtGr", "TerFl"]
OUT = f"work/geo/objects_oof_{args.tag}"; os.makedirs(OUT, exist_ok=True)


def hsv(rgb, valid):
    import cv2
    x = np.clip(np.moveaxis(rgb, 0, -1), 0, 255).astype(np.uint8)
    h = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float32)
    h[~valid] = 0
    hue = h[..., 0] * (2 * np.pi / 180.0)
    return {"hue_sin": np.sin(hue) * (h[..., 1] / 255), "hue_cos": np.cos(hue) * (h[..., 1] / 255), "sat": h[..., 1] / 255}


def unit_stack(u):
    d = GD.load_unit(u, GSD[u["method"]])
    base = d["lum"]
    if PALETTE:
        pal, is_col = GF.palette_scalar(d["rgb"], d["valid"])
        if is_col:
            base = pal * 255.0  # rainbow-coded map: physics features on the decoded value, not on luminance
    names, A = GF.stack(base, d["valid"], d["gsd"], scales_m=OBJ_SCALES, bg_m=(5.0, 15.0))
    extra = hsv(d["rgb"], d["valid"])
    for k, v in extra.items():
        names.append(k); A = np.concatenate([A, v[None].astype(np.float32)])
    const = {f"m_{m}": float(u["method"] == m) for m in METHODS}
    const.update(is_profile=float(u["kind"] == "profile"))  # map resolution removed: it identified regions (v0 top feature)
    gt = d["objects"]
    lab = np.zeros(d["lum"].shape, bool)
    if gt:
        lab = features.rasterize([(g, 1) for g in gt], out_shape=lab.shape, transform=d["T"], all_touched=False).astype(bool)
    return d, names, A, const, lab, gt


def to_frame(names, A, const, rr, cc):
    X = pd.DataFrame({n: A[i][rr, cc] for i, n in enumerate(names)})
    for k, v in const.items():
        X[k] = v
    return X


def polygons(prob, valid, T, gsd, thr, min_area=0.3):
    m = (ndi.gaussian_filter(prob, 1.0) >= thr) & valid
    m = ndi.binary_opening(m, iterations=1)
    lab, n = ndi.label(m)
    out = []
    if n == 0:
        return out
    means = ndi.mean(prob, lab, index=np.arange(1, n + 1)); sizes = ndi.sum(np.ones_like(prob), lab, index=np.arange(1, n + 1))
    for geom, val in RF.shapes(lab.astype(np.int32), mask=lab > 0, transform=T):
        i = int(val) - 1
        if sizes[i] * gsd * gsd < min_area:
            continue
        g = shape(geom)
        if not g.is_valid:
            g = g.buffer(0)
        if g.geom_type == "MultiPolygon":
            g = max(g.geoms, key=lambda x: x.area)
        out.append((g.simplify(gsd, preserve_topology=True), float(means[i])))
    return out


def main():
    t0 = time.time()
    U = [u for u in GD.discover(args.root) if u["objects"]]
    rng = np.random.default_rng(0)
    rows, meta, ys, regs_s, feats = [], [], [], [], None
    for u in U:
        try:
            d, names, A, const, lab, gt = unit_stack(u)
        except Exception as e:
            print("skip", u["region"], u["key"], repr(e)); continue
        v = d["valid"]
        pos = np.argwhere(lab & v); neg = np.argwhere(~lab & v)
        pos = pos[rng.choice(len(pos), min(len(pos), args.npos), replace=False)] if len(pos) else pos
        neg = neg[rng.choice(len(neg), min(len(neg), args.nneg), replace=False)] if len(neg) else neg
        pts = np.r_[pos, neg]
        X = to_frame(names, A, const, pts[:, 0], pts[:, 1])
        if feats is None:
            feats = list(X.columns)
        uid = f"{u['region']}|{u['method']}|{u['key']}"
        rows.append(X[feats].to_numpy(np.float32)); ys.append(np.r_[np.ones(len(pos)), np.zeros(len(neg))].astype(np.int8))
        regs_s.append(np.full(len(pts), u["region"], dtype=object))
        meta.append(dict(uid=uid, region=u["region"], method=u["method"], kind=u["kind"], n_gt=len(gt),
                         pos_px=int((lab & v).sum()), valid_px=int(v.sum()), gsd=round(d["gsd"], 3),
                         med_obj_m2=round(float(np.median([g.area for g in gt])), 1) if gt else 0))
        print(json.dumps(meta[-1], ensure_ascii=False), f"{time.time() - t0:.0f}s", flush=True)
        del A
    XS = np.concatenate(rows); del rows; YS = np.concatenate(ys); RS = np.concatenate(regs_s); M = pd.DataFrame(meta)
    print(f"\n{len(XS)} sampled pixels, {len(M)} units, {M.n_gt.sum()} objects", flush=True)
    import lightgbm as lgb
    regions = sorted(M.region.unique()); imp = np.zeros(len(feats))
    results = []  # (uid, method, region, thr) -> counts
    THRS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for r in regions:
        tr = RS != r
        m = lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.7,
                               subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, n_jobs=8, verbose=-1)
        m.fit(XS[tr], YS[tr]); imp += m.booster_.feature_importance("gain")
        for u in [u for u in U if u["region"] == r]:
            uid = f"{u['region']}|{u['method']}|{u['key']}"
            if uid not in set(M.uid):
                continue
            d, names, A, const, lab, gt = unit_stack(u)
            v = d["valid"]; rr, cc = np.nonzero(v)
            prob = np.zeros(v.shape, np.float32)
            for c0 in range(0, len(rr), 200_000):  # chunked prediction keeps memory flat
                sl = slice(c0, c0 + 200_000)
                prob[rr[sl], cc[sl]] = m.predict_proba(to_frame(names, A, const, rr[sl], cc[sl])[feats].to_numpy(np.float32))[:, 1]
            np.savez_compressed(f"{OUT}/{uid.replace('|', '__')}.npz", prob=prob.astype(np.float16), valid=v)
            for t in THRS:
                preds = polygons(prob, v, d["T"], d["gsd"], t)
                tp, fp, fn, _ = soft_match(preds, gt)
                results.append(dict(uid=uid, region=r, method=u["method"], kind=u["kind"], thr=t, tp=tp, fp=fp, fn=fn))
            del A
        print(f"region {r} done {time.time() - t0:.0f}s", flush=True)
    R = pd.DataFrame(results); R.to_csv(f"work/geo/objects_counts_{args.tag}.csv", index=False)
    # honest threshold per method: chosen on other regions
    rep = []
    for meth, g in R.groupby("method"):
        tot = np.zeros(3)
        for r in g.region.unique():
            o = g[g.region != r].groupby("thr")[["tp", "fp", "fn"]].sum()
            if not len(o):
                continue
            best = max(o.index, key=lambda t: prf(*o.loc[t].values)[2])
            tot += g[(g.region == r) & (g.thr == best)][["tp", "fp", "fn"]].sum().values
        orc = g.groupby("thr")[["tp", "fp", "fn"]].sum(); ob = max(orc.index, key=lambda t: prf(*orc.loc[t].values)[2])
        P_, R_, F_ = prf(*tot)
        rep.append(dict(method=meth, tp=int(tot[0]), fp=int(tot[1]), fn=int(tot[2]), P=round(P_, 3), R=round(R_, 3), F1_honest=round(F_, 3),
                        oracle_thr=ob, F1_oracle=round(prf(*orc.loc[ob].values)[2], 3)))
    allc = np.sum([[x["tp"], x["fp"], x["fn"]] for x in rep], axis=0)
    rep.append(dict(method="ALL (pooled object_poly)", tp=int(allc[0]), fp=int(allc[1]), fn=int(allc[2]), F1_honest=round(prf(*allc)[2], 3)))
    rep = pd.DataFrame(rep); rep.to_csv(f"work/geo/objects_report_{args.tag}.csv", index=False); M.to_csv("work/geo/objects_units.csv", index=False)
    pd.set_option("display.width", 220); print(rep.to_string(index=False))
    print("per region (oracle-free view at thr 0.5):"); print(R[R.thr == 0.5].groupby(["region", "method"])[["tp", "fp", "fn"]].sum().to_string())
    print("top features:", pd.Series(imp, feats).sort_values(ascending=False).head(12).round(0).to_dict())
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
