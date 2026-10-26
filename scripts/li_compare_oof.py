"""Compare LiDAR OOF probability sets (pickles of region → (7,H,W)) under the same nested decoding (decode_site_v2,
thr × comp_thr chosen on the other folds), with per-class F1 and per-site kurgan recall.

  python scripts/li_compare_oof.py work/li/oof.pkl work/li/oof_aug.pkl [ens:work/li/oof.pkl+work/li/oof_aug.pkl]
"""
import os, sys, pickle
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)


def _pk(path):
    """pickle from .pkl or (streamed, no temp file) .pkl.zst."""
    path = os.path.join(ROOT, path)
    if path.endswith(".zst") or (not os.path.exists(path) and os.path.exists(path + ".zst")):
        import subprocess
        zp = path if path.endswith(".zst") else path + ".zst"
        pr = subprocess.Popen(["zstd", "-dc", "-T4", zp], stdout=subprocess.PIPE)
        out = pickle.load(pr.stdout); pr.wait()
        return out
    return pickle.load(open(path, "rb"))


def load(spec):
    if spec.startswith("ens:"):                      # ens:a.pkl*2+b.pkl → weighted mean (weights default 1)
        parts = [(q.split("*")[0], float(q.split("*")[1]) if "*" in q else 1.0) for q in spec[4:].split("+")]
        ps = [(_pk(p), w) for p, w in parts]
        W = sum(w for _, w in ps)
        return {r: (sum(p[r].astype(np.float32) * w for p, w in ps) / W).astype(np.float16) for r in ps[0][0]}
    return _pk(spec)


def main(specs):
    from koz3.li_train import labels, site_items, decode_site_v2, CLS
    from koz3.labels import to_gt_geojson
    from koz3.nested import fold_counts, nested
    from koz3.score import score
    L = labels(); items = {it["region"]: it for it in site_items(L)}
    for spec in specs:
        P = load(spec)
        fold = {r: items[r]["fold"] for r in P}; folds = sorted(set(fold.values()))
        gt = {f: to_gt_geojson(L[L.region.isin([r for r in P if fold[r] == f])]) for f in folds}
        cache = {}

        def make(s, f):
            t, comp, du = s if len(s) == 3 else (*s, 0.05); feats = []
            for r, p in P.items():
                if fold[r] != f:
                    continue
                if (r, t, comp, du) not in cache:
                    it = items[r]
                    cache[(r, t, comp, du)] = decode_site_v2(p.astype(np.float32), it["valid"], dict(it["meta"], region=r), thr=t, dual=du, comp_thr=comp)
                feats += cache[(r, t, comp, du)]
            return {"type": "FeatureCollection", "features": feats}
        duals = [float(d) for d in os.environ.get("DUALS", "0.05").split(",")]
        S = [(t, c, d) for t in (0.1, 0.15, 0.2) for c in (0.5,) for d in duals] if os.environ.get("DUALS") else [(t, c) for t in (0.05, 0.08, 0.1, 0.15, 0.2) for c in (0.4, 0.5)]
        C = fold_counts(make, S, folds, gt, CLS)
        h, ch, o, os_, tot = nested(C)
        print(f"{spec:40s} honest {h:.4f} (oracle {o:.4f} @ {os_}) " + " ".join(
            f"{c[:8]} {2 * v[0] / max(2 * v[0] + v[2] + 0.5 * v[1], 1):.3f}" for c, v in tot.items() if v[0] + v[2]), flush=True)
        rec = []
        for r in ("036_Тимерево", "021_Новенькое", "028_Вышегжа_2", "011_Эрентуна", "102_Гнёздово_Заольшанская", "180_Гнёздово_Лесная_2", "030_Эсбю"):
            if r in P:
                it = items[r]; s_ = ch[fold[r]]
                fs = decode_site_v2(P[r].astype(np.float32), it["valid"], dict(it["meta"], region=r), thr=s_[0], dual=s_[2] if len(s_) == 3 else 0.05, comp_thr=s_[1])
                _, res = score({"type": "FeatureCollection", "features": fs}, to_gt_geojson(L[L.region == r]))
                k = [res[c] for c in ("kurgany_povrezhdennye", "kurgany_tselye") if c in res]
                tp, fn = sum(v["tp"] for v in k), sum(v["fn"] for v in k)
                rec.append(f"{r[:10]} {tp / max(tp + fn, 1):.2f}")
        print("    kurgan recall by site: " + ", ".join(rec), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
