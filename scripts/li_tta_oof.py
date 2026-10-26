"""LiDAR TTA study: 3 views (identity, h-flip, v-flip; deployed) vs d8 (4 rotations × 2 flips) for the copy-paste/openness
models (tag _augx), held-out fold models, channels from the DTM exactly as at inference. Nested decoding threshold."""
import os, sys
import numpy as np
os.environ["RELIEF_EXTRA"] = "1"
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT)


def main(tag="_augx"):
    from koz3 import hm, relief
    from koz3.li_train import labels, site_items, load_model, decode_site_v2, _dtm, _DTM, N_HM, N_MK, CLS
    from koz3.labels import to_gt_geojson
    from koz3.score import score
    from koz3.nested import nested
    L = labels(); items = [it for it in site_items(L) if it["fold"] >= 0]
    folds = sorted({it["fold"] for it in items}); TH = (0.15, 0.2, 0.25)
    C = {(m, t): {f: {} for f in folds} for m in ("flip3", "d8") for t in TH}
    for f in folds:
        mdl = load_model(os.path.join(ROOT, f"work/li/li{tag}_f{f}.pt"))
        for it in [i for i in items if i["fold"] == f]:
            d, v, res = _dtm(it["region"]); x, _ = relief.channels_tiled(d, v, res); _DTM.pop(it["region"], None)
            gt = to_gt_geojson(L[L.region == it["region"]]); meta = dict(it["meta"], region=it["region"])
            for mode, tta in ((("d8", "d8"),) if os.environ.get("D8_ONLY") else (("flip3", True), ("d8", "d8"))):
                p = hm.predict_tiled(mdl, x, N_HM + N_MK, tta=tta)
                for t in TH:
                    _, r = score({"type": "FeatureCollection", "features": decode_site_v2(p, it["valid"], meta, thr=t, dual=0.05)}, gt, CLS)
                    acc = C[(mode, t)][f]
                    for c, vv in r.items():
                        acc[c] = tuple(np.add(acc.get(c, (0, 0, 0)), (vv["tp"], vv["fp"], vv["fn"])))
            print("done", it["region"], flush=True)
    for mode in (("d8",) if os.environ.get("D8_ONLY") else ("flip3", "d8")):
        h, ch, o, os_, tot = nested({k: v for k, v in C.items() if k[0] == mode})
        print(f"{mode:6s} honest {h:.4f} (oracle {o:.4f} @ {os_}) " + " ".join(f"{c[:8]} {2*v[0]/max(2*v[0]+v[2]+0.5*v[1],1):.3f}" for c, v in tot.items() if v[0] + v[2]))


if __name__ == "__main__":
    main(*sys.argv[1:])
