"""Class-aware ensemble: per class family, the CNN / DINOv3-heads mixing weight w (0 … 1) is chosen TOGETHER with the
class's decoding thresholds on the other folds (nested), so each class can lean on the model that is better for it
(e.g. architecture → CNN, intact kurgans → DINOv3 heads). Uses the oof_ens_<w> maps from ensemble.py and the decoding /
official scoring of sp_dense/evaluate.py (HF_POINTS / HF_SCORE / PEAK_K honoured).
  HF_SCORE=mean HF_POINTS=experiments/claude/outputs/hf_dinov3_loc_points_oof.csv PEAK_K=7 \
      python experiments/claude/sp_cnn/select_ensemble.py
"""
import glob, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE, "..", "sp_dense"))
import evaluate as E  # noqa: E402
WS = (0.0, 0.25, 0.5, 0.75, 1.0)


def main():
    import geopandas as gpd
    from koz3.labels import to_gt_geojson
    from koz3.score import score, WEIGHTS
    from koz3.sp_prep import site_folds
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[L["mod"] == "SpOr"]; fo = site_folds(L)
    cands = {w: E.candidates(f"work/sp_cnn/oof_ens_{w}") for w in WS}
    C0 = cands[0.5]; regions = sorted(set(C0.region)); folds = sorted({fo[r] for r in regions})
    files = sorted(glob.glob(os.path.join(ROOT, "work/sp_cnn/oof_ens_0.5/*.npz")))
    LG = E.layered_gt(L[L.region.isin(regions)], files)
    gt = {f: to_gt_geojson(LG[LG.region.map(fo) == f]) for f in folds}
    grid = {"k": [(t, 0.3, kg, fr) for t in E.T_K for kg in (9.0, 0.8, 0.7) for fr in E.FRAME[:2]],
            "arkhitektury": [(t, fr) for t in E.T_A for fr in E.FRAME[:2]], "fortifikatsii": list(E.T_F), "gorodishcha": list(E.G_H)}
    fam_classes = {"k": E.K, "arkhitektury": ("arkhitektury",), "fortifikatsii": ("fortifikatsii",), "gorodishcha": ("gorodishcha",)}
    CNT = {}
    for w, C in cands.items():
        C = C.assign(fold=C.region.map(fo))
        for fam, opts in grid.items():
            if fam == "gorodishcha" and w != 0.5:
                continue                                      # hillforts come from the localiser: identical for every w
            for o in opts:
                a = dict(t_k=9, tau=9, t_a=9, t_f=9, g_h=9)
                if fam == "k":
                    a.update(t_k=o[0], tau=o[1], kg=o[2], fk=o[3])
                elif fam == "arkhitektury":
                    a.update(t_a=o[0], fa=o[1])
                else:
                    a[{"fortifikatsii": "t_f", "gorodishcha": "g_h"}[fam]] = o
                for f in folds:
                    _, res = score(E.fc(E.emit(C[C.fold == f], **a)), gt[f])
                    CNT[(fam, w, o, f)] = {c: (v["tp"], v["fp"], v["fn"]) for c, v in res.items()}
        print(f"w={w} scored", flush=True)
    tot, chosen = {}, {}
    for fam, opts in grid.items():
        keys = [(w, o) for w in (WS if fam != "gorodishcha" else (0.5,)) for o in opts]
        for f in folds:
            val = lambda k: sum(WEIGHTS[c] * E.f1(*np.sum([CNT[(fam,) + k + (g,)][c] for g in folds if g != f], 0)) for c in fam_classes[fam])
            k = max(keys, key=val); chosen[(fam, f)] = k
            for c in fam_classes[fam]:
                tot[c] = np.add(tot.get(c, (0, 0, 0)), CNT[(fam,) + k + (f,)][c])
    pres = [c for c in tot if tot[c][0] + tot[c][2]]
    print(f"\nCLASS-AWARE ENSEMBLE honest: {sum(WEIGHTS[c] * E.f1(*tot[c]) for c in pres) / sum(WEIGHTS[c] for c in pres):.4f}")
    for c in pres:
        tp, fp, fn = (int(v) for v in tot[c]); print(f"  {c:24s} F1 {E.f1(tp, fp, fn):.3f}  tp {tp} fp {fp} fn {fn}")
    for fam in grid:
        print(f"  {fam:14s} chosen w per fold: {[chosen[(fam, f)][0] for f in folds]}")


if __name__ == "__main__":
    main()
