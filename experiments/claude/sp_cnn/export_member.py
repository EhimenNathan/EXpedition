"""Export out-of-fold heat maps (sp_cnn/train.py or ensemble.py output) as a v21 satellite fusion member:
work/spor/oof_det_<name>_cands.parquet with region, cls, score, x, y (EPSG:3857), read by
scripts/sp_assemble_oof.models(extra=[<name>]).
Peaks (sp_dense/evaluate.decode_scene, PEAK_K window) → kurgans emitted as BOTH classes with score heat·share and
heat·(1 − share) (≥ 0.02; the fusion keeps per-class maxima for the damaged / intact hedge), architecture and
fortifications with their heat. No hillfort rows (hillforts come from the localiser / gate).
Every region is predicted by a model that never saw its 20 km site cluster, so the file is out of fold for any fold
scheme built on site clusters.
  PEAK_K=7 python experiments/claude/sp_cnn/export_member.py <oof_dir> <name>
"""
import glob, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(HERE, "..", "sp_dense"))
K = ("kurgany_povrezhdennye", "kurgany_tselye")


def main(oof_dir, name):
    from rasterio.transform import Affine
    from evaluate import decode_scene
    rows = []
    for p in sorted(glob.glob(os.path.join(ROOT, oof_dir, "*.npz"))):
        z = np.load(p, allow_pickle=True)
        s = dict(T=Affine(*z["T"]), crs=str(z["crs"]), valid=z["valid"], region=str(z["region"]), tif=str(z["tif"]))
        C = decode_scene(z["p"], s, tmin=0.02)
        for r in C.itertuples():
            if r.fam == "k":
                for c, sc in ((K[0], r.score * r.share), (K[1], r.score * (1 - r.share))):
                    if sc >= 0.02:
                        rows.append((r.region, c, float(sc), r.x, r.y, r.tif))
            elif r.fam in ("arkhitektury", "fortifikatsii"):
                rows.append((r.region, r.fam, float(r.score), r.x, r.y, r.tif))
    D = pd.DataFrame(rows, columns=["region", "cls", "score", "x", "y", "tif"])
    out = os.path.join(ROOT, f"work/spor/oof_det_{name}_cands.parquet"); os.makedirs(os.path.dirname(out), exist_ok=True)
    D.to_parquet(out); print(out, len(D), D.cls.value_counts().to_dict())


if __name__ == "__main__":
    main(*sys.argv[1:])
