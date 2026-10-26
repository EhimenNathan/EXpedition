"""Read-only audit of LiDAR DTM coverage versus held-out kurgan recall."""
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
S = json.loads((ROOT / "li_prior_robustness.json").read_text())["site_counts"]
rows = []
for site, rec in S.items():
    p = ROOT.parent / "data" / "las_prep" / f"{site}.npz"
    if not p.exists():
        continue
    with np.load(p) as z:
        valid = z["valid"]
        src_res = float(z["origin"][2])
        valid_fraction = float(valid.mean())
    c = rec["prior"]
    tp = sum(c[k][0] for k in ("kurgany_povrezhdennye", "kurgany_tselye"))
    fn = sum(c[k][2] for k in ("kurgany_povrezhdennye", "kurgany_tselye"))
    rows.append(dict(site=site, fold=rec["fold"], valid_fraction=valid_fraction,
                     source_resolution=src_res, kurgan_gt=tp + fn,
                     kurgan_recall=tp / (tp + fn) if tp + fn else None))
rows.sort(key=lambda r: r["site"])
(ROOT / "li_dtm_quality_audit.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False))
subset = [r for r in rows if r["kurgan_gt"] >= 3]
for name in ("valid_fraction", "source_resolution"):
    corr = spearmanr([r[name] for r in subset], [r["kurgan_recall"] for r in subset])
    print(name, "Spearman", round(float(corr.statistic), 3), "p", round(float(corr.pvalue), 4), flush=True)
print("sites", len(rows), "with >=3 GT", len(subset), flush=True)
for r in sorted(subset, key=lambda r: r["kurgan_recall"])[:15]:
    print(r, flush=True)
