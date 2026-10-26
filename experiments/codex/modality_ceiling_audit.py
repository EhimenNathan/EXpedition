"""Validation-shaped headroom by modality, using saved exact-scorer OOF counts.

This is an oracle diagnostic, not a performance estimate: replacing a modality
with perfect detections reveals only its maximum possible contribution under
the current validation sampling mix.
"""
import json
import pickle
from pathlib import Path
import numpy as np
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT), str(ROOT / "scripts")]
from koz3.nested import weighted, _add_all

C = pickle.load(open(ROOT / "work/joint_thr_counts.pkl", "rb"))
COMP = {"SpOr": 30, "Li": 13, "Ae": 4, "MagGr": 2,
        "EtGr": 1, "TerFl": 1, "Or": 1}
ST = {"SpOr": 1.0, "Li": 0.15, "Ae": None, "Or": 0.45}
regs = {m: sorted({k[1] for k in C if k[0] == m}) for m in COMP}
rng = np.random.default_rng(20261025)
draws = [{m: list(rng.choice(regs[m], COMP[m], replace=len(regs[m]) < COMP[m]))
          for m in COMP} for _ in range(3000)]

def value(d, perfect=()):
    allc = []
    for m in COMP:
        for r in d[m]:
            c = C[(m, r, ST.get(m))]
            if m in perfect:
                c = {cl: (tp + fn, 0, 0) for cl, (tp, fp, fn) in c.items()}
            allc.append(c)
    return weighted(_add_all(allc))

base = np.array([value(d) for d in draws])
out = {"baseline_mean": float(base.mean()), "baseline_q05_q95": np.quantile(base, [.05, .95]).tolist(),
       "modality_oracle": {}}
for m in COMP:
    v = np.array([value(d, (m,)) for d in draws])
    delta = v - base
    out["modality_oracle"][m] = {"mean": float(v.mean()), "gain": float(delta.mean()),
                                  "gain_q05_q95": np.quantile(delta, [.05, .95]).tolist()}
print(json.dumps(out, indent=2, ensure_ascii=False))
(Path(__file__).with_suffix(".json")).write_text(json.dumps(out, indent=2, ensure_ascii=False))
