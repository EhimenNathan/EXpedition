"""Fit the per-expert job cost priors used by koz3_infer.run (overhead_s + s_per_mb × input MB).

Usage: python scripts/calibrate_budget.py REHEARSAL_LOG INPUT_DIR
REHEARSAL_LOG is the stderr of a full-quality rehearsal (lines "<expert> <region> <n> feats <t>s");
INPUT_DIR is the validation input it ran on.  Prints the "budget" block for inference/models/config.json.
Fit on the rehearsal hardware closest to the platform (RTX 4090); priors only need the right ratios between
experts because measured speed replaces them as soon as each expert has run.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "inference"))
from koz3_infer.common import region_files  # noqa: E402

LINE = re.compile(r"^\[\s*[\d.]+s\]\s+(\w+)\s+(\S+)\s+\d+ feats\s+([\d.]+)s")


def main(log_path, in_dir):
    from koz3_infer.run import EXPERTS, _jobs_for
    import importlib
    in_dir = Path(in_dir)
    plan = {}
    for r in sorted(d for d in in_dir.iterdir() if d.is_dir()):
        for fam, paths in region_files(r).items():
            plan.setdefault(fam, []).append((r, paths))
    mb = {}
    for name in EXPERTS:
        fams = importlib.import_module(f"koz3_infer.experts.{name}").FAMILIES
        for region, paths in _jobs_for(type("M", (), {"FAMILIES": fams}), plan):
            mb[(name, region.name)] = sum(p.stat().st_size for p in paths) / 1e6
    obs = {}
    for line in open(log_path, encoding="utf-8", errors="replace"):
        m = LINE.match(line.strip())
        if m and (m[1], m[2]) in mb:
            obs.setdefault(m[1], []).append((mb[(m[1], m[2])], float(m[3])))
    out = {}
    for name, xy in obs.items():
        x, y = np.array(xy).T
        A = np.stack([np.ones_like(x), x], 1)
        (over, per_mb), *_ = np.linalg.lstsq(A, y, rcond=None)
        if per_mb < 0 or over < 0:                  # degenerate fit: fall back to a pure per-MB rate
            over, per_mb = 0.0, float(y.sum() / max(x.sum(), 1e-6))
        out[name] = [round(float(over), 2), round(float(per_mb), 4)]
        print(f"{name:6s} n={len(x):3d} total {y.sum():7.1f}s  overhead {over:6.2f}s  {per_mb:.4f} s/MB", file=sys.stderr)
    print(json.dumps({"budget": out}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:3])
