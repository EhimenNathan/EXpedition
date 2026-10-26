"""Orchestrates the experts over all regions within the time budget.

Features accumulate in COLLECTED (memory only); main.py writes them once at the end.
Each expert is a module in koz3_infer.experts with  FAMILY  and  predict(region, paths, deadline) -> [features].
An expert whose weights are missing is skipped, so a partial build still runs end to end.
"""
import importlib
import os
import time
import traceback

from .common import log, region_dirs, region_files

BUDGET_S = float(os.environ.get("KOZ3_BUDGET_S", 26 * 60))   # platform kills at 30 min
EXPERTS = ["geo", "las", "sp_or", "ae", "or_"]   # value per second: satellite (30 regions) before the few big Ae / Or sheets
COLLECTED = []


def _jobs_for(mod, plan):
    """Return one job per region across all families handled by an expert.

    Several experts inspect all of their supported inputs in one predict call.
    A region containing two such families must therefore be queued once, with
    the union of its paths, rather than once per family.  The latter repeats the
    whole inference pass and creates duplicate same-image predictions.
    """
    by_region = {}
    for fam in mod.FAMILIES:
        for region, paths in plan.get(fam, []):
            by_region.setdefault(region, set()).update(paths)
    return [(region, sorted(paths, key=str)) for region, paths in by_region.items()]


def run(in_dir, t0):
    deadline = t0 + BUDGET_S
    regions = region_dirs(in_dir)
    log(f"{len(regions)} regions in {in_dir}")
    plan = {}
    for r in regions:
        for fam, paths in region_files(r).items():
            plan.setdefault(fam, []).append((r, paths))
    log("data families: " + ", ".join(f"{k}={len(v)}" for k, v in sorted(plan.items())))
    from .models import cfg
    off = set(cfg().get("experts_off", []))
    for name in EXPERTS:
        if name in off:
            log(f"expert {name}: disabled by config")
            continue
        try:
            mod = importlib.import_module(f".experts.{name}", __package__)
        except Exception as e:
            log(f"expert {name} unavailable: {e!r}")
            continue
        if not getattr(mod, "READY", lambda: True)():
            log(f"expert {name}: weights missing, skipped")
            continue
        jobs = _jobs_for(mod, plan)
        jobs.sort(key=lambda j: sum(p.stat().st_size for p in j[1]))   # small regions first
        for region, paths in jobs:
            if time.time() > deadline:
                log(f"budget spent: skipping {name} on {region.name}")
                continue
            ts = time.time()
            try:
                feats = mod.predict(region, paths, deadline)
                COLLECTED.extend(feats)
                log(f"{name:12s} {region.name:28s} {len(feats):5d} feats {time.time() - ts:6.1f}s")
            except Exception:
                traceback.print_exc()
                log(f"{name} failed on {region.name}")
    feats = list(COLLECTED)
    try:
        from .models import cfg
        dd = cfg().get("dedupe", {})
        if dd.get("enabled"):
            from koz3.dedupe import dedupe_region
            # EtGr reference objects are defined independently for every source
            # map/profile (including depth layers).  Cross-image dedupe therefore
            # destroys valid one-to-one targets.  This is a metric/data-contract
            # invariant, rather than a tunable configuration choice.
            skip = set(dd.get("skip_markups", ["li"])) | {"etgr"}
            skip_cls = set(dd.get("skip_classes", ["finds_points"]))
            out_ = lambda f: f["properties"].get("markup_type") in skip or f["properties"]["class_name"] in skip_cls
            n0 = len(feats)
            feats = [f for f in feats if out_(f)] + dedupe_region([f for f in feats if not out_(f)], dd.get("dist_m", 3.0))
            log(f"region dedupe within data type: {n0} → {len(feats)} features")
    except Exception:
        traceback.print_exc()
    try:
        from .models import cfg
        fc = cfg().get("fuse", {})
        if fc.get("enabled"):
            from koz3.fuse import fuse
            alpha = {tuple(k.split("->")): v for k, v in fc.get("alpha", {}).items()}
            n0 = len(feats)
            feats = fuse(feats, alpha=alpha, beta=fc.get("beta", 0.5))
            log(f"cross-modal fusion: {n0} → {len(feats)} features")
    except Exception:
        traceback.print_exc()
        feats = list(COLLECTED)
    return feats
