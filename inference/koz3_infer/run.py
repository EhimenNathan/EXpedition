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
# Prior cost of one job: overhead_s + s_per_mb × input MB.  Only the ratio between experts matters until the first
# jobs are measured; scripts/calibrate_budget.py fits them from a rehearsal log (config "budget" overrides these).
RATES = {"geo": (5.0, 0.5), "las": (20.0, 0.25), "sp_or": (5.0, 0.15), "ae": (10.0, 0.3), "or_": (20.0, 0.05)}


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


def _job_cost(name, paths, rates):
    over, per_mb = rates.get(name, (10.0, 0.2))
    return over + per_mb * sum(p.stat().st_size for p in paths) / 1e6


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
    rates = {**RATES, **{k: tuple(v) for k, v in cfg().get("budget", {}).items()}}
    queue = []                      # (name, mod, region, paths, prior cost s) in execution order
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
        queue += [(name, mod, region, paths, _job_cost(name, paths, rates)) for region, paths in jobs]
    log(f"{len(queue)} jobs, prior cost {sum(q[4] for q in queue):.0f}s for a {BUDGET_S:.0f}s budget")
    spent, prior = {}, {}           # measured vs prior seconds per expert → speed ratio for its remaining jobs
    for i, (name, mod, region, paths, cost) in enumerate(queue):
        now = time.time()
        if now > deadline:
            log(f"budget spent: skipping {name} on {region.name}")
            continue
        # Speed ratio: an expert's own measurement once it has run, else the slowest ratio seen so far
        # (a slow machine or unusually heavy data slows every expert; never assume faster than the prior).
        seen = [spent[k] / prior[k] for k in spent if prior[k] > 0]
        default = max([1.0] + seen)
        ratio = lambda k: spent[k] / prior[k] if prior.get(k) else default
        reserve = sum(c * ratio(n) for n, _, _, _, c in queue[i + 1:])
        predicted = cost * ratio(name) + reserve
        # Preserve the deployed full-quality path byte for byte while all remaining work fits.  Reserve deadlines
        # only after measured speed shows real pressure; otherwise a harmless prior reserve can trigger an expert's
        # own quality degradation even with many minutes of slack.
        if predicted <= deadline - now:
            job_deadline = deadline
        else:
            job_deadline = min(deadline, max(deadline - reserve, now + 0.3 * cost * ratio(name)))
        if predicted > deadline - now and job_deadline < deadline - 1:
            log(f"time pressure: {name} on {region.name} gets {job_deadline - now:.0f}s ({reserve:.0f}s reserved)")
        ts = time.time()
        try:
            feats = mod.predict(region, paths, job_deadline)
            COLLECTED.extend(feats)
            log(f"{name:12s} {region.name:28s} {len(feats):5d} feats {time.time() - ts:6.1f}s (prior {cost:.0f}s)")
        except Exception:
            traceback.print_exc()
            log(f"{name} failed on {region.name}")
        spent[name] = spent.get(name, 0.0) + time.time() - ts
        prior[name] = prior.get(name, 0.0) + cost
    return finalize(list(COLLECTED))


def finalize(feats):
    """Region dedupe within data type, then cross-modal fusion.  Also used by the watchdog on partial results."""
    raw = list(feats)
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
        feats = raw
    return feats
