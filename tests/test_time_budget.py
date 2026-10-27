"""Regression check: a hung expert still yields one complete result file; time pressure is shared, not starved.

Runs inference/main.py in a subprocess with fake experts (no weights, no GPU):
1. watchdog: an expert that never returns → result.geojson written once by the watchdog, process exits 0;
2. budget sharing: a slow first expert must leave time for the second instead of consuming the whole budget.
"""
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import textwrap

ROOT = Path(__file__).resolve().parents[1]

DRIVER = textwrap.dedent('''
    import sys, time, types
    sys.path.insert(0, {inf!r}); sys.path.insert(1, {root!r})
    from shapely.geometry import Point, mapping
    import koz3_infer.run as R
    import koz3_infer.models as M
    M.cfg = lambda: {{}}                 # no dedupe / fusion / expert config needed

    def feat(region, cls):
        return {{"type": "Feature", "geometry": mapping(Point(0, 0).buffer(1)),
                 "properties": {{"class_name": cls, "region_name": region.name, "confidence": 0.9}}}}

    def make(name, behaviour):
        m = types.ModuleType("koz3_infer.experts." + name)
        m.FAMILIES = [name]
        m.CALLS = []
        def predict(region, paths, deadline):
            m.CALLS.append(deadline - time.time())
            behaviour(region, deadline)
            return [feat(region, name)]
        m.predict = predict
        sys.modules[m.__name__] = m
        return m

    mode = sys.argv[3]
    if mode == "hang":
        make("fast", lambda r, d: None)
        make("hang", lambda r, d: time.sleep(3600))
        R.EXPERTS = ["fast", "hang"]
    else:                                 # "share": 'slow' honours its deadline, 'late' must still get time
        make("slow", lambda r, d: time.sleep(max(0.0, min(3.0, d - time.time()))))
        late = make("late", lambda r, d: None)
        R.EXPERTS = ["slow", "late"]
        R.RATES = {{"slow": (0.5, 0.0), "late": (0.5, 0.0)}}
    R.region_files = lambda r: {{n: [r / "x.bin"] for n in R.EXPERTS}}
    import main
    main.main(["main.py", sys.argv[1], sys.argv[2]])
    if mode == "share":
        print("LATE_CALLS", len(late.CALLS))
''')


def run(mode, env_extra):
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for r in ("r1", "r2", "r3"):
            (tmp / "in" / r).mkdir(parents=True)
            (tmp / "in" / r / "x.bin").write_bytes(b"0")
        (tmp / "driver.py").write_text(DRIVER.format(inf=str(ROOT / "inference"), root=str(ROOT)))
        env = {**os.environ, **env_extra}
        p = subprocess.run([sys.executable, str(tmp / "driver.py"), str(tmp / "in"), str(tmp / "out"), mode],
                           capture_output=True, text=True, timeout=120, env=env)
        files = sorted(x.name for x in (tmp / "out").iterdir()) if (tmp / "out").exists() else []
        res = json.loads((tmp / "out" / "result.geojson").read_text()) if "result.geojson" in files else None
        return p, files, res


if __name__ == "__main__":
    p, files, res = run("hang", {"KOZ3_HARD_S": "4", "KOZ3_BUDGET_S": "600"})
    assert p.returncode == 0, p.stderr[-2000:]
    assert files == ["result.geojson"], files            # no temp file left behind, written exactly once
    assert len(res["features"]) == 3, len(res["features"])   # the fast expert's 3 regions survive the hang
    assert "WATCHDOG" in p.stderr, p.stderr[-2000:]

    p, files, res = run("share", {"KOZ3_HARD_S": "60", "KOZ3_BUDGET_S": "6"})
    assert p.returncode == 0, p.stderr[-2000:]
    assert "LATE_CALLS 3" in p.stdout, (p.stdout, p.stderr[-2000:])   # every late job ran despite the slow expert
    assert len(res["features"]) == 6, len(res["features"])
    print("PASS: hung expert → watchdog writes the 3 collected features once; "
          "slow expert leaves budget for the next expert (3/3 jobs ran)")
