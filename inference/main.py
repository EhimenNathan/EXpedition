"""KOZ-3 entry point.

Platform call: python inference/main.py /data/input /data/output/result.geojson
The second argument may also be a directory (then <dir>/result.geojson is written).

Zero-score rule (organisers, 22 Oct): the platform scores the FIRST result.geojson it sees once its
mtime is stable for 2 s. So results stay in memory until every expert has finished (or the time budget
is spent); then the FeatureCollection is written to temp_result.geojson in the same directory, fsynced,
and moved into place with a single os.replace. result.geojson never exists in a partial state.

Watchdog: if run() has not returned by KOZ3_HARD_S (default 28 min; the platform kills at 30), a daemon thread
post-processes and writes whatever has been collected, then exits the process. A single hung expert (an unusually
large point cloud, a stuck read) would otherwise mean no result file at all, which scores zero.
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # koz3_infer + koz3 (copied in by the build)
# packages the platform image may lack (pandas, transformers & deps): cp312 wheels vendored by the build. Prepended:
# the image's requirements are not published, and an older transformers there could predate D-FINE / our v5 config,
# so the exact tested versions win. (numpy, torch, rasterio, ... are not vendored and come from the image.)
_VENDOR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vendor")
if sys.version_info[:2] == (3, 12):        # wheels are cp312: prepend on the platform's Python
    sys.path.insert(1, _VENDOR)
else:                                      # other Pythons (local rehearsals) keep their own installs
    sys.path.append(_VENDOR)
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from koz3_infer.common import log, resolve_output_path, write_result_atomic  # noqa: E402
from koz3_infer.run import run  # noqa: E402


HARD_S = float(os.environ.get("KOZ3_HARD_S", 28 * 60))


def _watchdog(t0, out_path):
    time.sleep(max(0.0, t0 + HARD_S - time.time()))
    from koz3_infer.run import COLLECTED, finalize
    log(f"WATCHDOG at {time.time() - t0:.1f} s: run() still busy, writing {len(COLLECTED)} collected features")
    try:
        feats = finalize(list(COLLECTED))
    except Exception as e:
        log(f"watchdog finalize failed {e!r}; writing raw features")
        feats = list(COLLECTED)
    if write_result_atomic(feats, out_path):
        log(f"watchdog wrote {len(feats)} features at {time.time() - t0:.1f} s")
    os._exit(0)


def main(argv):
    t0 = time.time()
    in_dir = argv[1] if len(argv) > 1 else "/data/input"
    out_arg = argv[2] if len(argv) > 2 else "/data/output/result.geojson"
    out_path = resolve_output_path(out_arg)
    features = []
    from koz3_infer.models import cfg
    if cfg().get("probe") == "lidar":      # exit-code probe: no result, the bitmask is the exit code
        from koz3_infer.probe import run as probe_run
        sys.exit(probe_run(in_dir))
    diag = bool(cfg().get("diagnostic"))
    if not diag:
        threading.Thread(target=_watchdog, args=(t0, out_path), daemon=True).start()
    if diag:
        from koz3_infer import diag as D
        D.environment(); D.inputs(in_dir)
    try:
        features = run(in_dir, t0)
    except Exception as e:  # never die without writing: a partial answer scores, nothing scores zero
        import traceback
        traceback.print_exc()
        log(f"run() failed: {e!r}")
        from koz3_infer.run import COLLECTED
        features = list(COLLECTED)
    finally:
        if diag:   # diagnostic build: report and fail on purpose (the platform returns logs only for failed runs)
            D.summary(features)
            log(f"DIAGNOSTIC RUN finished at {time.time() - t0:.1f} s; no result written on purpose")
            sys.exit(3)
        if write_result_atomic(features, out_path):
            log(f"wrote {len(features)} features to {out_path} at {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main(sys.argv)
