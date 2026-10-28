"""Build v23 = the exact v22 archive + the DINOv3-SAT satellite hillfort localiser (nested OOF 0.3797 → 0.4097).

Everything in v22 stays byte-identical except:
  inference/koz3_infer/experts/sp_or.py      localiser block inserted after the scene-gate centre (verified anchor)
  inference/koz3/hf_dino_loc.py              localiser module (offline)
  inference/models/sp_dino_loc/              backbone (fp16, or int8 with --int8) + hf_dino_loc.npz (probe, 1 MB)
  inference/models/config.json               sp_or.dino_loc = {"dir": "sp_dino_loc"}
Checks: parity of the packed localiser vs the out-of-fold token path on N training scenes (point distance in metres,
map-max difference), seconds per scene, archive < 2 GB.
  HF_HUB_OFFLINE=1 python scripts/make_v23_from_v22.py <v22.zip> <v23.zip> [--int8] [--n 8]
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
ANCHOR = """    X, Y = Transformer.from_crs(crs, 3857, always_xy=True).transform(*(t * (w / 2, h / 2)))
    return p, X, Y"""
BLOCK = '''    X, Y = Transformer.from_crs(crs, 3857, always_xy=True).transform(*(t * (w / 2, h / 2)))
    dl = cfg()["sp_or"].get("dino_loc")             # DINOv3-SAT hillfort localiser: point at the predicted hillfort
    if dl and p >= cfg()["sp_or"].get("gate_thr", 0.1):   # instead of the scene centre (nested OOF 0.3797 → 0.4097)
        try:
            loc = _localiser(dl)
            if loc is not None:
                X, Y, mx = loc(str(path), crs)
                log(f"sp_or {path.name}: DINOv3 hillfort point (map max {mx:.3f})")
        except Exception as e:                          # never lose the scene: keep the centre
            log(f"sp_or {path.name}: DINOv3 localiser failed {e!r}; scene centre kept")
    return p, X, Y


def _localiser(dl):
    """Load once: backbone (local folder) + probe weights from inference/models/<dl['dir']>/."""
    if "loc" not in _GATE:
        import os
        import torch
        from koz3.hf_dino_loc import Localiser
        from ..models import model_dir
        d = model_dir(dl.get("dir", "sp_dino_loc"))
        bb, w = os.path.join(d, "backbone"), os.path.join(d, "hf_dino_loc.npz")
        _GATE["loc"] = Localiser(bb, w, device="cuda" if torch.cuda.is_available() else "cpu") \\
            if os.path.isdir(bb) and os.path.exists(w) else None
    return _GATE["loc"]'''


def patch_sp_or(path):
    s = open(path, encoding="utf-8").read()
    assert "_localiser" not in s, "sp_or.py already contains the localiser"
    assert s.count(ANCHOR) == 1, "scene-gate anchor not found exactly once — v22 sp_or.py differs; stop"
    assert "from ..models import cfg" in s and "log" in s, "expected imports (cfg, log) missing"
    open(path, "w", encoding="utf-8").write(s.replace(ANCHOR, BLOCK))


def int8_backbone(d):
    """Per-tensor symmetric int8 for 2-D weights (stored with scales; dequantised to fp16 at load by hf_dino_loc)."""
    import torch
    from safetensors.torch import load_file, save_file
    for f in glob.glob(os.path.join(d, "*.safetensors")):
        t = load_file(f); q = {}
        for k, v in t.items():
            if v.ndim == 2 and v.numel() > 4096:
                s_ = v.float().abs().amax().clamp(min=1e-8) / 127
                q[k + ".q8"] = (v.float() / s_).round().clamp(-127, 127).to(torch.int8); q[k + ".q8scale"] = s_.reshape(1)
            else:
                q[k] = v
        os.remove(f); save_file(q, f.replace(".safetensors", ".q8.safetensors"))


def parity(inf_dir, n):
    """Packed localiser (imported from the unpacked archive) vs the out-of-fold token path with the same final probe."""
    import torch
    from scipy.ndimage import gaussian_filter, zoom
    sys.path.insert(0, os.path.join(inf_dir)); sys.modules.pop("koz3", None); sys.modules.pop("koz3.hf_dino_loc", None)
    from koz3.hf_dino_loc import Localiser, probe_map
    d = os.path.join(inf_dir, "models", "sp_dino_loc")
    loc = Localiser(os.path.join(d, "backbone"), os.path.join(d, "hf_dino_loc.npz"), device="cuda" if torch.cuda.is_available() else "cpu")
    Z5 = np.load(os.path.join(ROOT, "work/hf_dinov3/tokens.npz"), allow_pickle=True)
    Z7 = np.load(os.path.join(ROOT, "work/hf_dinov3/tokens768.npz"), allow_pickle=True)
    import rasterio
    out = []
    for s in np.linspace(0, len(Z7["meta"]) - 1, n).astype(int):
        meta = Z7["meta"][s]; path = glob.glob(os.path.join(ROOT, "data/train-expds-3", meta[0], "04_*", meta[1]))[0]
        ms = []
        for side, Z in ((512, Z5), (768, Z7)):
            idx = Z["sid"] == s; gh, gw = Z["grid"][s]; ms.append(probe_map(loc.W, side, Z["tok"][idx].astype(np.float32), gh, gw))
        gh, gw = ms[-1].shape
        m = np.mean([zoom(a, (gh / a.shape[0], gw / a.shape[1]), order=1) if a.shape != (gh, gw) else a for a in ms], 0)
        g = gaussian_filter(m, 0.5, mode="nearest"); a = int(np.argmax(g)); pos = Z7["pos"][Z7["sid"] == s]
        with rasterio.open(path) as ds:
            crs = ds.crs.to_string()
        t = time.time(); x, y, mx = loc(path, crs); dt = time.time() - t
        out.append((meta[0][:22], float(np.hypot(x - pos[a, 2], y - pos[a, 3])), abs(mx - float(g.max())), dt))
        print(f"   {out[-1][0]:22s} point Δ {out[-1][1]:7.2f} m (3857)  map-max Δ {out[-1][2]:.4f}  {dt:.1f}s", flush=True)
    return out


def parity_only(inf_dir, n):
    """Child process: block PIL / torchvision (absent from the platform image) and put the archive's vendored packages
    first on the path (as inference/main.py does), then run the parity check and print its rows as JSON."""
    for m in ("PIL", "torchvision"):
        sys.modules[m] = None
    vend = os.path.join(inf_dir, "vendor")
    if os.path.isdir(vend):
        sys.path.insert(0, vend)
    P = parity(inf_dir, n)
    import transformers
    print("transformers", transformers.__version__, "from", os.path.dirname(transformers.__file__))
    print("PARITY_JSON", json.dumps(P))


def main():
    if "--parity-only" in sys.argv:
        i = sys.argv.index("--parity-only"); n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 8
        return parity_only(sys.argv[i + 1], n)
    ap = argparse.ArgumentParser()
    ap.add_argument("v22"); ap.add_argument("v23"); ap.add_argument("--int8", action="store_true"); ap.add_argument("--n", type=int, default=8)
    a = ap.parse_args()
    from koz3.hf_dino_loc import save_backbone
    tmp = a.v23 + ".dir"; shutil.rmtree(tmp, ignore_errors=True)
    with zipfile.ZipFile(a.v22) as z:
        z.extractall(tmp)
    inf = os.path.join(tmp, "inference")
    v22_size = os.path.getsize(a.v22); print(f"v22 {a.v22}: {v22_size / 1e9:.3f} GB")
    patch_sp_or(os.path.join(inf, "koz3_infer", "experts", "sp_or.py"))
    shutil.copy(os.path.join(ROOT, "koz3", "hf_dino_loc.py"), os.path.join(inf, "koz3", "hf_dino_loc.py"))
    d = os.path.join(inf, "models", "sp_dino_loc"); os.makedirs(d, exist_ok=True)
    save_backbone(out=os.path.join(d, "backbone"))
    if a.int8:
        int8_backbone(os.path.join(d, "backbone"))
    shutil.copy(os.path.join(ROOT, "models", "hf_dino_loc.npz"), os.path.join(d, "hf_dino_loc.npz"))
    cp = os.path.join(inf, "models", "config.json"); c = json.load(open(cp, encoding="utf-8"))
    c.setdefault("sp_or", {})["dino_loc"] = {"dir": "sp_dino_loc"}
    json.dump(c, open(cp, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("parity on training scenes in a PLATFORM-LIKE subprocess (no PIL / torchvision, the archive's vendor first):")
    import subprocess
    r = subprocess.run([sys.executable, os.path.abspath(__file__), "--parity-only", inf, "--n", str(a.n)], capture_output=True, text=True,
                       env={**os.environ, "HF_HUB_OFFLINE": "1"})
    print("\n".join(l for l in (r.stdout + r.stderr).splitlines() if "warn" not in l.lower() and "loading" not in l.lower())[-3000:])
    if r.returncode != 0 or "PARITY_JSON" not in r.stdout:
        print("PARITY FAILED: the packed localiser does not run in a platform-like environment — do not submit"); sys.exit(1)
    P = json.loads(r.stdout.split("PARITY_JSON")[-1].strip())
    moved = sum(p[1] > 1.0 for p in P)
    print(f"scenes whose point moved > 1 m: {moved}/{len(P)}, worst map-max Δ {max(p[2] for p in P):.4f}, "
          f"mean {np.mean([p[3] for p in P]):.1f}s/scene")
    if moved > len(P) // 4 or max(p[2] for p in P) > 0.05:   # a rare one-token flip from GPU / CPU rounding is tolerable
        print("PARITY FAILED: the packed localiser does not reproduce the validated one — do not submit"); sys.exit(1)
    if os.path.exists(a.v23):
        os.remove(a.v23)
    with zipfile.ZipFile(a.v23, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, fs in os.walk(tmp):
            for f in fs:
                p = os.path.join(dp, f); z.write(p, os.path.relpath(p, tmp))
    size = os.path.getsize(a.v23)
    print(f"v23 {a.v23}: {size / 1e9:.3f} GB (+{(size - v22_size) / 1e9:.3f})", "OK" if size < 2e9 else "TOO BIG (re-run with --int8)")
    for p in ("inference/main.py", "inference/koz3_infer/experts/sp_or.py", "inference/models/config.json"):
        print("md5", p, hashlib.md5(open(os.path.join(tmp, p), "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
