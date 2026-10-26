"""Build the platform archive: build/submission/ → build/koz3_submission.zip.

  python scripts/build_submission.py --geo work/geo/geo_f*.pt --li work/li/li_f*.pt --sp_or work/spor/hm_f*.pt \
      [--ae ...] [--or ...] [--config work/config.json] [--image wooferclaw/archeology:latest]

Weights are exported to TorchScript (the platform image has torch but not smp / timm); only the koz3 modules
the experts import are copied. The archive must stay under 2 GB.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
import sys
import zipfile

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
KOZ3_MODULES = ["__init__", "dtm", "relief", "hm", "sp_io", "geo_prep", "geo_train", "li_train", "sp_train", "labels", "fuse", "peaks", "geo_feats", "geo_finds_core", "geo_data", "gbm_np", "geo_objects_core", "prev_sp_io", "prev_sp_det", "sp_fuse", "det_train", "dedupe", "ae_det_train", "tv_det_train"]
EXPERT_BUILD = {   # expert dir → (builder module, example input channels)
    "geo": ("koz3.geo_train", 3),
    "li": ("koz3.li_train", 9),
    "sp_or": ("koz3.sp_train", 3),
    "ae": ("koz3.sp_train", 3),
    "or": ("koz3.sp_train", 3),
    "sp_scene": ("koz3.scene_cls", 3),     # whole-scene hillfort gate (small satellite screenshots)
}


import torch


def export_ts(module, ch, src, dst):
    """Trace on the GPU (a CPU trace of the timm ConvNeXt U-Net bakes a CPU tensor into the decoder), check the
    trace against eager on the GPU at a different size, store fp16 (load_ts restores fp32)."""
    import importlib
    m = importlib.import_module(module).build_model(pretrained=False).eval()
    m.load_state_dict({k: v.float() for k, v in torch.load(src, map_location="cpu").items()})
    m = m.cuda()
    with torch.no_grad():
        ts = torch.jit.trace(m, torch.randn(1, ch, 256, 256, device="cuda"))
        x = torch.randn(1, ch, 320, 448, device="cuda")
        assert float((m(x) - ts(x)).abs().max()) < 1e-3, src
    ts.half()
    torch.jit.save(ts, dst)
    del m, ts
    torch.cuda.empty_cache()


def export_det(ckpt, srcs, d):
    """HF D-FINE / RT-DETRv2 run eagerly (TorchScript of these graphs takes > 10 min per call): save the model
    config (5 classes, no pretrained backbone download) and fp16 state dicts; inference rebuilds with from_config."""
    from transformers import AutoConfig
    from koz3.sp_train import CLS
    cfg = AutoConfig.from_pretrained(ckpt, num_labels=len(CLS), id2label=dict(enumerate(CLS)), label2id={c: i for i, c in enumerate(CLS)})
    cfg.use_pretrained_backbone = False
    if not any("denoising_class_embed" in k for k in torch.load(srcs[0], map_location="cpu")):
        cfg.num_denoising = 0            # trained without denoising queries (DET_NUM_DN=0): no denoising embedding
    cfg.save_pretrained(d)
    for i, s_ in enumerate(srcs):
        sd = torch.load(s_, map_location="cpu")
        torch.save({k: v.half() if v.is_floating_point() else v for k, v in sd.items()}, os.path.join(d, f"w_{i}.pt"))


def main():
    ap = argparse.ArgumentParser()
    for k in EXPERT_BUILD:
        ap.add_argument(f"--{k}", nargs="*", default=[])
    ap.add_argument("--sp_dfine", nargs="*", default=[])
    ap.add_argument("--sp_rtdetr", nargs="*", default=[])
    ap.add_argument("--config")
    ap.add_argument("--extra_geo", nargs="*", default=[])
    ap.add_argument("--image", default="wooferclaw/archeology:latest")
    ap.add_argument("--out", default=os.path.join(ROOT, "build"))
    a = ap.parse_args()
    sub = os.path.join(a.out, "submission")
    shutil.rmtree(sub, ignore_errors=True)
    shutil.copytree(os.path.join(ROOT, "inference"), os.path.join(sub, "inference"),
                    ignore=shutil.ignore_patterns("__pycache__", "models"))
    os.makedirs(os.path.join(sub, "inference", "koz3"))
    for m in KOZ3_MODULES:
        shutil.copy(os.path.join(ROOT, "koz3", m + ".py"), os.path.join(sub, "inference", "koz3", m + ".py"))
    for k, (mod, ch) in EXPERT_BUILD.items():
        srcs = sorted(p for pat in getattr(a, k) for p in glob.glob(pat))
        if not srcs:
            continue
        d = os.path.join(sub, "inference", "models", k); os.makedirs(d)
        for i, s in enumerate(srcs):
            export_ts(mod, ch, s, os.path.join(d, f"{k}_{i}.ts"))
        print(f"{k}: {len(srcs)} models")
    for f in (a.extra_geo or []):           # finds GBM json etc.
        d = os.path.join(sub, "inference", "models", "geo"); os.makedirs(d, exist_ok=True); shutil.copy(f, d)
    shutil.copytree(os.path.join(ROOT, "build", "vendor"), os.path.join(sub, "inference", "vendor"),
                    ignore=shutil.ignore_patterns("__pycache__", "bin"))
    for k, ckpt in (("sp_dfine", "ustc-community/dfine-large-coco"), ("sp_rtdetr", "PekingU/rtdetr_v2_r50vd")):
        srcs = sorted(p for pat in getattr(a, k) for p in glob.glob(pat))
        if srcs:
            d = os.path.join(sub, "inference", "models", k); os.makedirs(d)
            export_det(ckpt, srcs, d)
            print(f"{k}: {len(srcs)} models")
    if a.config:
        shutil.copy(a.config, os.path.join(sub, "inference", "models", "config.json"))
    json.dump({"image": a.image, "entry_point": "python inference/main.py"}, open(os.path.join(sub, "metadata.json"), "w"), indent=1)
    open(os.path.join(sub, "requirements.txt"), "w").write(
        "# runtime (all present in the image): torch numpy scipy rasterio shapely pyproj laspy opencv\n")
    zp = os.path.join(a.out, "koz3_submission.zip")
    with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
        for dp, _, fs in os.walk(sub):
            for f in fs:
                p = os.path.join(dp, f)
                z.write(p, os.path.relpath(p, sub))
    size = os.path.getsize(zp)
    print(f"{zp}: {size / 1e9:.3f} GB", "OK" if size < 2e9 else "TOO BIG")
    for p in ("inference/main.py", "inference/koz3_infer/run.py", "inference/models/config.json"):
        fp = os.path.join(sub, p)
        if os.path.exists(fp):
            print("md5", p, hashlib.md5(open(fp, "rb").read()).hexdigest())


if __name__ == "__main__":
    main()
