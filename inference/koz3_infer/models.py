"""Model / config locations inside the submission (inference/models/<expert>/*.pt, inference/models/config.json)."""
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
_CFG = None

DEFAULT_CFG = {
    "geo": {"f_thr": 0.15, "r_find": 0.5, "o_thr": 0.3, "o_min_m2": 0.5, "input": "rgb", "finds_source": "gbm"},
    "li": {"thr": 0.15, "dual": 0.05, "comp_thr": 0.5, "fort_min_px": 30, "gor_min_px": 600, "r_out": 1.0},
    "sp_or": {"thr": 0.15, "dual": 0.4, "r_out": 1.0, "fused_thr": 0.25, "match_m": 8.0, "gate_thr": 0.1},
    "ae": {"det_thr": 0.7, "classes": ["kurgany_povrezhdennye"], "n_models": 2, "r_out": 1.0},
    "or": {"det_thr": 0.5, "classes": ["arkhitektury"], "n_models": 2, "r_out": 1.0},
    # cross-modal twins: LiDAR detections copied into co-registered optical layers (nested OOF 0.3480 → 0.3654 on 16 regions)
    # organisers: overlapping images of one data type → one object (duplicates are false positives)
    "dedupe": {"enabled": True, "dist_m": 3.0, "skip_markups": ["li"], "skip_classes": ["finds_points"]},
    "fuse": {"enabled": True, "beta": 0.0, "alpha": {"li->sp_or": 0.1, "li->ae": 0.1}},
}


def model_dir(name):
    return os.path.join(ROOT, name)


def cfg():
    global _CFG
    if _CFG is None:
        _CFG = json.loads(json.dumps(DEFAULT_CFG))
        f = os.path.join(ROOT, "config.json")
        if os.path.exists(f):
            for k, v in json.load(open(f)).items():
                if isinstance(v, dict):
                    _CFG.setdefault(k, {}).update(v)
                else:
                    _CFG[k] = v
    return _CFG


def ts_files(name):
    d = model_dir(name)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".ts")) if os.path.isdir(d) else []


def load_ts(path):
    """TorchScript models: inference needs only torch (no smp / timm in the platform image)."""
    import torch
    return torch.jit.load(path, map_location="cuda").float().eval()   # stored fp16, run fp32 (+autocast)
