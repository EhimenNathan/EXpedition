"""Place the DINOv3-SAT hillfort localiser into the submission tree:
  inference/models/sp_dino_loc/backbone/   (facebook/dinov3-vitl16-pretrain-sat493m, fp16, ~0.6 GB, offline)
  inference/models/sp_dino_loc/hf_dino_loc.npz   (probe weights fitted on all training scenes, ~1 MB)
Needs the backbone in the local Hugging Face cache (or HF_TOKEN). Then checks one training scene end-to-end.
  python scripts/build_dino_loc_models.py [scene.tif]
"""
import os
import shutil
import sys
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)


def main(check=None):
    from koz3.hf_dino_loc import Localiser, save_backbone
    d = os.path.join(ROOT, "inference/models/sp_dino_loc"); os.makedirs(d, exist_ok=True)
    if not os.path.isdir(os.path.join(d, "backbone")):
        save_backbone(out=os.path.join(d, "backbone"))
    shutil.copy(os.path.join(ROOT, "models/hf_dino_loc.npz"), os.path.join(d, "hf_dino_loc.npz"))
    size = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(d) for f in fs)
    print(f"{d}: {size / 1e9:.2f} GB")
    if check:
        import torch
        loc = Localiser(os.path.join(d, "backbone"), os.path.join(d, "hf_dino_loc.npz"), device="cuda" if torch.cuda.is_available() else "cpu")
        t = time.time(); print("point", loc(check), f"{time.time() - t:.1f}s")


if __name__ == "__main__":
    main(*sys.argv[1:])
