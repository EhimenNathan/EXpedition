"""Map-level ensemble of the satellite CNN (2 m grid) and the dense DINOv3 heads (8 m tokens, bilinearly upsampled with
area-aligned grids: both grids start at the same 0.5 m image origin) → work/sp_cnn/oof_ens_<w>/ (self-describing maps).
p = w·CNN + (1 − w)·dense for channels 0–3 (kurgan heat, damaged share, architecture, fortification).
  python experiments/claude/sp_cnn/ensemble.py [w=0.5]
"""
import glob, os, sys
import numpy as np
from scipy.ndimage import zoom

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")


def main(w=0.5):
    w = float(w); src = os.environ.get("CNN_OOF", "resnet18"); suffix = "" if src == "resnet18" else "_" + src
    out = os.path.join(ROOT, f"work/sp_cnn/oof_ens_{w}{suffix}"); os.makedirs(out, exist_ok=True)
    for p in sorted(glob.glob(os.path.join(ROOT, f"work/sp_cnn/oof_{src}/*.npz"))):
        z = dict(np.load(p, allow_pickle=True)); c = z["p"].astype(np.float32)
        d = np.load(os.path.join(ROOT, "work/sp_dense/oof", os.path.basename(p)))["p"][:4].astype(np.float32)
        u = np.stack([zoom(ch, 4, order=1, grid_mode=True, mode="nearest") for ch in d])[:, :c.shape[1], :c.shape[2]]
        if u.shape[1:] != c.shape[1:]:
            u = np.pad(u, ((0, 0), (0, c.shape[1] - u.shape[1]), (0, c.shape[2] - u.shape[2])), mode="edge")
        z["p"] = (w * c + (1 - w) * u).astype(np.float16)
        np.savez_compressed(os.path.join(out, os.path.basename(p)), **z)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
