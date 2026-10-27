"""LiDAR hillfort localiser, stage 1: whole-region DINOv3-SAT tokens of the 0.5 m relief image (work/li_cnn/img, as
li_dense/extract.relief_image), longer side SIDE px, identity + h-flip; labels = token centre inside a LiDAR hillfort.
Same output format as sp_hillfort_dinov3_tokens.py, so sp_hillfort_dinov3_eval.py runs with KOZ3_MOD=Li.
  python experiments/claude/li_dense/hillfort_tokens.py <out.npz> <side>
"""
import glob, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.join(HERE, "..", "..", "..")
sys.path.insert(0, ROOT)
PATCH = 16


def main(out, side):
    import cv2, geopandas as gpd, torch
    from pyproj import Transformer
    from rasterio.transform import Affine
    from shapely import contains_xy
    from transformers import AutoModel, AutoImageProcessor
    side = int(side); torch.set_num_threads(os.cpu_count())
    MODEL = "facebook/dinov3-vitl16-pretrain-sat493m"; tk = os.environ.get("HF_TOKEN")
    proc = AutoImageProcessor.from_pretrained(MODEL, token=tk); mean, std = np.array(proc.image_mean, np.float32), np.array(proc.image_std, np.float32)
    model = AutoModel.from_pretrained(MODEL, token=tk).eval(); skip = 1 + model.config.num_register_tokens
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); L = L[L["mod"] == "Li"]
    HF = {r: g.geometry.union_all() for r, g in L[L.class_name == "gorodishcha"].groupby("region")}
    Tok, Y, POS, SID, GRID, META = [], [], [], [], [], []
    for k, m in enumerate(sorted(glob.glob(os.path.join(ROOT, "work/li_cnn/meta/*.npz")))):
        z = np.load(m, allow_pickle=True); region = str(z["region"]); h, w = (int(v) for v in z["valid"])
        img = np.load(m.replace("/meta/", "/img/").replace(".npz", ".npy"), mmap_mode="r")[:h, :w]
        s = side / max(h, w); gh, gw = max(1, round(h * s / PATCH)), max(1, round(w * s / PATCH))
        im = cv2.resize(np.ascontiguousarray(img), (gw * PATCH, gh * PATCH), interpolation=cv2.INTER_AREA)
        T = Affine(*z["T"]) * Affine.scale(w / (gw * PATCH), h / (gh * PATCH))
        views = []
        for flip in (False, True):
            x = im[:, ::-1] if flip else im
            x = torch.from_numpy(((x.astype(np.float32) / 255 - mean) / std).transpose(2, 0, 1).copy())[None]
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                t = model(pixel_values=x).last_hidden_state[0, skip:].float().numpy().reshape(gh, gw, -1)
            views.append(t[:, ::-1] if flip else t)
        tok = np.mean(views, 0).reshape(gh * gw, -1)
        rr, cc = np.mgrid[0:gh, 0:gw]
        X, Yp = T * ((cc.ravel() + 0.5) * PATCH, (rr.ravel() + 0.5) * PATCH)
        X3, Y3 = Transformer.from_crs(str(z["crs"]), 3857, always_xy=True).transform(np.asarray(X), np.asarray(Yp))
        y = contains_xy(HF[region], X3, Y3) if region in HF else np.zeros(gh * gw, bool)
        Tok.append(tok.astype(np.float16)); Y.append(y); SID.append(np.full(gh * gw, k)); GRID.append((gh, gw))
        POS.append(np.c_[(cc.ravel() + 0.5) / gw * 2 - 1, (rr.ravel() + 0.5) / gh * 2 - 1, X3, Y3]); META.append((region, region, 0.5, h, w))
        print(k + 1, region[:24], gh, gw, int(y.sum()), flush=True)
    np.savez(os.path.join(ROOT, out), tok=np.concatenate(Tok), y=np.concatenate(Y), pos=np.concatenate(POS), sid=np.concatenate(SID),
             grid=np.array(GRID), meta=np.array(META, dtype=object))


if __name__ == "__main__":
    main(*sys.argv[1:])
