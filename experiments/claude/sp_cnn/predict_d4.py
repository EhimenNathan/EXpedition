"""Re-predict out-of-fold maps with the saved fold models and full d4 test-time augmentation (8 flips / rotations,
inverse-transformed and averaged) instead of the h-flip pair. Same output format as train.py.
  CNN_WORK=work/sp_cnn KOZ3_MOD=SpOr python experiments/claude/sp_cnn/predict_d4.py <tag> <out_tag>
"""
import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import train as T  # noqa: E402


def predict_d4(model, img, valid, tile=512, step=384):
    import torch
    H, W = int(valid[0]), int(valid[1]); gh, gw = -(-H // T.STRIDE), -(-W // T.STRIDE)
    acc = np.zeros((T.NCH, gh, gw), np.float32); cnt = np.zeros((gh, gw), np.float32)
    rs = list(range(0, max(H - tile, 0) + 1, step)); rs += [max(H - tile, 0)] if rs[-1] != max(H - tile, 0) else []
    cs = list(range(0, max(W - tile, 0) + 1, step)); cs += [max(W - tile, 0)] if cs[-1] != max(W - tile, 0) else []
    n = tile // T.STRIDE; wy = np.minimum(np.arange(n) + 1, n - np.arange(n)).astype(np.float32); wgt = np.minimum.outer(wy, wy)
    for r in rs:
        for c in cs:
            a = np.zeros((tile, tile, 3), np.float32); part = np.asarray(img[r:r + tile, c:c + tile], np.float32)
            a[:part.shape[0], :part.shape[1]] = part
            x = torch.from_numpy(((a - T.MEAN) / T.STD).transpose(2, 0, 1)[None].copy())
            views = [torch.rot90(x, k, (2, 3)) for k in range(4)]; views += [torch.flip(v, (3,)) for v in views]
            with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
                o = torch.sigmoid(model(torch.cat(views)).float())
            outs = [torch.rot90(o[k], -k, (1, 2)) for k in range(4)] + [torch.rot90(torch.flip(o[4 + k], (2,)), -k, (1, 2)) for k in range(4)]
            o = torch.stack(outs).mean(0).numpy()
            gr, gc = r // T.STRIDE, c // T.STRIDE; h, w = min(n, gh - gr), min(n, gw - gc)
            acc[:, gr:gr + h, gc:gc + w] += o[:, :h, :w] * wgt[:h, :w]; cnt[gr:gr + h, gc:gc + w] += wgt[:h, :w]
    return acc / np.maximum(cnt, 1e-6)


def main(tag="resnet18", out_tag="resnet18_d4", arch="resnet18", seed=0):
    import geopandas as gpd, torch
    from rasterio.transform import Affine
    from koz3.sp_prep import site_folds
    torch.set_num_threads(os.cpu_count())
    L = gpd.read_parquet(os.path.join(T.ROOT, "data/labels_3857.parquet")); fo = site_folds(L[L["mod"] == T.MOD])
    S = T.load_all(); out = os.path.join(T.W_, "oof_" + out_tag); os.makedirs(out, exist_ok=True)
    for f in sorted({fo[s["region"]] for s in S}):
        m = T.build(arch, pretrained=False); m.load_state_dict(torch.load(os.path.join(T.W_, "models", f"{tag}_f{f}_s{int(seed)}.pt"), map_location="cpu")); m.eval()
        for s in [s for s in S if fo[s["region"]] == f]:
            if os.path.exists(os.path.join(out, s["idx"] + ".npz")):
                continue
            p = predict_d4(m, s["img"], s["valid"]); Tg = Affine(*s["T"]) * Affine.scale(T.STRIDE, T.STRIDE)
            np.savez_compressed(os.path.join(out, s["idx"] + ".npz"), p=p.astype(np.float16), T=np.array(Tg[:6]), crs=s["crs"],
                                valid=np.array(p.shape[1:]), region=s["region"], tif=s["tif"])
        print(f"fold {f} done", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
