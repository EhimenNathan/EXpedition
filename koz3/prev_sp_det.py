"""(Ported from the previous team; model call swapped to TorchScript.) Satellite detector (D-FINE, HF transformers) shared by training OOF (scripts/train_sp_det.py) and inference
(inference/run.py), so the scored code and the submitted code are the same.

* build(config_dir, weight_files, device): models from a saved DFineConfig (no hub access) + fold state dicts.
* detect_scene(models, img, T, ...): tiles of `chip` px with `ov` overlap over a (4, H, W) uint8 scene (RGB + validity,
  koz3.sp_io.read_scene / scripts/prep_sp.py), flip TTA, all fold models pooled, at most `cap` detections per tile and
  model, per-class NMS across tiles and models, boxes -> inscribed ellipses in the scene CRS (affine T).
  Returns [(class index, shapely polygon, score)]."""
import numpy as np
import torch
from shapely import affinity
from shapely.geometry import Point


@torch.no_grad()
def _detect(model, x, amp):
    """TorchScript detector (koz3.det_train._Wrap trace: x → (logits, pred_boxes)); flip TTA as in training OOF."""
    with torch.autocast(device_type=x.device.type, dtype=torch.float16, enabled=amp):
        l1, b1 = model(x); l2, b2 = model(torch.flip(x, [3]))
    s1, b1 = l1.float().sigmoid(), b1.float(); s2, b2 = l2.float().sigmoid(), b2.float()
    b2 = b2.clone(); b2[..., 0] = 1 - b2[..., 0]
    return [(torch.cat([s1[i], s2[i]]), torch.cat([b1[i], b2[i]])) for i in range(len(x))]


@torch.no_grad()
def detect_scene(models, img, T, chip=640, ov=160, min_score=0.05, cap=100, bs=8, nms_iou=0.5, time_left=None):
    import torchvision
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu"); amp = dev.type == "cuda"
    H, W = img.shape[1:]; step = chip - ov
    tiles = [(y, x) for y in range(0, max(H - ov, 1), step) for x in range(0, max(W - ov, 1), step)]
    B_, S_, C_ = [], [], []
    for k in range(0, len(tiles), bs):
        if time_left is not None and time_left() < 60:
            break
        bt = tiles[k:k + bs]; X = []
        for y, x in bt:
            c = np.asarray(img[:, y:y + chip, x:x + chip])
            c = np.pad(c, ((0, 0), (0, chip - c.shape[1]), (0, chip - c.shape[2])))
            X.append(c[:3].astype(np.float32) / 255.0 * (c[3] > 0))
        xb = torch.from_numpy(np.stack(X)).to(dev)
        for m in models:
            for (y, x), (sc, bx) in zip(bt, _detect(m, xb, amp)):
                score, cls = sc.max(1); keep = score >= min_score
                if keep.sum() > cap:
                    keep &= score >= score[keep].topk(cap).values[-1]
                if not keep.any():
                    continue
                b = bx[keep] * chip
                B_.append(torch.stack([b[:, 0] - b[:, 2] / 2 + x, b[:, 1] - b[:, 3] / 2 + y, b[:, 0] + b[:, 2] / 2 + x, b[:, 1] + b[:, 3] / 2 + y], 1))
                S_.append(score[keep]); C_.append(cls[keep])
    if not B_:
        return []
    B, S, C = torch.cat(B_), torch.cat(S_), torch.cat(C_)
    out = []
    for i in torchvision.ops.batched_nms(B, S, C, nms_iou).tolist():
        x0, y0, x1, y1 = B[i].tolist()
        (X0, Y0), (X1, Y1) = T * (x0, y0), T * (x1, y1)
        e = affinity.scale(Point((X0 + X1) / 2, (Y0 + Y1) / 2).buffer(1.0, 16), abs(X1 - X0) / 2, abs(Y1 - Y0) / 2)
        if e.area > 0:
            out.append((int(C[i]), e, float(S[i])))
    return out
