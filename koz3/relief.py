"""Relief channels from a DTM on the GPU (shared by LiDAR training and inference)."""
import numpy as np
import torch
import torch.nn.functional as F

import os

NET_RES = 0.5
EXTRA = os.environ.get("RELIEF_EXTRA") == "1"      # + positive / negative openness and sky-view factor (10 m horizon)
N_CH = 12 if EXTRA else 9
UNCLIP = os.environ.get("RELIEF_UNCLIP") == "1"    # local relief via asinh (no ±2 m saturation of tall mounds)


def _gauss(x, sigma_px):
    k = int(3 * sigma_px) | 1
    k = max(k, 3)
    r = torch.arange(k, device=x.device, dtype=x.dtype) - k // 2
    g = torch.exp(-r ** 2 / (2 * sigma_px ** 2)); g = g / g.sum()
    x = F.conv2d(F.pad(x, (k // 2, k // 2, 0, 0), mode="replicate"), g.view(1, 1, 1, k))
    return F.conv2d(F.pad(x, (0, 0, k // 2, k // 2), mode="replicate"), g.view(1, 1, k, 1))


def _gauss_big(x, sigma_px):
    """Large-sigma blur via downsample → blur → upsample (fast, accurate enough for trend removal)."""
    if sigma_px <= 12:
        return _gauss(x, sigma_px)
    f = int(sigma_px // 6)
    xs = F.avg_pool2d(F.pad(x, (0, (-x.shape[-1]) % f, 0, (-x.shape[-2]) % f), mode="replicate"), f)
    xs = _gauss(xs, sigma_px / f)
    return F.interpolate(xs, scale_factor=f, mode="bilinear", align_corners=False)[..., :x.shape[-2], :x.shape[-1]]


def _openness(z, r, radius_m=10.0, n_dir=8):
    """Positive / negative openness and sky-view factor (Yokoyama 2002; Zakšek 2011), direction-independent mound /
    ditch cues: per direction the max elevation (and max depression) angle to the horizon within radius_m."""
    K = max(2, int(round(radius_m / r)))
    H, W = z.shape[-2:]
    zp = F.pad(z, (K, K, K, K), mode="replicate")
    pos, neg, svf = 0, 0, 0
    for a in np.arange(n_dir) * 2 * np.pi / n_dir:
        ca, sa = np.cos(a), np.sin(a)
        up = torch.full_like(z, -np.pi / 2); dn = torch.full_like(z, -np.pi / 2)
        for k in range(1, K + 1, max(1, K // 10)):
            dy, dx = int(round(k * sa)), int(round(k * ca))
            d = np.hypot(dx, dy) * r
            if d == 0:
                continue
            zs = zp[..., K + dy:K + dy + H, K + dx:K + dx + W]
            ang = torch.atan((zs - z) / d)
            up = torch.maximum(up, ang); dn = torch.maximum(dn, -ang)
        pos = pos + (np.pi / 2 - up); neg = neg + (np.pi / 2 - dn); svf = svf + torch.sin(up.clamp(min=0))
    pos, neg, svf = pos / n_dir, neg / n_dir, 1 - svf / n_dir
    return [((pos - np.pi / 2) / 0.1).clamp(-4, 4), ((neg - np.pi / 2) / 0.1).clamp(-4, 4), ((svf - 0.97) / 0.03).clamp(-4, 4)]


@torch.no_grad()
def channels(dtm, valid, src_res, dev="cuda"):
    """dtm (H,W) float32 at src_res → (N_CH, h, w) float32 at NET_RES, plus valid (h,w) bool."""
    f = NET_RES / src_res
    h, w = int(round(dtm.shape[0] / f)), int(round(dtm.shape[1] / f))
    fi = int(round(f))
    if f > 1 and abs(f - fi) < 1e-6:   # integer factor: pad to a multiple so area bins are exact (no drift)
        py, px = (-dtm.shape[0]) % fi, (-dtm.shape[1]) % fi
        if py or px:
            dtm = np.pad(dtm, ((0, py), (0, px)), mode="edge"); valid = np.pad(valid, ((0, py), (0, px)))
    z = torch.from_numpy(np.ascontiguousarray(dtm)).to(dev)[None, None]
    v = torch.from_numpy(valid.astype(np.float32)).to(dev)[None, None]
    if abs(f - 1) > 1e-6:
        if f > 1 and abs(f - fi) < 1e-6:
            z = F.avg_pool2d(z, fi)[..., :h, :w]; v = F.avg_pool2d(v, fi)[..., :h, :w]
        else:
            z = F.interpolate(z, size=(h, w), mode="area" if f > 1 else "bilinear")
            v = F.interpolate(v, size=(h, w), mode="area" if f > 1 else "nearest")
    r = NET_RES
    outs = []
    for s_m in (2.0, 5.0, 10.0, 25.0):
        d_ = (z - _gauss_big(z, s_m / r)) / 0.5
        outs.append(torch.asinh(d_) if UNCLIP else d_.clamp(-4, 4))
    zs = _gauss(z, 0.75)
    gx = (zs[..., :, 2:] - zs[..., :, :-2]) / (2 * r); gx = F.pad(gx, (1, 1, 0, 0), mode="replicate")
    gy = (zs[..., 2:, :] - zs[..., :-2, :]) / (2 * r); gy = F.pad(gy, (0, 0, 1, 1), mode="replicate")
    slope = torch.atan(torch.sqrt(gx ** 2 + gy ** 2))
    outs.append(slope / 0.5)
    for az in (315.0, 45.0):
        a, alt = np.radians(az), np.radians(35.0)
        asp = torch.atan2(-gx, gy)
        hs = np.sin(alt) * torch.cos(slope) + np.cos(alt) * torch.sin(slope) * torch.cos(a - asp)
        outs.append(hs * 2 - 1)
    lap = _gauss(z, 1.0 / r)
    lap = F.conv2d(F.pad(lap, (1, 1, 1, 1), mode="replicate"),
                   torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=dev, dtype=z.dtype).view(1, 1, 3, 3)) / (r * r)
    outs.append((lap / 0.2).clamp(-4, 4))
    if EXTRA:
        outs += _openness(z, r)
    vb = (v > 0.5).float()
    outs.append(vb * 2 - 1)
    x = torch.cat(outs, 1)[0] * vb[0]
    return x.cpu().numpy().astype(np.float32), vb[0, 0].bool().cpu().numpy()


def channels_tiled(dtm, valid, src_res, tile_px=3072, margin_px=384, dev="cuda"):
    """Same output as channels() but computed in overlapping windows (bounded GPU memory for multi-km sites).
    Window size / margin are in NET_RES pixels; the margin covers the widest filter (25 m Gaussian, 3σ = 150 px)."""
    f = NET_RES / src_res
    H, W = int(round(dtm.shape[0] / f)), int(round(dtm.shape[1] / f))
    if H * W <= tile_px * tile_px:
        return channels(dtm, valid, src_res, dev)
    out = np.zeros((N_CH, H, W), np.float32); vout = np.zeros((H, W), bool)
    for y0 in range(0, H, tile_px):
        for x0 in range(0, W, tile_px):
            ya, xa = max(0, y0 - margin_px), max(0, x0 - margin_px)
            yb, xb = min(H, y0 + tile_px + margin_px), min(W, x0 + tile_px + margin_px)
            sy0, sx0 = int(round(ya * f)), int(round(xa * f))
            sh, sw = int(round((yb - ya) * f)), int(round((xb - xa) * f))      # exactly f × output size: bins align
            dw = dtm[sy0:sy0 + sh, sx0:sx0 + sw]; vw = valid[sy0:sy0 + sh, sx0:sx0 + sw]
            py, px = sh - dw.shape[0], sw - dw.shape[1]
            if py or px:
                dw = np.pad(dw, ((0, py), (0, px)), mode="edge"); vw = np.pad(vw, ((0, py), (0, px)))
            x, v = channels(np.ascontiguousarray(dw), vw, src_res, dev)
            h, w = min(tile_px, H - y0), min(tile_px, W - x0)
            oy, ox = y0 - ya, x0 - xa
            hh, ww = min(h, x.shape[1] - oy), min(w, x.shape[2] - ox)
            out[:, y0:y0 + hh, x0:x0 + ww] = x[:, oy:oy + hh, ox:ox + ww]
            vout[y0:y0 + hh, x0:x0 + ww] = v[oy:oy + hh, ox:ox + ww]
    return out, vout
