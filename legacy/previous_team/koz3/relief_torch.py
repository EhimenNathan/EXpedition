"""GPU relief channels from a padded DTM batch. All outputs are site-independent (trend-removed or derivative)
and roughly unit-scaled so one network sees comparable inputs everywhere.
Input:  dtm (B,1,H+2P,W+2P) metres, gsd metres/px, P = PAD_M/gsd.  Output: (B,C,H,W) float32."""
import math
import torch
import torch.nn.functional as F

PAD_M = 30.0  # metres of context needed by the widest filter (3 * 8.5 m LoG sigma, 20 m LRM sigma clipped)
CHANNELS = ["lrm2", "lrm7", "lrm20", "slope", "log2", "log5", "log12", "hs315", "hs45", "svf", "opos", "oneg"]


def _gauss1d(sigma_px, device):
    r = max(1, int(math.ceil(3 * sigma_px)))
    x = torch.arange(-r, r + 1, device=device, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / sigma_px) ** 2); return k / k.sum(), r


def gblur(x, sigma_px):
    k, r = _gauss1d(sigma_px, x.device)
    x = F.pad(x, (r, r, r, r), mode="replicate")
    x = F.conv2d(x, k.view(1, 1, 1, -1)); return F.conv2d(x, k.view(1, 1, -1, 1))


def grads(x, gsd):
    gx = (F.pad(x, (1, 1, 0, 0), mode="replicate")[..., :, 2:] - F.pad(x, (1, 1, 0, 0), mode="replicate")[..., :, :-2]) / (2 * gsd)
    gy = (F.pad(x, (0, 0, 1, 1), mode="replicate")[..., 2:, :] - F.pad(x, (0, 0, 1, 1), mode="replicate")[..., :-2, :]) / (2 * gsd)
    return gx, gy


def hillshade(gx, gy, az, alt=35.0):
    slope = torch.atan(torch.sqrt(gx ** 2 + gy ** 2)); aspect = torch.atan2(-gx, -gy)
    az, alt = math.radians(az), math.radians(alt)
    hs = math.sin(alt) * torch.cos(slope) + math.cos(alt) * torch.sin(slope) * torch.cos(az - aspect)
    return hs.clamp(0, 1)


def horizon(x, gsd, radius_m=10.0, n_dir=8, sign=1.0):
    """Mean max-elevation angle (radians) over n_dir directions; sign=-1 for nadir (negative openness)."""
    a = sign * x
    steps = sorted({max(1, int(round(s))) for s in torch.logspace(0, math.log10(radius_m / gsd), 10).tolist()})
    acc = torch.zeros_like(a)
    for k in range(n_dir):
        th = 2 * math.pi * k / n_dir; dx, dy = math.cos(th), -math.sin(th)
        m = torch.full_like(a, -math.pi / 2)
        for s in steps:
            ox, oy = int(round(dx * s)), int(round(dy * s))
            sh = torch.roll(a, shifts=(-oy, -ox), dims=(-2, -1))
            m = torch.maximum(m, torch.atan((sh - a) / (math.hypot(ox, oy) * gsd)))
        acc = acc + m
    return acc / n_dir


@torch.no_grad()
def relief(dtm, gsd=0.25, crop_px=None):
    """dtm: (B,1,H,W) with nodata filled. Returns (B,12,H',W') cropped by crop_px on each side."""
    x = dtm.float()
    s = lambda m: m / gsd
    lrm = [x - gblur(x, s(m)) for m in (2.5, 7.5, 20.0)]
    gx, gy = grads(x, gsd)
    slope = torch.atan(torch.sqrt(gx ** 2 + gy ** 2)) * (180 / math.pi)
    logs = []
    for r in (2.0, 5.0, 12.0):
        sig = s(r / math.sqrt(2)); g = gblur(x, sig)
        lap = F.conv2d(F.pad(g, (1, 1, 1, 1), mode="replicate"), torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], device=x.device, dtype=torch.float32).view(1, 1, 3, 3))
        logs.append(-lap * sig ** 2 / 1.0)  # scale-normalised, metres
    hz = horizon(x, gsd, 10.0, 8, 1.0); nz = horizon(x, gsd, 10.0, 8, -1.0)
    svf = 1 - torch.sin(hz.clamp(min=0))
    ch = [lrm[0] * 2.0, lrm[1] * 1.5, lrm[2] * 1.0, slope / 30.0, logs[0] * 4.0, logs[1] * 2.0, logs[2] * 1.0,
          hillshade(gx, gy, 315) * 2 - 1, hillshade(gx, gy, 45) * 2 - 1, (svf - 0.9) * 10,
          (math.pi / 2 - hz) * (2 / math.pi) * 4 - 3, (math.pi / 2 - nz) * (2 / math.pi) * 4 - 3]
    out = torch.cat(ch, 1).clamp(-5, 5)
    if crop_px:
        out = out[..., crop_px:-crop_px, crop_px:-crop_px]
    return out
