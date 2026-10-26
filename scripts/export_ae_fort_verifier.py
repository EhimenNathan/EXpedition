"""Fit the final OOF-stacked Ae fort verifier and export one TorchScript module."""
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

ROOT = Path("/home/nathan/Expedition Competition")
d = pd.read_parquet(ROOT / "Codex Work/ae_fort_crop_records.parquet")
X = np.load(ROOT / "Codex Work/ae_fort_crop_features.npz")["dino"].astype(np.float32)
sc = StandardScaler().fit(X)
lr = LogisticRegression(C=.003, class_weight="balanced", max_iter=3000).fit(sc.transform(X), d.target)


class Verifier(torch.nn.Module):
    def __init__(self, backbone, mean, scale, coef, intercept):
        super().__init__(); self.backbone = backbone
        self.register_buffer("mean", torch.from_numpy(mean.astype(np.float32)))
        self.register_buffer("scale", torch.from_numpy(scale.astype(np.float32)))
        self.register_buffer("coef", torch.from_numpy(coef.astype(np.float32)))
        self.register_buffer("intercept", torch.tensor(float(intercept), dtype=torch.float32))

    def forward(self, x):
        z = self.backbone(x).float()
        return torch.sigmoid(((z - self.mean) / self.scale * self.coef).sum(1) + self.intercept)


backbone = timm.create_model("vit_small_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                             dynamic_img_size=True).cuda().eval()
m = Verifier(backbone, sc.mean_, sc.scale_, lr.coef_[0], lr.intercept_[0]).cuda().eval()
example = torch.zeros(1, 3, 224, 224, device="cuda")
with torch.inference_mode():
    traced = torch.jit.trace(m, example, strict=False)
    a = m(example).cpu(); b = traced(example).cpu()
    assert torch.allclose(a, b, atol=1e-5), (a, b)
out = ROOT / "work/ae/ae_fort_verifier.ts"
traced.save(str(out))
print(out, out.stat().st_size, "train prior", float(d.target.mean()), "coef norm", float(np.linalg.norm(lr.coef_)))
