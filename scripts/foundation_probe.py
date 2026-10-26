"""Foundation-backbone quick test (apples to apples): frozen backbones + linear probe on the satellite small scenes,
same 152 scenes and the same 4 site folds as koz3/scene_cls.py (deployed hillfort gate, fine-tuned ConvNeXt-T: OOF AUC
0.892). Tasks: "scene contains a hillfort", "scene contains kurgans". Features = backbone embedding (ViT: CLS ⊕ mean
patch token; CNN: pooled pre-logits), averaged over identity / h-flip / v-flip (as the deployed gate). Probe =
standardised logistic regression, C chosen by inner CV on the training folds only. Reports OOF AUC per backbone.

  python scripts/foundation_probe.py
"""
import os
import sys

import numpy as np
import torch

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

BACKBONES = {
    "convnext_tiny.in12k_ft_in1k (ImageNet-22k)": ("convnext_tiny.in12k_ft_in1k", 384),
    "DINOv2 ViT-S/14 (LVD-142M, self-supervised)": ("vit_small_patch14_dinov2.lvd142m", 378),
    "DINOv2 ViT-B/14 (LVD-142M, self-supervised)": ("vit_base_patch14_dinov2.lvd142m", 378),
}


@torch.no_grad()
def embed(name, size, chips):
    import timm
    import torch.nn.functional as F
    from koz3.scene_cls import load
    m = timm.create_model(name, pretrained=True, num_classes=0, img_size=size) if "vit" in name else timm.create_model(name, pretrained=True, num_classes=0)
    m = m.cuda().eval()
    out = []
    for c in chips:
        x = torch.from_numpy(load(c))[None].cuda()                       # 384 px, ImageNet normalisation (as deployed)
        if x.shape[-1] != size:
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        fs = []
        for fl in ((), (-1,), (-2,)):
            z = x.flip(fl) if fl else x
            if "vit" in name:
                t = m.forward_features(z)                                    # (1, 1 + N, D)
                fs.append(torch.cat([t[:, 0], t[:, 1:].mean(1)], 1))
            else:
                fs.append(m(z))
        out.append(torch.stack(fs).mean(0)[0].float().cpu().numpy())
    del m; torch.cuda.empty_cache()
    return np.stack(out)


def probe(X, y, fold):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score
    p = np.zeros(len(y))
    for f in sorted(set(fold)):
        tr, te = fold != f, fold == f
        best, bc = -1, None
        for C in (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0):                   # inner CV over the OTHER folds only
            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=5000, class_weight="balanced"))
            s = cross_val_score(pipe, X[tr], y[tr], groups=fold[tr], cv=GroupKFold(len(set(fold[tr]))), scoring="roc_auc").mean()
            if s > best:
                best, bc = s, C
        pipe = make_pipeline(StandardScaler(), LogisticRegression(C=bc, max_iter=5000, class_weight="balanced")).fit(X[tr], y[tr])
        p[te] = pipe.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, p)


def main():
    from koz3.scene_cls import scenes
    chips = scenes()
    y_gor = np.array([c["y"] for c in chips])
    y_kur = np.array([int(any(cl in (0, 1) for cl, _ in c["objs"])) for c in chips])
    fold = np.array([c["fold"] for c in chips])
    print(f"{len(chips)} scenes; hillfort {y_gor.sum()}, kurgans {y_kur.sum()}; folds {np.bincount(fold)}")
    print("reference: deployed fine-tuned ConvNeXt-T hillfort OOF AUC 0.892")
    for label, (name, size) in BACKBONES.items():
        try:
            X = embed(name, size, chips)
        except Exception as e:
            print(f"{label}: unavailable ({e!r})"); continue
        print(f"{label:46s} dim {X.shape[1]:5d}  linear-probe OOF AUC  hillfort {probe(X, y_gor, fold):.3f}  kurgans {probe(X, y_kur, fold):.3f}", flush=True)


if __name__ == "__main__":
    main()
