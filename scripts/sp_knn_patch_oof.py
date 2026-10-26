"""DINOv2-B patch tokens (cached, 152 satellite small scenes, 27×27 patches): patch-level "contains a kurgan" scoring with
(a) linear probe, (b) k-NN semantic search (faiss, cosine, k=20), (c) sparse random projection 768→96 + k-NN.
Trained on the other site folds only; patch AUC and scene-level kurgan AUC (max patch score)."""
import os, sys
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "scripts"))


def main():
    import faiss, geopandas as gpd
    from shapely import STRtree
    from shapely.geometry import Point
    from sklearn.linear_model import LogisticRegression
    from sklearn.random_projection import SparseRandomProjection
    from sklearn.metrics import roc_auc_score
    from koz3.scene_cls import scenes
    from sp_gor_localise_oof import patch_geo, G
    chips = scenes(); T = np.load(os.path.join(ROOT, "work/spor/dinov2b_patch_tokens.npy")).astype(np.float32)
    L = gpd.read_parquet(os.path.join(ROOT, "data/labels_3857.parquet")); Lk = L[(L["mod"] == "SpOr") & L.class_name.isin(["kurgany_povrezhdennye", "kurgany_tselye"])]
    geoms = list(Lk.geometry.buffer(8)); tree = STRtree(geoms)          # patch ≈ 14–37 m: kurgan within 8 m of the patch centre
    Y = np.zeros((len(chips), G * G), np.int8); M = np.zeros((len(chips), G * G), bool)
    for i, c in enumerate(chips):
        mx, my, ins = patch_geo(c); M[i] = ins.ravel()
        for k, (a, b) in enumerate(zip(mx.ravel(), my.ravel())):
            if ins.ravel()[k]:
                pt = Point(a, b); Y[i, k] = any(geoms[j].contains(pt) for j in tree.query(pt))
    fold = np.array([c["fold"] for c in chips]); print(int(Y[M].sum()), "kurgan patches of", int(M.sum()))
    res = {m: np.zeros(Y.shape, np.float32) for m in ("probe", "knn", "rp+knn")}
    for f in sorted(set(fold)):
        tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
        Xtr = T[tr].reshape(-1, T.shape[-1])[M[tr].ravel()]; ytr = Y[tr].ravel()[M[tr].ravel()]
        mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
        clf = LogisticRegression(C=0.01, max_iter=3000, class_weight="balanced").fit((Xtr - mu) / sd, ytr)
        for i in te:
            res["probe"][i] = clf.predict_proba((T[i] - mu) / sd)[:, 1]
        for nm, proj in (("knn", None), ("rp+knn", SparseRandomProjection(96, random_state=0).fit(Xtr))):
            A = Xtr if proj is None else proj.transform(Xtr).astype(np.float32)
            A = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-6); idx = faiss.IndexFlatIP(A.shape[1]); idx.add(A)
            for i in te:
                Q = T[i] if proj is None else proj.transform(T[i]).astype(np.float32)
                Q = Q / (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-6); _, I = idx.search(np.ascontiguousarray(Q), 20)
                res[nm][i] = ytr[I].mean(1)
    yk = Y.reshape(len(chips), -1).max(1) if False else np.array([int(any(cl in (0, 1) for cl, _ in c["objs"])) for c in chips])
    for nm, R in res.items():
        print(f"{nm:7s} patch AUC {roc_auc_score(Y[M], R[M]):.3f}   scene kurgan AUC (max patch) {roc_auc_score(yk, np.where(M, R, -1).max(1)):.3f}")
    print("reference: scene-level kurgan probes ConvNeXt 0.798 / DINOv2 CLS 0.767")


if __name__ == "__main__":
    main()
