"""LightGBM binary-classifier inference in numpy (the platform image has no lightgbm): booster.dump_model() JSON →
flat arrays → vectorised traversal. Exact for numerical splits (the only kind our models use); missing values follow
default_left."""
import json

import numpy as np


class NumpyGBM:
    def __init__(self, dump):
        d = json.load(open(dump)) if isinstance(dump, str) else dump
        self.features = d["feature_names"]
        self.trees = [self._flatten(t["tree_structure"]) for t in d["tree_info"]]

    @staticmethod
    def _flatten(root):
        feat, thr, left, right, dleft, val = [], [], [], [], [], []

        def add(n):
            i = len(feat)
            feat.append(-1); thr.append(0.0); left.append(-1); right.append(-1); dleft.append(True); val.append(0.0)
            if "leaf_value" in n:
                val[i] = n["leaf_value"]
                return i
            assert n.get("decision_type", "<=") == "<=", n.get("decision_type")
            feat[i] = n["split_feature"]; thr[i] = n["threshold"]; dleft[i] = n.get("default_left", True)
            left[i] = add(n["left_child"]); right[i] = add(n["right_child"])
            return i
        add(root)
        return tuple(np.array(a) for a in (feat, thr, left, right, dleft, val))

    def raw(self, X):
        try:
            import torch
            if torch.cuda.is_available():
                return self.raw_torch(X)
        except Exception:
            pass
        return self.raw_compact(X)

    def raw_torch(self, X, chunk=2_000_000):
        """Same traversal on the GPU (dense levels; leaves point to themselves). Exact up to float64 rounding."""
        import torch
        dev = "cuda"
        if not hasattr(self, "_torch_trees"):
            self._torch_trees = [tuple(torch.as_tensor(a, device=dev) for a in t[:6]) + (t[6],) for t in self._dense]
        out = np.zeros(len(X))
        for c0 in range(0, len(X), chunk):
            Xt = torch.as_tensor(np.asarray(X[c0:c0 + chunk], np.float64), device=dev)
            n = Xt.shape[0]; rows = torch.arange(n, device=dev); acc = torch.zeros(n, dtype=torch.float64, device=dev)
            for feat, thr, left, right, dleft, val, depth in self._torch_trees:
                node = torch.zeros(n, dtype=torch.long, device=dev)
                for _ in range(depth):
                    x = Xt[rows, feat[node]]
                    go_left = torch.where(torch.isnan(x), dleft[node], x <= thr[node])
                    node = torch.where(go_left, left[node], right[node])
                acc += val[node]
            out[c0:c0 + chunk] = acc.cpu().numpy()
        return out

    def raw_dense(self, X):
        """Dense fixed-depth traversal in numpy (slower than raw_compact on CPU; basis of raw_torch)."""
        X = np.asarray(X, np.float64)
        out = np.zeros(len(X))
        rows = np.arange(len(X))
        for feat, thr, left, right, dleft, val, depth in self._dense:
            node = np.zeros(len(X), np.int64)
            for _ in range(depth):
                f = feat[node]
                x = X[rows, f]
                go_left = np.where(np.isnan(x), dleft[node], x <= thr[node])
                node = np.where(go_left, left[node], right[node])
            out += val[node]
        return out

    @property
    def _dense(self):
        if not hasattr(self, "_dense_cache"):
            dc = []
            for feat, thr, left, right, dleft, val in self.trees:
                n = len(feat); idx = np.arange(n)
                leaf = feat < 0
                f2 = np.where(leaf, 0, feat); l2 = np.where(leaf, idx, left); r2 = np.where(leaf, idx, right)
                # depth of the tree (longest root-leaf path)
                depth = np.zeros(n, np.int64)
                for i in range(n - 1, -1, -1):
                    if not leaf[i]:
                        depth[i] = 1 + max(depth[left[i]], depth[right[i]])
                dc.append((f2, np.where(leaf, np.inf, thr), l2, r2, dleft, val, int(depth[0])))
            self._dense_cache = dc
        return self._dense_cache

    def raw_compact(self, X):
        X = np.asarray(X, np.float64)
        out = np.zeros(len(X))
        for feat, thr, left, right, dleft, val in self.trees:
            node = np.zeros(len(X), np.int64)
            while True:
                f = feat[node]
                inner = f >= 0
                if not inner.any():
                    break
                idx = np.nonzero(inner)[0]; nd = node[idx]
                x = X[idx, f[idx]]
                go_left = np.where(np.isnan(x), dleft[nd], x <= thr[nd])
                node[idx] = np.where(go_left, left[nd], right[nd])
            out += val[node]
        return out

    def predict_proba(self, X):
        """sklearn-compatible (n, 2) [P(0), P(1)] — the ported objects core indexes [:, 1]."""
        p = 1.0 / (1.0 + np.exp(-self.raw(X)))
        return np.stack([1 - p, p], 1)
