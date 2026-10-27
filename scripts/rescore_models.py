"""Second-stage candidate models shared by sp_rescore_oof.py and li_rescore_oof.py (out-of-fold, fold = held-out group).

Kinds (comma list in SP_RESCORE_MODELS / LI_RESCORE_MODELS; default "lgbm" = the original scripts):
  lgbm        LightGBM, the original settings
  catboost    CatBoost (ordered boosting: less target leakage / overfitting on small data)
  ngboost     NGBoost, Bernoulli (natural-gradient probabilistic boosting)
  avg         rank mean of lgbm and catboost
  <kind>+anom the same model with two anomaly features added:
              if_region  IsolationForest score within the candidate's own region (unsupervised, inference-time only)
              oc_pos     IsolationForest fitted on the TRAINING folds' true objects only (one-class "looks like a real
                         object"), cross-fitted so no row is scored by a model that saw it
"""
import numpy as np


def _oc_source(y, fo, f, g):
    """Rows whose positives fit the one-class forest that scores fold g within outer fold f: never fold f (held out) nor
    fold g (the rows being scored)."""
    return (y == 1) & (fo != f) & (fo != g)


def _anomaly(D, feats, fold, label):
    from sklearn.ensemble import IsolationForest
    X = D[feats].to_numpy(np.float32); X = np.nan_to_num(X)
    reg = D.region.to_numpy(); y = D[label].to_numpy(); fo = D[fold].to_numpy()
    ifr = np.zeros(len(D))
    for r in np.unique(reg):
        m = reg == r
        if m.sum() >= 20:
            ifr[m] = IsolationForest(n_estimators=100, random_state=0).fit(X[m]).score_samples(X[m])
            ifr[m] -= np.median(ifr[m])
    oc = {}                                                   # (outer fold f) → oc_pos for all rows, cross-fitted
    folds = np.unique(fo)
    for f in folds:
        v = np.zeros(len(D))
        for g in folds:
            src = _oc_source(y, fo, f, g)
            if src.sum() >= 10:
                v[fo == g] = IsolationForest(n_estimators=200, random_state=0).fit(X[src]).score_samples(X[fo == g])
        oc[f] = v
    return ifr, oc


def _model(kind):
    if kind == "lgbm":
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=40, subsample=0.8,
                                  subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=0, verbose=-1)
    if kind == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=600, depth=5, learning_rate=0.05, l2_leaf_reg=5, random_seed=0, verbose=0,
                                  thread_count=-1, allow_writing_files=False)
    if kind == "ngboost":
        from ngboost import NGBClassifier
        from ngboost.distns import Bernoulli
        from sklearn.tree import DecisionTreeRegressor
        return NGBClassifier(Dist=Bernoulli, n_estimators=400, learning_rate=0.03, verbose=False, random_state=0,
                             Base=DecisionTreeRegressor(criterion="squared_error", max_depth=4, min_samples_leaf=40))
    raise ValueError(kind)


def oof_predict(D, feats, kind, fold="fold", label="lab"):
    """→ out-of-fold P(label) for every row of D (each fold predicted by a model fitted on the other folds)."""
    from scipy.stats import rankdata
    anom = kind.endswith("+anom"); base = kind[:-5] if anom else kind
    if base == "avg":
        a = oof_predict(D, feats, "lgbm" + ("+anom" if anom else ""), fold, label)
        b = oof_predict(D, feats, "catboost" + ("+anom" if anom else ""), fold, label)
        return (rankdata(a) + rankdata(b)) / (2 * len(a))
    X = np.nan_to_num(D[feats].to_numpy(np.float32)); y = D[label].to_numpy(); fo = D[fold].to_numpy()
    if anom:
        ifr, oc = _anomaly(D, feats, fold, label)
    p = np.full(len(D), np.nan)
    for f in np.unique(fo):
        tr, te = fo != f, fo == f
        Xf = np.c_[X, ifr, oc[f]] if anom else X
        p[te] = _model(base).fit(Xf[tr], y[tr]).predict_proba(Xf[te])[:, 1]
    return p


def self_test():
    """Synthetic 4-fold data where the label depends on two features: every kind must beat chance out of fold, and the
    one-class feature must not leak (a fold's rows are never scored by a forest fitted on that fold's positives)."""
    import pandas as pd
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(0); n = 2000
    D = pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n), "c": rng.normal(size=n)})
    D["lab"] = (D.a + 0.5 * D.b + 0.5 * rng.normal(size=n) > 1.0).astype(int)
    D["fold"] = np.arange(n) % 4; D["region"] = np.arange(n) % 40
    for kind in ("lgbm", "catboost", "ngboost", "avg", "lgbm+anom"):
        auc = roc_auc_score(D.lab, oof_predict(D, ["a", "b", "c"], kind))
        assert auc > 0.85, (kind, auc)
        print(f"  {kind:10s} OOF AUC {auc:.3f}")
    fo = D.fold.to_numpy(); y = D.lab.to_numpy()
    for f in range(4):
        for g in range(4):
            src = _oc_source(y, fo, f, g)
            assert not src[fo == f].any() and not src[fo == g].any() and src.any(), (f, g)
    _, oc = _anomaly(D, ["a", "b", "c"], "fold", "lab")
    assert set(oc) == {0, 1, 2, 3} and all(np.isfinite(v).all() for v in oc.values())
    print("self-test PASS: all kinds learn out of fold; anomaly features cross-fitted")


if __name__ == "__main__":
    self_test()
