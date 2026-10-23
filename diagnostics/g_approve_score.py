"""Score effect of the approve stacking diagnostic, paired over 3 seeds (train OOF only; diagnostic, not a deployable model)."""
from common import *   # noqa: F401,F403
from e_score_sim import score_from, quantile_abstain, pipeline, GRID_A
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

rows = load_train()
S = [r["state"] for r in rows]
amt = np.array([s["amount"] for s in S]); tier = np.array([["free", "pro", "enterprise"].index(s["customer_tier"]) for s in S])
caps = np.array([32.0, 122.0, 500.0])[tier]
deltas = []
for seed in (13, 14, 15):
    Z, yr, ya, yg, fold = load_cache(seed)
    base = pipeline(Z, yr, ya, yg, seed=seed)
    X = np.column_stack([Z[:, 5], amt, tier, amt == 0, amt / caps, np.minimum(amt / caps, 1.5)])
    p = np.zeros(len(ya))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, ya):
        g = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0)
        g.fit(X[tr], ya[tr]); p[te] = g.predict_proba(X[te])[:, 1]
    p = np.clip(p, M.P_EPS, 1 - M.P_EPS)
    P = crossfit_route_any(Z[:, :5], yr, "temperature", seed=seed)
    pg = crossfit_binary_any(Z[:, 6], yg, "platt", seed=seed)
    full = np.array([T.approve_cost(ya, p >= t) for t in GRID_A]); thr = T.pick_threshold(GRID_A, -full, 0.75)
    gab = quantile_abstain(np.maximum(pg, 1 - pg), 0.83)
    qg = np.array([T.grounded_quality(yg, pg, float(t), gab) for t in np.round(np.arange(.3, .701, .01), 2)])
    gthr = T.pick_threshold(np.round(np.arange(.3, .701, .01), 2), qg, 0.5)
    for rule in ("min_conf", "max_risk"):
        ab = quantile_abstain(np.maximum(p, 1 - p), 0.83) if rule == "min_conf" else quantile_abstain(-M.approve_risk(p), 0.83)
        r = score_from(P, yr, p, ya, pg, yg, quantile_abstain(P.max(1), 0.83), ab, gab, thr, gthr)
        print("seed %d approve=stacked(fine amount) rule=%-8s total %.2f (deployed %.2f, %+.2f) | approve quality %.3f vs %.3f | approve abst %.3f vs %.3f | approve skill %.3f vs %.3f"
              % (seed, rule, r["total"], base["total"], r["total"] - base["total"], r["q"][1], base["q"][1],
                 r["abst_parts"][1], base["abst_parts"][1], r["skill"][1], base["skill"][1]))
        if rule == "min_conf": deltas.append(r["total"] - base["total"])
print("mean delta (min_conf rule): %+.2f   (seed std of the baseline total itself: ~0.24)" % np.mean(deltas))
