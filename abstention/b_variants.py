"""
Step 2: FIXED list of confidence-ranking variants, paired against candidate B on identical OOF folds.
Everything is cross-fitted (any second stage is fitted on other rows and applied to held-out rows) on train OOF logits.

    python abstention/b_variants.py screen     # seeds 13,14,15 (whole list)
    python abstention/b_variants.py confirm V  # seeds 16,17,18, ONE pre-selected variant V
"""
import json
import sys

from common_abs import *   # noqa: F401,F403
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

prob = problem()
yr, ya, yg = prob.y_route, prob.y_app, prob.y_gr
n = prob.n
S = prob.states
HAS_MK = np.array([bool(MARKER_RE.search(s["body"])) for s in S])
TIER_I = np.array([M.TIERS.index(t) if t in M.TIERS else -1 for t in prob.tiers])
SIDE = np.column_stack([HAS_MK, prob.amounts == 0, TIER_I == 0, TIER_I == 1, TIER_I == 2,
                        np.log1p([len(s["body"]) for s in S]) / 6, np.log1p([len(s["subject"]) for s in S]) / 4])


def kfold(y, seed, k=5):
    return StratifiedKFold(k, shuffle=True, random_state=seed + 1000).split(np.zeros(len(y)), y)


def remap_top(P, c):
    """Set the top-class probability to c, spreading the rest in proportion; keep argmax stable."""
    top = P.argmax(1); idx = np.arange(len(P))
    oth = P.copy(); oth[idx, top] = 0
    q = oth / oth.sum(1, keepdims=True)
    qmax = q.max(1)
    c = np.clip(c, qmax / (1 + qmax) + 1e-6, 1 - M.P_EPS)
    out = q * (1 - c)[:, None]
    out[idx, top] = c
    return np.clip(out, M.P_EPS, 1) / np.clip(out, M.P_EPS, 1).sum(1, keepdims=True)


# ------------------------------------------------------------------ route variants
def route_features(Z, side):
    Pr = M.softmax_T(Z[:, :5], 1.0)
    srt = np.sort(Pr, 1)[:, ::-1]
    Zs = np.sort(Z[:, :5], 1)[:, ::-1]
    top = Pr.argmax(1)
    f = [np.log(srt[:, 0] / (1 - srt[:, 0] + 1e-9) + 1e-9), srt[:, 0] - srt[:, 1], entropy(Pr),
         Zs[:, 0] - Zs[:, 1], Zs[:, 0] - Zs[:, 2], np.eye(5)[top]]
    X = np.column_stack(f)
    return np.column_stack([X, SIDE]) if side else X


def route_conf_recal(Z, seed, side, base="temperature"):
    """Estimate P(top-1 correct) from the model's own logits (+ optional input flags); remap the top-class mass."""
    Pb = T.crossfit_route(Z[:, :5], yr, base, seed)
    corr = (Z[:, :5].argmax(1) == yr).astype(float)
    X = route_features(Z, side)
    s = np.zeros(n)
    for tr, te in kfold(yr, seed):
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(sc.transform(X[tr]), corr[tr])
        s[te] = lr.decision_function(sc.transform(X[te]))
    c = T.crossfit_binary(s, corr, "isotonic", seed)            # calibrate the score -> P(correct)
    return remap_top(Pb, c)


def route_matrix_scaling(Z, seed, C=1.0):
    P = np.zeros((n, 5))
    for tr, te in kfold(yr, seed):
        lr = LogisticRegression(C=C, max_iter=2000).fit(Z[tr, :5], yr[tr])
        P[te] = lr.predict_proba(Z[te, :5])
    P = np.clip(P, M.P_EPS, 1)
    return P / P.sum(1, keepdims=True)


def route_blend(Z, seed):
    a = T.crossfit_route(Z[:, :5], yr, "ovr_isotonic", seed)
    b = T.crossfit_route(Z[:, :5], yr, "temperature", seed)
    return 0.5 * (a + b)


def route_tiebreak(Z, seed):
    a = T.crossfit_route(Z[:, :5], yr, "ovr_isotonic", seed)
    b = M.softmax_T(Z[:, :5], 1.0)
    out = a + 1e-4 * b
    return out / out.sum(1, keepdims=True)


# ------------------------------------------------------------------ approve variants
def approve_side_layer(Z, seed, extra_heads=False, C=0.3):
    """Calibration layer: logistic regression on [approve logit, per-tier amount-vs-ceiling hats (+ other heads)],
    cross-fitted, then isotonic-calibrated like B. Tier ceilings are re-derived inside each training fold."""
    z = Z[:, 5]
    s = np.zeros(n)
    for tr, te in kfold(ya.astype(int), seed):
        caps = T.compute_tier_caps(prob, tr)
        blk = M.amount_block_v2(prob.amounts, prob.tiers, caps)
        X = np.column_stack([z, blk] + ([Z[:, :5], Z[:, 6]] if extra_heads else []))
        sc = StandardScaler().fit(X[tr])
        lr = LogisticRegression(C=C, max_iter=3000).fit(sc.transform(X[tr]), ya[tr])
        s[te] = lr.decision_function(sc.transform(X[te]))
    return T.choose_binary_cal(s, ya.astype(float), seed)[1]


def approve_gbm_reference(Z, seed):
    """NOT eligible (a separate boosted learner); shown only to bound what amount/tier information is worth."""
    s = np.zeros(n)
    for tr, te in kfold(ya.astype(int), seed):
        caps = T.compute_tier_caps(prob, tr)
        amt = prob.amounts; cap = np.array([caps[t] if t in caps else 100.0 for t in prob.tiers])
        X = np.column_stack([Z[:, 5], amt, TIER_I, amt == 0, amt / cap])
        g = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0)
        g.fit(X[tr], ya[tr]); s[te] = np.log(np.clip(g.predict_proba(X[te])[:, 1], 1e-4, 1 - 1e-4) / np.clip(g.predict_proba(X[te])[:, 0], 1e-4, 1))
    return T.choose_binary_cal(s, ya.astype(float), seed)[1]


def approve_tiebreak(Z, seed):
    pa = T.choose_binary_cal(Z[:, 5], ya.astype(float), seed)[1]
    return np.clip(pa + 1e-4 * M.sigmoid(Z[:, 5]), M.P_EPS, 1 - M.P_EPS)


ROUTE = {
    "R0 B (ovr isotonic)": lambda Z, s, P0: P0,
    "R1 temperature (pre-B)": lambda Z, s, P0: T.crossfit_route(Z[:, :5], yr, "temperature", s),
    "R2 matrix scaling (multinomial LR on logits)": lambda Z, s, P0: route_matrix_scaling(Z, s),
    "R3 confidence recal. from own logits": lambda Z, s, P0: route_conf_recal(Z, s, side=False),
    "R4 R3 + input flags (marker, amount==0, tier, lengths)": lambda Z, s, P0: route_conf_recal(Z, s, side=True),
    "R5 blend(iso, temperature)": lambda Z, s, P0: route_blend(Z, s),
    "R6 B + tie-break": lambda Z, s, P0: route_tiebreak(Z, s),
}
APPROVE = {
    "A0 B (isotonic on logit)": lambda Z, s, pa0: pa0,
    "A1 calib. layer: logit + amount/tier-ceiling hats": lambda Z, s, pa0: approve_side_layer(Z, s, False),
    "A2 A1 + other trunk heads (route, grounded)": lambda Z, s, pa0: approve_side_layer(Z, s, True),
    "A3 B + tie-break": lambda Z, s, pa0: approve_tiebreak(Z, s),
    "A4 [reference only, not eligible] GBM on logit+amount": lambda Z, s, pa0: approve_gbm_reference(Z, s),
}


def run(seeds, route_names=None, approve_names=None):
    out = {}
    for seed in seeds:
        Z, _ = load_oof(seed)
        P0, pa0, pg0 = baseline_B(Z, seed)
        base = score(P0, pa0, pg0, seed)
        rows = {"B baseline": base}
        for name, fn in ROUTE.items():
            if name.startswith("R0") or (route_names and name not in route_names):
                continue
            rows[name] = score(fn(Z, seed, P0), pa0, pg0, seed)
        for name, fn in APPROVE.items():
            if name.startswith("A0") or (approve_names and name not in approve_names):
                continue
            rows[name] = score(P0, fn(Z, seed, pa0), pg0, seed)
        out[seed] = rows
    return out


def report(out, seeds, title):
    names = list(next(iter(out.values())).keys())
    print("=" * 150); print(title); print("=" * 150)
    print("%-58s | %-24s | mean  | min   | dQual  dCalib  dAbst | abst parts route/approve (B: %s) | ECE route/approve | Brier-skill route/approve"
          % ("variant", "paired delta total by seed", "/".join("%.3f" % out[seeds[0]]["B baseline"]["abst_parts"][i] for i in (0, 1))))
    res = {}
    for nm in names[1:]:
        d = [out[s][nm]["total"] - out[s]["B baseline"]["total"] for s in seeds]
        dq = np.mean([out[s][nm]["quality"] - out[s]["B baseline"]["quality"] for s in seeds])
        dc = np.mean([out[s][nm]["calibration"] - out[s]["B baseline"]["calibration"] for s in seeds])
        da = np.mean([out[s][nm]["abstention"] - out[s]["B baseline"]["abstention"] for s in seeds])
        ap = [np.mean([out[s][nm]["abst_parts"][i] for s in seeds]) for i in (0, 1)]
        ec = [np.mean([out[s][nm]["ece"][i] for s in seeds]) for i in (0, 1)]
        bs = [np.mean([out[s][nm]["brier_skill"][i] for s in seeds]) for i in (0, 1)]
        res[nm] = {"deltas": d, "mean": float(np.mean(d)), "min": float(min(d))}
        print("%-58s | %-24s | %+.2f | %+.2f | %+.4f %+.4f %+.4f | %.3f / %.3f | %.4f / %.4f | %.4f / %.4f"
              % (nm[:58], " ".join("%+.2f" % x for x in d), np.mean(d), min(d), dq, dc, da, *ap, *ec, *bs))
    return res


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "screen":
        out = run(SCREEN_SEEDS)
        res = report(out, SCREEN_SEEDS, "SCREEN on seeds 13/14/15 (paired vs B; same folds). Strict bar: delta > +0.30 on every seed.")
        json.dump({"results": {str(s): v for s, v in out.items()}, "paired": res}, open("abstention/screen_results.json", "w"), indent=1, default=float)
    else:
        names = sys.argv[2:]
        out = run(CONFIRM_SEEDS, [n_ for n_ in names if n_ in ROUTE], [n_ for n_ in names if n_ in APPROVE])
        res = report(out, CONFIRM_SEEDS, "CONFIRMATION on FRESH seeds 16/17/18: " + " + ".join(names))
        json.dump({"results": {str(s): v for s, v in out.items()}, "paired": res}, open("abstention/confirm_results.json", "w"), indent=1, default=float)
