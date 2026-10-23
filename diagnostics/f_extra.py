"""Follow-up experiments (train only, OOF): paired seed deltas, tie-breaking, fair route oracle, approve stacking, grounded gap sizing."""
import re
from collections import Counter
from common import *   # noqa: F401,F403
from e_score_sim import pipeline, show, score_from, quantile_abstain, GRID_A
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

caches = {s: load_cache(s) for s in (13, 14, 15)}
rows = load_train()
S = [r["state"] for r in rows]

print("=== A. PAIRED across seeds: delta total vs deployed recipe (same OOF logits, so paired)")
variants = {"route=ovr_isotonic": dict(route_cal="ovr_isotonic"), "route=raw": dict(route_cal="raw"),
            "approve=platt": dict(app_cal="platt"), "cov=0.80": dict(cov=0.80), "cov=0.85": dict(cov=0.85)}
for name, kw in variants.items():
    d = []
    for s, (Z, yr, ya, yg, fold) in caches.items():
        d.append(pipeline(Z, yr, ya, yg, seed=s, **kw)["total"] - pipeline(Z, yr, ya, yg, seed=s)["total"])
    print("  %-22s deltas by seed %s  mean %+.2f" % (name, np.round(d, 2), np.mean(d)))

print("\n=== B. TIE-BREAKING: isotonic keeps calibration but flattens ranking; add a tiny monotone raw term")
Z, yr, ya, yg, fold = caches[13]
for name, z, y in (("approve", Z[:, 5], ya), ("grounded", Z[:, 6], yg)):
    for kind in ("platt", "isotonic"):
        p = crossfit_binary_any(z, y, kind)
        p_tb = np.clip(p + 1e-4 * M.sigmoid(z), M.P_EPS, 1 - M.P_EPS)
        for tag, pp in (("", p), ("+tiebreak", p_tb)):
            m = binary_metrics(pp, y)
            print("  %-9s %-9s%-10s ECE %.4f Brier %.4f abstention-part %.3f  unique p %d" % (name, kind, tag, m["ece15"], m["brier"], m["abstention_part"], m["unique_p"]))
P_oi = crossfit_route_any(Z[:, :5], yr, "ovr_isotonic")
Pt = P_oi + 1e-4 * M.softmax_T(Z[:, :5], 1.0); Pt /= Pt.sum(1, keepdims=True)
for tag, PP in (("route ovr_isotonic", P_oi), ("route ovr_isotonic+tiebreak", Pt)):
    m = route_metrics(PP, yr)
    print("  %-30s ECE %.4f Brier %.4f abstention-part %.3f  acc %.4f" % (tag, m["ece15_top"], m["brier"], m["abstention_part"], m["acc"]))

print("\n=== C. FAIR route oracle: fix marker-row errors, then RE-CALIBRATE (a real fix would be re-fit)")
mk = re.compile("Priority for me is:|My real question is this:|The part I need you to action is:|The main thing I need resolved is:|What I actually need help with is this:|Most importantly:|Less important, but while I am writing:|Some context first:|For background,|Separately, and this is less urgent,|On a related note,|Also worth mentioning,")
has_mk = np.array([bool(mk.search(s["body"])) for s in S])
Zr = Z[:, :5].copy()
P0 = M.softmax_T(Zr, T.fit_temperature(Zr, yr))
wrong = (P0.argmax(1) != yr) & has_mk
for frac in (0.5, 1.0):
    Zr2 = Zr.copy()
    idx = np.where(wrong)[0]
    idx = idx[: int(len(idx) * frac)]
    for i in idx:                              # swap the wrong top logit with the true class logit
        a = Zr2[i].argmax(); Zr2[i, [a, yr[i]]] = Zr2[i, [yr[i], a]]
    Z2 = Z.copy(); Z2[:, :5] = Zr2
    base = pipeline(Z, yr, ya, yg)
    show("route oracle: fix %d%% of marker errors, recalibrated" % (100 * frac), pipeline(Z2, yr, ya, yg), base)
    show("   same + OvR-isotonic route calibration", pipeline(Z2, yr, ya, yg, route_cal="ovr_isotonic"), base)

print("\n=== D. APPROVE stacking diagnostic: does finer amount/tier resolution beat the current approve logit?  (OOF-on-OOF, 5-fold)")
amt = np.array([s["amount"] for s in S]); tier = np.array([["free", "pro", "enterprise"].index(s["customer_tier"]) for s in S])
caps = np.array([32.0, 122.0, 500.0])[tier]
X0 = Z[:, [5]]
X1 = np.column_stack([Z[:, 5], amt, tier, amt == 0, amt / caps, np.minimum(amt / caps, 1.5)])
def cv_p(X, seed=3):
    p = np.zeros(len(ya))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, ya):
        g = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0)
        g.fit(X[tr], ya[tr]); p[te] = g.predict_proba(X[te])[:, 1]
    return p
p_cur = crossfit_binary_any(Z[:, 5], ya, "isotonic")
p_s0 = cv_p(X0); p_s1 = cv_p(X1)
for tag, pp in (("current (isotonic on logit)", p_cur), ("GBM on logit only", p_s0), ("GBM on logit + fine amount/tier", p_s1)):
    m = binary_metrics(pp, ya)
    full = min(T.approve_cost(ya, pp >= t) for t in GRID_A)
    print("  %-34s Brier %.4f  AUROC %.4f  ECE %.4f  best full-coverage cost %.4f  abstention-part %.3f" % (tag, m["brier"], m["auroc_label"], m["ece15"], full, m["abstention_part"]))
soft = (p_cur > 0.1) & (p_cur < 0.7)
print("  rows in the 'soft zone' (0.1<p<0.7): %d (%.1f%%); Brier there: current %.4f vs stacked %.4f" % (
    soft.sum(), 100 * soft.mean(), np.mean((p_cur[soft] - ya[soft]) ** 2), np.mean((p_s1[soft] - ya[soft]) ** 2)))

print("\n=== E. GROUNDED: how many errors sit on a known normaliser gap?")
pg = crossfit_binary_any(Z[:, 6], yg, "platt")
err = (pg >= 0.5) != yg
gap_re = {"'quarter of an hour'": re.compile(r"quarter of an hour", re.I),
          "'N hundred and M'": re.compile(r"hundred and", re.I),
          "'thousand'": re.compile(r"thousand", re.I),
          "hyphenated number (e.g. sixty-five)": re.compile(r"(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)-\w+", re.I)}
anyg = np.zeros(len(S), dtype=bool)
for name, rx in gap_re.items():
    m = np.array([bool(rx.search(s["draft_reply"] + " " + s["kb_article"])) for s in S])
    anyg |= m
    print("  %-38s rows %4d (%.1f%%)  model error rate %.3f  errors %3d (%.1f%% of all errors)" % (name, m.sum(), 100 * m.mean(), err[m].mean(), err[m].sum(), 100 * err[m].sum() / err.sum()))
print("  ANY of the above: rows %d, error rate %.3f vs %.4f elsewhere; they hold %.1f%% of all grounded errors" % (anyg.sum(), err[anyg].mean(), err[~anyg].mean(), 100 * err[anyg].sum() / err.sum()))
# spurious numeral mismatch: draft numeral absent from KB numerals, yet its time-span exists in KB's spans
spur = []
for s in S:
    dt = M._merge_numbers(M._tokens(s["draft_reply"], False)); kt = M._merge_numbers(M._tokens(s["kb_article"], False))
    dn, _, dq = M._quantities(dt); kn, _, kq = M._quantities(kt)
    spur.append(bool(dn - kn) and bool(dq) and dq <= kq)
spur = np.array(spur)
print("  spurious numeral-mismatch (draft numeral not in KB, but ALL draft time-spans match KB spans): rows %d, grounded rate %.3f, model error %.3f, errors %d"
      % (spur.sum(), yg[spur].mean(), err[spur].mean(), err[spur].sum()))
