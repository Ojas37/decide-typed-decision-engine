"""Section 2: APPROVE diagnostics on cross-fitted OOF probabilities (train only)."""
from collections import Counter, defaultdict
from common import *   # noqa: F401,F403

Z, yr, ya, yg, fold = load_cache(13)
rows = load_train()
S = [r["state"] for r in rows]
n = len(ya)
p = crossfit_binary_any(Z[:, 5], ya, "isotonic", seed=13)      # deployed calibrator, cross-fitted
dec = json.load(open("artifacts/meta.json"))["decision"]["approve"]
THR = dec["threshold"]
rep = {}

def counts(y, pred):
    tp = int((pred & y).sum()); fp = int((pred & ~y).sum()); fn = int((~pred & y).sum()); tn = int((~pred & ~y).sum())
    return tp, fp, fn, tn

print("base rate approve=%.3f  always-hold cost %.4f  always-approve cost %.4f" % (ya.mean(), ya.mean(), 3 * (1 - ya.mean())))

# ---- A. threshold sweep, full coverage (probabilities separate from decision) ------------
print("\nTHRESHOLD SWEEP, no abstention (cost = (3*FA + 1*FH)/N)")
print(" thr   FA   FH   FArate  FHrate  cost   FA-share-of-cost  approve%  macroF1")
sweep = []
for t in (0.30, 0.40, 0.50, 0.60, 0.65, 0.70, 0.73, 0.75, 0.78, 0.80, 0.82, 0.84, 0.86, 0.90):
    pred = p >= t
    tp, fp, fn, tn = counts(ya, pred)
    cost = (3 * fp + fn) / n
    f1 = SC.macro_f1(list(zip(ya.tolist(), pred.tolist())), [True, False])
    sweep.append({"thr": t, "FA": fp, "FH": fn, "cost": cost, "macro_f1": f1, "approve_rate": float(pred.mean())})
    print(" %.2f %5d %5d  %.4f  %.4f  %.4f   %.2f              %.3f    %.4f%s"
          % (t, fp, fn, fp / n, fn / n, cost, 3 * fp / max(1, 3 * fp + fn), pred.mean(), f1, "  <- deployed" if abs(t - THR) < .005 else ""))
rep["sweep_full_coverage"] = sweep

# ---- B. what the probabilities look like ------------------------------------------------
print("\nDISTRIBUTION of calibrated p: p<0.1: %.1f%%   0.1-0.7: %.1f%%   0.7-0.9: %.1f%%   >0.9: %.1f%%   max p = %.3f"
      % (100 * np.mean(p < .1), 100 * np.mean((p >= .1) & (p < .7)), 100 * np.mean((p >= .7) & (p < .9)), 100 * np.mean(p >= .9), p.max()))
elig = p >= 0.7
print("rows with p>=0.7 ('eligible'): %d (%.1f%%), observed approve rate %.3f -> %d held-back rows inside the eligible set (FA at any threshold <=0.7)"
      % (elig.sum(), 100 * elig.mean(), ya[elig].mean(), (~ya[elig]).sum()))
print("rows with p<0.1: %d, observed approve rate %.4f (hold side is very clean: only %d approvals missed)" % ((p < .1).sum(), ya[p < .1].mean(), ya[p < .1].sum()))

# ---- C. with the deployed abstention rule ---------------------------------------------------
rule = dec["abstain"]
ab = np.asarray(M.abstain_binary(p, rule))
pred = p >= THR
print("\nDEPLOYED POLICY thr=%.2f, abstain rule %s thr=%.3f" % (THR, rule["kind"], rule["thr"]))
tp, fp, fn, tn = counts(ya[~ab], pred[~ab])
m = (~ab).sum()
print("  answered %d (%.1f%%): FA %d  FH %d  cost %.4f (always-hold on answered %.4f)  | abstained rows: approve-rate %.3f, would-have-cost %.4f"
      % (m, 100 * m / n, fp, fn, (3 * fp + fn) / m, ya[~ab].mean(), ya[ab].mean(), approve_cost := ((3 * (pred[ab] & ~ya[ab]).sum() + (~pred[ab] & ya[ab]).sum()) / max(1, ab.sum()))))
print("  abstention catches %.1f%% of all false approvals and %.1f%% of all false holds"
      % (100 * (ab & pred & ~ya).sum() / max(1, (pred & ~ya).sum()), 100 * (ab & ~pred & ya).sum() / max(1, (~pred & ya).sum())))
rep["deployed"] = {"answered": int(m), "FA": fp, "FH": fn, "cost_answered": (3 * fp + fn) / m}

# ---- D. risk-coverage (selfcheck metric: error of p>0.5 ranked by max(p,1-p)) -----------------
conf = np.maximum(p, 1 - p); corr = ((p > .5) == ya).astype(float)
print("\nRISK-COVERAGE by confidence=max(p,1-p) (this is what the abstention axis scores)")
for c in selective_curve(conf, corr, (0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)):
    print("  cov %.2f  error %.4f  (oracle %.4f)" % (c["coverage"], c["risk"], c["oracle_risk"]))
# cost-weighted risk-coverage ranking by expected cost of best action
risk = M.approve_risk(p)
order = np.argsort(risk, kind="stable")
print("RISK-COVERAGE of committed decision (thr %.2f), cost per answered row, ranking by expected-cost (lower better) vs confidence:" % THR)
for c in (0.7, 0.8, 0.83, 0.9, 1.0):
    k = int(round(n * c))
    for nm, ordr in (("exp-cost", order), ("conf", np.argsort(-conf, kind="stable"))):
        kk = ordr[:k]; pr = p[kk] >= THR; y = ya[kk]
        print("  cov %.2f rank=%-8s cost %.4f  FA %4d FH %4d" % (c, nm, (3 * (pr & ~y).sum() + (~pr & y).sum()) / k, (pr & ~y).sum(), (~pr & y).sum()), end="")
    print()

# ---- E. where do the false approvals / false holds live? -------------------------------------
def amt_bin(a):
    for e in (0.001, 10, 25, 50, 100, 200, 500):
        if a < e: return "<%g" % e
    return ">=500"
fa = (p >= THR) & ~ya; fh = (p < THR) & ya
print("\nFALSE APPROVALS=%d  FALSE HOLDS=%d (full coverage, thr %.2f)" % (fa.sum(), fh.sum(), THR))
for name, key in (("tier", lambda s: s["customer_tier"]), ("amount", lambda s: amt_bin(s["amount"]))):
    g = defaultdict(lambda: [0, 0, 0, 0])
    for i, s in enumerate(S):
        k = key(s); g[k][0] += 1; g[k][1] += int(fa[i]); g[k][2] += int(fh[i]); g[k][3] += int(p[i] >= THR)
    print(" by %s: %s" % (name, {k: "n=%d FA=%d FH=%d appr=%d" % tuple(v) for k, v in sorted(g.items())}))
g = defaultdict(lambda: [0, 0, 0])
for i, s in enumerate(S):
    k = (s["customer_tier"], amt_bin(s["amount"]))
    g[k][0] += 1; g[k][1] += int(fa[i]); g[k][2] += int(fh[i])
print(" tier x amount cells with most error cost:")
for k, (nn, a, b) in sorted(g.items(), key=lambda kv: -(3 * kv[1][1] + kv[1][2]))[:8]:
    print("   %-22s n=%4d FA=%3d FH=%3d  cost-share %.1f%%" % (k, nn, a, b, 100 * (3 * a + b) / (3 * fa.sum() + fh.sum())))
print(" by TRUE route: FA", dict(Counter(Q[yr[i]] for i in np.where(fa)[0])), "| FH", dict(Counter(Q[yr[i]] for i in np.where(fh)[0])))
rep["fa_by_cell"] = {str(k): v for k, v in g.items()}

# ---- F. most confident mistakes ---------------------------------------------------------------------
print("\nMOST CONFIDENT FALSE APPROVALS (highest p, label=hold)")
for i in [j for j in np.argsort(-p) if not ya[j]][:8]:
    print("  p=%.3f %-10s $%-8.2f route=%-9s | %s | %s" % (p[i], S[i]["customer_tier"], S[i]["amount"], Q[yr[i]], S[i]["subject"][:28], S[i]["body"][:95]))
print("MOST CONFIDENT FALSE HOLDS (lowest p, label=approve)")
for i in [j for j in np.argsort(p) if ya[j]][:8]:
    print("  p=%.3f %-10s $%-8.2f route=%-9s | %s | %s" % (p[i], S[i]["customer_tier"], S[i]["amount"], Q[yr[i]], S[i]["subject"][:28], S[i]["body"][:95]))
save_json("approve_diagnostics.json", rep)
