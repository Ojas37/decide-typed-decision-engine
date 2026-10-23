"""Approve: locate amount cutoffs and test whether the 'held-back eligible' rows are predictable (train only)."""
from collections import Counter, defaultdict
from common import *   # noqa: F401,F403
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

Z, yr, ya, yg, fold = load_cache(13)
rows = load_train()
S = [r["state"] for r in rows]
n = len(ya)
p = crossfit_binary_any(Z[:, 5], ya, "isotonic", seed=13)
amt = np.array([s["amount"] for s in S]); tier = np.array([s["customer_tier"] for s in S])
elig = p >= 0.7

print("== 1. AMOUNT CUTOFFS. approve rate among model-eligible-TYPE rows is confounded by type, so show raw rate and rate among rows with p_text high")
print("   Using rows the model scores >=0.7 OR whose only reason for low p is amount: approximated by logit rank within tier")
for t, lo, hi, step in (("free", 15, 50, 2.5), ("pro", 80, 150, 5), ("enterprise", 400, 600, 20)):
    print(" tier=%s" % t)
    line = []
    e = lo
    while e < hi:
        m = (tier == t) & (amt >= e) & (amt < e + step)
        if m.sum() >= 5:
            line.append("%g:%.2f(n%d)" % (e, ya[m].mean(), m.sum()))
        e += step
    print("   " + "  ".join(line))

# cutoff via best single-threshold split restricted to rows with p>=0.5 or label True? use label-based: for rows whose p-without-amount is high we lack it; instead
# estimate the cutoff as the amount where P(approve|tier, amount>x) drops, using approved rows only: max approved amount per tier
for t in ("free", "pro", "enterprise"):
    a = amt[(tier == t) & ya]
    print(" tier %-10s approved amounts: p50 %.1f  p90 %.1f  p99 %.1f  max %.2f   | held amounts below max-approved: %d of %d"
          % (t, np.median(a), np.quantile(a, .9), np.quantile(a, .99), a.max(), ((amt[(tier == t) & ~ya] <= a.max())).sum(), ((tier == t) & ~ya).sum()))
    # smallest amount above which no approvals except noise
    srt = np.sort(a)
    print("      approvals sorted tail:", np.round(srt[-8:], 2))

print("\n== 2. IS THE HELD-BACK 18%% INSIDE THE ELIGIBLE SET PREDICTABLE?  (eligible = oof p>=0.7, n=%d, hold rate %.3f)" % (elig.sum(), 1 - ya[elig].mean()))
yh = (~ya[elig]).astype(int)
Se = [S[i] for i in np.where(elig)[0]]
feats = []
for s in Se:
    toks, _ = M.Featurizer.extract(s)
    d = {t: 1 for t in toks if t.startswith(("s:", "b:", "bs:", "sf:"))}
    d["tier=" + s["customer_tier"]] = 1
    d["amt_bin=%d" % min(int(s["amount"] // 10), 60)] = 1
    d["amt0"] = int(s["amount"] == 0)
    feats.append(d)
dv = DictVectorizer()
X = dv.fit_transform(feats)
keep = np.asarray((X > 0).sum(0)).ravel() >= 8
X = X[:, keep]; names = np.array(dv.get_feature_names_out())[keep]
for C in (0.03, 0.1, 0.3):
    pr = cross_val_predict(LogisticRegression(C=C, max_iter=2000), X, yh, cv=5, method="predict_proba")[:, 1]
    print("  LR (text+tier+amount bins) C=%.2f  5-fold AUROC for 'held back' inside eligible set: %.3f" % (C, roc_auc_score(yh, pr)))
print("  AUROC of the deployed model's own p inside the eligible set: %.3f" % roc_auc_score(yh, -p[elig]))
lr = LogisticRegression(C=0.3, max_iter=3000).fit(X, yh)
co = lr.coef_[0]
print("  features most associated with HOLD inside eligible set:")
for j in np.argsort(-co)[:10]: print("     %+.2f  %s" % (co[j], names[j][:90]))
print("  features most associated with APPROVE inside eligible set:")
for j in np.argsort(co)[:10]: print("     %+.2f  %s" % (co[j], names[j][:90]))

print("\n== 3. relation to other labels inside the eligible set")
print("  hold rate by route label:", {Q[c]: round(float(1 - ya[elig & (yr == c)].mean()), 3) for c in range(5)})
print("  hold rate by grounded label:", {str(v): round(float(1 - ya[elig & (yg == v)].mean()), 3) for v in (True, False)})
print("  hold rate by tier:", {t: round(float(1 - ya[elig & (tier == t)].mean()), 3) for t in ("free", "pro", "enterprise")})
bins = [(0, 0.001), (0.001, 10), (10, 25), (25, 50), (50, 100), (100, 200), (200, 500)]
print("  hold rate by amount bin:", {"%g-%g" % b: (round(float(1 - ya[elig & (amt >= b[0]) & (amt < b[1])].mean()), 3), int((elig & (amt >= b[0]) & (amt < b[1])).sum())) for b in bins})
print("  hold rate by fold id (is it stable?):", {int(f): round(float(1 - ya[elig & (fold == f)].mean()), 3) for f in range(5)})

print("\n== 4. same question on the HOLD side: approvals inside p<0.1 (n=%d): %d" % ((p < .1).sum(), ya[p < .1].sum()))
print("\n== 5. approve vs 'noise-like' flips: hold rate among rows far below every cutoff (amount 1-10): %.3f ; rows in same cells but amount>=500: approve rate %.3f"
      % (1 - ya[(amt >= 1) & (amt < 10)].mean(), ya[amt >= 500].mean()))
# is there any positive approval where the amount is far above its tier cutoff?  noise in the other direction
print("   approvals with amount >= 500: %d / %d ; free approvals with amount>=60: %d / %d ; pro approvals with amount>=150: %d / %d"
      % (ya[amt >= 500].sum(), (amt >= 500).sum(), ya[(tier == 'free') & (amt >= 60)].sum(), ((tier == 'free') & (amt >= 60)).sum(),
         ya[(tier == 'pro') & (amt >= 150)].sum(), ((tier == 'pro') & (amt >= 150)).sum()))
