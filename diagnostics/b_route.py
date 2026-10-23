"""Section 1: ROUTE diagnostics on cross-fitted OOF predictions (train only)."""
from collections import Counter, defaultdict
from common import *   # noqa: F401,F403

Z, yr, ya, yg, fold = load_cache(13)
rows = load_train()
S = [r["state"] for r in rows]
P = T.crossfit_temperature(Z[:, :5], yr, 13)          # same calibrated route probs the deployed recipe yields
pred = P.argmax(1)
conf = P.max(1)
corr = (pred == yr)
n = len(yr)
rep = {}

print("overall acc %.4f  macro-F1 %.4f  errors %d / %d" % (corr.mean(), route_metrics(P, yr)["macro_f1"], (~corr).sum(), n))

cm = np.zeros((5, 5), dtype=int)
for t, p in zip(yr, pred):
    cm[t, p] += 1
print("\nCONFUSION (rows=true, cols=pred)      " + " ".join("%9s" % q[:8] for q in Q))
for i, q in enumerate(Q):
    print("  %-10s %s" % (q, " ".join("%9d" % v for v in cm[i])))
print("\n%-10s %7s %7s %7s %7s" % ("class", "prec", "recall", "F1", "support"))
per = {}
for i, q in enumerate(Q):
    tp = cm[i, i]; fp = cm[:, i].sum() - tp; fn = cm[i].sum() - tp
    pr = tp / (tp + fp); rc = tp / (tp + fn); f1 = 2 * pr * rc / (pr + rc)
    per[q] = {"precision": pr, "recall": rc, "f1": f1, "support": int(cm[i].sum())}
    print("%-10s %7.3f %7.3f %7.3f %7d" % (q, pr, rc, f1, cm[i].sum()))
rep["confusion"] = cm.tolist(); rep["per_class"] = per

pairs = Counter((Q[t], Q[p]) for t, p in zip(yr, pred) if t != p)
print("\nLARGEST ERROR FLOWS (true -> predicted):")
for (t, p), c in pairs.most_common(8):
    print("  %-9s -> %-9s %4d  (%.1f%% of errors)" % (t, p, c, 100 * c / (~corr).sum()))
rep["top_error_pairs"] = [[t, p, c] for (t, p), c in pairs.most_common(12)]

# ---- where do errors come from? by subject template ----------------------------
subj = [s["subject"].strip().lower() for s in S]
by = defaultdict(lambda: [0, 0, Counter()])
for i, s in enumerate(subj):
    by[s][0] += 1
    by[s][1] += int(not corr[i])
    by[s][2][Q[yr[i]]] += 1
tot_err = int((~corr).sum())
print("\nTOP SUBJECTS BY ERROR COUNT (n, errors, share of all errors, label mix)")
top = sorted(by.items(), key=lambda kv: -kv[1][1])[:14]
for s, (nn, ee, mix) in top:
    print("  %-42s n=%4d err=%3d (%.1f%% of errors, err-rate %.2f)  %s"
          % (s[:42], nn, ee, 100 * ee / tot_err, ee / nn, dict(mix.most_common(3))))
cum = 0
sorted_all = sorted(by.items(), key=lambda kv: -kv[1][1])
for k, (s, (nn, ee, mix)) in enumerate(sorted_all[:40]):
    cum += ee
print("  top-40 subjects hold %.1f%% of all route errors (of %d distinct subjects)" % (100 * cum / tot_err, len(by)))

# ---- label noise vs model failure: leave-one-out subject majority ---------------
cnt = {s: v[2] for s, v in by.items()}
loo_maj = []
for i, s in enumerate(subj):
    c = cnt[s].copy(); c[Q[yr[i]]] -= 1
    loo_maj.append(c.most_common(1)[0][0] if sum(c.values()) > 0 else None)
loo_pred = np.array([Q.index(m) if m else -1 for m in loo_maj])
print("\nSUBJECT-ONLY predictor (leave-one-out subject majority): acc %.4f   model acc %.4f"
      % (np.mean(loo_pred == yr), corr.mean()))
err_idx = np.where(~corr)[0]
agree = np.mean(pred[err_idx] == loo_pred[err_idx])
print("model errors where model sided with the subject-majority route: %.1f%%  "
      "(true label contradicts what the subject usually means)" % (100 * agree))
lab_vs_maj = np.mean(yr[err_idx] == loo_pred[err_idx])
print("model errors where the TRUE label equals subject-majority (model deviated from subject): %.1f%%" % (100 * lab_vs_maj))
noisy = np.array([ (yr[i] != loo_pred[i]) for i in range(n)])
print("rows whose label differs from subject-majority: %.1f%% of all rows -> model gets %.1f%% of those right, %.1f%% of the rest"
      % (100 * noisy.mean(), 100 * corr[noisy].mean(), 100 * corr[~noisy].mean()))
rep["subject_only_acc"] = float(np.mean(loo_pred == yr))
rep["errors_model_follows_subject_majority"] = float(agree)

# does the body ever rescue a wrong-looking subject? ambiguous (multi-issue) subjects
def entropy(c):
    t = sum(c.values()); return -sum(v / t * math.log(v / t) for v in c.values())
amb = {s for s, (nn, ee, mix) in by.items() if nn >= 8 and entropy(mix) > 0.5}
am = np.array([s in amb for s in subj])
print("\nAMBIGUOUS-subject rows (label entropy>0.5 nats): %d (%.1f%%): accuracy %.3f | other rows: %.3f ; they hold %.1f%% of errors"
      % (am.sum(), 100 * am.mean(), corr[am].mean(), corr[~am].mean(), 100 * (~corr[am]).sum() / tot_err))
print("  ambiguous subjects:", sorted(amb)[:12])
for s in sorted(amb)[:6]:
    nn, ee, mix = by[s]
    print("    %-38s n=%4d err=%.2f  %s" % (s[:38], nn, ee / nn, dict(mix.most_common(4))))

# tier / amount / body-length slices
for name, vals in (("tier", [s["customer_tier"] for s in S]),
                   ("amount==0", ["zero" if s["amount"] == 0 else "nonzero" for s in S])):
    g = defaultdict(list)
    for v, c in zip(vals, corr): g[v].append(c)
    print("acc by %s: %s" % (name, {k: round(float(np.mean(v)), 3) for k, v in g.items()}))

# ---- confidence on correct vs wrong ---------------------------------------------
print("\nCONFIDENCE  correct: mean %.3f median %.3f | wrong: mean %.3f median %.3f | AUROC(conf->correct) %.3f"
      % (conf[corr].mean(), np.median(conf[corr]), conf[~corr].mean(), np.median(conf[~corr]),
         roc_auc_score(corr, conf)))
qs = [0.1, 0.25, 0.5, 0.75, 0.9]
print("  quantiles of confidence  correct:", np.round(np.quantile(conf[corr], qs), 3), " wrong:", np.round(np.quantile(conf[~corr], qs), 3))
hi = conf >= 0.95
print("  wrong predictions made with conf>=0.95: %d (%.1f%% of errors); conf>=0.90: %d (%.1f%% of errors)"
      % ((~corr & hi).sum(), 100 * (~corr & hi).sum() / tot_err, (~corr & (conf >= .9)).sum(),
         100 * (~corr & (conf >= .9)).sum() / tot_err))

covs = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)
curve = selective_curve(conf, corr.astype(float), covs)
print("\nSELECTIVE RISK (answer the most-confident fraction):")
print("  coverage   risk    oracle-risk   macro-F1 on answered")
order = sorted(range(n), key=lambda i: -conf[i])
for c in curve:
    k = max(1, int(round(n * c["coverage"]))); kept = order[:k]
    f1 = SC.macro_f1([(Q[yr[i]], Q[pred[i]]) for i in kept], Q)
    c["macro_f1_answered"] = f1
    print("  %.2f      %.4f  %.4f        %.4f" % (c["coverage"], c["risk"], c["oracle_risk"], f1))
rep["selective_risk"] = curve

# the deployed abstention threshold
thr = M.QUEUES and json.load(open("artifacts/meta.json"))["decision"]["route"]["abstain"]["thr"]
ab = conf < thr
print("\ndeployed route abstain thr %.3f -> abstains %.1f%%; error among abstained %.3f, among answered %.3f; abstained rows hold %.1f%% of errors"
      % (thr, 100 * ab.mean(), (~corr[ab]).mean(), (~corr[~ab]).mean(), 100 * (~corr[ab]).sum() / tot_err))
rep["deployed_abstain"] = {"thr": thr, "abstain_rate": float(ab.mean()), "share_of_errors_caught": float((~corr[ab]).sum() / tot_err)}

# a few example errors
print("\nSAMPLE ERRORS (confident, wrong):")
cand = [i for i in np.argsort(-conf) if not corr[i]][:8]
for i in cand:
    print("  conf %.2f true=%-9s pred=%-9s | %s | %s" % (conf[i], Q[yr[i]], Q[pred[i]], S[i]["subject"][:40], S[i]["body"][:110].replace("\n", " ")))
save_json("route_diagnostics.json", rep)
