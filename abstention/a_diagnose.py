"""Step 1: why do wrong predictions get high confidence, and how much ranking headroom exists? (train OOF only)"""
from collections import Counter, defaultdict
from common_abs import *   # noqa: F401,F403
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold

SEED = 13
Z, fold = load_oof(SEED)
prob = problem()
S = prob.states
yr, ya = prob.y_route, prob.y_app
n = prob.n
P, pa, pg = baseline_B(Z, SEED)
top = P.argmax(1); conf = P.max(1); corr = top == yr
subj = [s["subject"].strip().lower() for s in S]
has_mk = np.array([bool(MARKER_RE.search(s["body"])) for s in S])

print("=" * 100); print("ROUTE (candidate B probabilities, seed %d)" % SEED); print("=" * 100)
pr, risk, err = part(conf, corr)
print("full error %.4f | risk@80 %.4f | abstention part %.3f | AUROC(conf->correct) %.3f | distinct conf values %d"
      % (err, risk, pr, roc_auc_score(corr, conf), len(np.unique(np.round(conf, 9)))))

# --- error mass by confidence band
bands = [(0, .5), (.5, .7), (.7, .8), (.8, .9), (.9, .95), (.95, .98), (.98, 1.01)]
print("\nconfidence band     n    claimed  actual-acc  errors  share-of-errors  cum-share(from top conf)")
tot_e = (~corr).sum(); cum = 0
rows_b = []
for lo, hi in reversed(bands):
    m = (conf >= lo) & (conf < hi)
    if m.sum():
        cum += (~corr[m]).sum()
        print("  [%.2f,%.2f)  %5d   %.3f    %.3f      %4d    %5.1f%%           %5.1f%%"
              % (lo, hi, m.sum(), conf[m].mean(), corr[m].mean(), (~corr[m]).sum(), 100 * (~corr[m]).sum() / tot_e, 100 * cum / tot_e))
k80 = int(round(.8 * n)); keep = np.argsort(-conf, kind="stable")[:k80]
print("rows kept at 80%% coverage: %d, errors kept %d (%.1f%% of all errors), errors dropped %d" % (k80, (~corr[keep]).sum(), 100 * (~corr[keep]).sum() / tot_e, tot_e - (~corr[keep]).sum()))

# --- classify each error
cnt = defaultdict(Counter)
for s, y in zip(subj, yr): cnt[s][y] += 1
dom_share, dom_cls, tot = np.zeros(n), np.zeros(n, int), np.zeros(n)
for i, s in enumerate(subj):
    c = cnt[s].copy(); c[yr[i]] -= 1             # leave-one-out
    t = sum(c.values())
    if t > 0:
        k, v = c.most_common(1)[0]; dom_cls[i] = k; dom_share[i] = v / t; tot[i] = t
clear_subject = (tot >= 20) & (dom_share >= 0.85)
vague = (tot >= 8) & (dom_share < 0.6)
noise_like = clear_subject & ~has_mk & (yr != dom_cls) & (top == dom_cls)
cat = np.full(n, "other", dtype=object)
cat[vague & ~has_mk] = "vague subject (body decides)"
cat[has_mk] = "marker (multi-issue) body"
cat[clear_subject & ~has_mk] = "clear subject, label agrees"
cat[noise_like] = "clear subject, LABEL contradicts (noise-like)"
print("\nerror anatomy (route):")
print("  %-48s %5s %6s %7s %9s %10s %13s" % ("category", "rows", "acc", "errors", "% errors", "mean conf", "errors@conf>=.9"))
for c in ["clear subject, label agrees", "clear subject, LABEL contradicts (noise-like)", "marker (multi-issue) body", "vague subject (body decides)", "other"]:
    m = cat == c
    if c == "clear subject, label agrees":       # rows where label agrees with the subject majority
        pass
    if m.sum():
        e = ~corr[m]
        print("  %-48s %5d %6.3f %7d %8.1f%% %10.3f %13d" % (c, m.sum(), corr[m].mean(), e.sum(), 100 * e.sum() / tot_e, conf[m].mean(), (e & (conf[m] >= .9)).sum()))
nl = noise_like
print("\n  noise-like rows: model confidence on them (it follows the subject, label flipped): mean %.3f, median %.3f  vs  other clear rows %.3f"
      % (conf[nl].mean(), np.median(conf[nl]), conf[clear_subject & ~nl & ~has_mk].mean()))
print("  => %d of %d errors (%.1f%%) are noise-like flips that the model gets 'wrong' with HIGH confidence by construction."
      % (nl.sum() - 0, tot_e, 100 * (nl & ~corr).sum() / tot_e))

# --- headroom probes: can ANY available signal rank errors better than conf?
def ent_enc(train_idx, test_idx):
    """fold-safe subject statistics (entropy, count, max share) from the training part only"""
    c2 = defaultdict(Counter)
    for i in train_idx: c2[subj[i]][yr[i]] += 1
    out = np.zeros((len(test_idx), 3))
    for j, i in enumerate(test_idx):
        c = c2.get(subj[i])
        if c:
            t = sum(c.values()); p_ = np.array(list(c.values())) / t
            out[j] = [-(p_ * np.log(p_)).sum(), np.log1p(t), p_.max()]
        else:
            out[j] = [0, 0, 1]
    return out

srt = np.sort(P, axis=1)[:, ::-1]
base_f = np.column_stack([conf, srt[:, 0] - srt[:, 1], entropy(P), top, P])
Zs = np.sort(Z[:, :5], axis=1)[:, ::-1]
logit_f = np.column_stack([Zs[:, 0] - Zs[:, 1], Zs[:, 0] - Zs[:, 2], Zs[:, 0]])
side = np.column_stack([has_mk, [s["amount"] == 0 for s in S], [["free", "pro", "enterprise"].index(s["customer_tier"]) for s in S],
                        [len(s["body"]) for s in S], [len(s["subject"]) for s in S]])
def probe(feats, use_subject, label):
    sc = np.zeros(n)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=7).split(feats, corr):
        Xtr, Xte = feats[tr], feats[te]
        if use_subject:
            Xtr = np.column_stack([Xtr, ent_enc(tr, tr)]); Xte = np.column_stack([Xte, ent_enc(tr, te)])
        g = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0)
        g.fit(Xtr, corr[tr]); sc[te] = g.predict_proba(Xte)[:, 1]
    p_, r_, _ = part(sc, corr)
    print("  %-62s AUROC %.3f  risk@80 %.4f  part %.3f  (%+.2f total pts if route only)" % (label, roc_auc_score(corr, sc), r_, p_, (p_ - pr) * 5))
    return sc
print("\nRANKING-HEADROOM PROBES (GBM predicting correctness, 5-fold CV; upper-bound style, NOT deployable models)")
print("  current confidence (B): AUROC %.3f  risk@80 %.4f  part %.3f" % (roc_auc_score(corr, conf), risk, pr))
probe(base_f, False, "calibrated probs only (top, margin, entropy, class, all probs)")
probe(np.column_stack([base_f, logit_f]), False, "+ raw-logit gaps")
probe(np.column_stack([base_f, logit_f, side]), False, "+ marker flag, amount==0, tier, text lengths")
probe(np.column_stack([base_f, logit_f, side]), True, "+ subject label-entropy / count (fold-safe target encoding)")

# --- ceiling from label-noise alone: oracle that ranks every NON-noise error last
nn_err = (~corr) & ~nl
order_cost = np.where(nn_err, 1.0, 0.0)
rank = np.argsort(-(1 - order_cost) + 1e-9 * np.random.RandomState(0).rand(n), kind="stable")
kept = rank[:k80]
print("\nCEILING if every non-noise-like error were ranked last (noise-like flips stay unrankable): risk@80 %.4f -> part %.3f"
      % ((~corr[kept]).mean(), 1 - (~corr[kept]).mean() / err))

# =====================================================================================
print("\n" + "=" * 100); print("APPROVE (candidate B probabilities, seed %d)" % SEED); print("=" * 100)
cona = np.maximum(pa, 1 - pa); corra = (pa > .5) == ya
pr_a, risk_a, err_a = part(cona, corra)
print("full error %.4f | risk@80 %.4f | abstention part %.3f | AUROC(conf->correct) %.3f | distinct conf values %d"
      % (err_a, risk_a, pr_a, roc_auc_score(corra, cona), len(np.unique(np.round(cona, 9)))))
reg = np.where(pa < 0.1, "hold side  (p<0.1)", np.where(pa >= 0.7, "approvable core (p>=0.7)", "soft zone (0.1<=p<0.7)"))
tot_ea = (~corra).sum()
print("\n  %-28s %6s %7s %8s %9s %10s %9s" % ("region", "rows", "claimed", "actual", "errors", "% errors", "AUROC-in"))
for r_ in ["hold side  (p<0.1)", "soft zone (0.1<=p<0.7)", "approvable core (p>=0.7)"]:
    m = reg == r_
    lab = (~corra[m]).astype(int)
    au = roc_auc_score(lab, -cona[m]) if 0 < lab.sum() < m.sum() else float("nan")
    print("  %-28s %6d %7.3f %8.3f %9d %9.1f%% %9.3f" % (r_, m.sum(), cona[m].mean(), corra[m].mean(), (~corra[m]).sum(), 100 * (~corra[m]).sum() / tot_ea, au))
keep_a = np.argsort(-cona, kind="stable")[:k80]
print("  kept at 80%%: errors %d of %d; kept composition: %s" % ((~corra[keep_a]).sum(), tot_ea, dict(Counter(reg[keep_a]))))
cum_a = [(lo, (~corra[(cona >= lo)]).sum()) for lo in (.99, .9, .8, .7)]
print("  errors by confidence floor:", {">= %.2f" % lo: int(c) for lo, c in cum_a})

amt = prob.amounts; tier = np.array([["free", "pro", "enterprise"].index(t) if t in M.TIERS else -1 for t in prob.tiers])
caps = T.compute_tier_caps(prob, np.arange(n))
rat = amt / np.array([caps[M.TIERS[t]] for t in tier])
base_a = np.column_stack([pa, np.log(pa / (1 - pa)), Z[:, 5]])
def probe_a(feats, label):
    sc = np.zeros(n)
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=7).split(feats, corra):
        g = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200, l2_regularization=1.0, random_state=0)
        g.fit(feats[tr], corra[tr]); sc[te] = g.predict_proba(feats[te])[:, 1]
    p_, r_, _ = part(sc, corra)
    print("  %-62s AUROC %.3f  risk@80 %.4f  part %.3f  (%+.2f total pts if approve only)" % (label, roc_auc_score(corra, sc), r_, p_, (p_ - pr_a) * 5))
print("\nRANKING-HEADROOM PROBES for approve (GBM predicting correctness of the p>0.5 call; upper-bound style)")
print("  current confidence (B): AUROC %.3f  risk@80 %.4f  part %.3f" % (roc_auc_score(corra, cona), risk_a, pr_a))
probe_a(base_a, "approve logit/prob only")
probe_a(np.column_stack([base_a, amt, tier, rat, amt == 0]), "+ amount, tier, amount/tier-ceiling")
probe_a(np.column_stack([base_a, amt, tier, rat, amt == 0, Z[:, :5], Z[:, 6]]), "+ route logits, grounded logit (other heads of the shared trunk)")
probe_a(np.column_stack([base_a, amt, tier, rat, amt == 0, Z[:, :5], Z[:, 6], has_mk, [len(s['body']) for s in S]]), "+ marker flag, body length")
core = reg == "approvable core (p>=0.7)"
print("\n  inside the approvable core the hold label is noise: AUROC of ANY of the above for 'held back' was 0.48-0.54 in the earlier diagnostic")
