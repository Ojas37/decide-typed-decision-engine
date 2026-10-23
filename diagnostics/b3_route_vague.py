"""Route: vague-subject single-issue rows, marker inventory, and noise ceiling (train only)."""
import re
from collections import Counter, defaultdict
from common import *   # noqa: F401,F403

Z, yr, ya, yg, fold = load_cache(13)
rows = load_train()
S = [r["state"] for r in rows]
P = T.crossfit_temperature(Z[:, :5], yr, 13)
pred = P.argmax(1); corr = pred == yr; conf = P.max(1)
subj = [s["subject"].strip().lower() for s in S]
N = len(rows)

# 1) marker inventory: text immediately before ':' or a leading comma clause, in bodies of the multi-issue subjects
multi_sub = {"two things", "help with 2 things", "few things to sort out", "a couple of issues", "several issues", "multiple problems"}
pre = Counter()
for i, s in enumerate(subj):
    if s in multi_sub:
        for m in re.finditer(r"([A-Z][^.:!?]{3,60}[:,])\s", S[i]["body"]):
            pre[m.group(1)] += 1
print("candidate marker phrases (multi-issue bodies):")
for k, v in pre.most_common(16): print("  %4d  %s" % (v, k))

markers = [k for k, v in pre.items() if v >= 15]
mk_re = re.compile("|".join(re.escape(m) for m in sorted(markers, key=len, reverse=True)), re.I)
has_marker = np.array([bool(mk_re.search(S[i]["body"])) for i in range(N)])
print("\nrows containing >=1 marker: %d (%.1f%%)  accuracy %.3f | rows without: %.3f"
      % (has_marker.sum(), 100 * has_marker.mean(), corr[has_marker].mean(), corr[~has_marker].mean()))
print("marker rows hold %.1f%% of all route errors; by subject type: multi-subject %.0f%% of marker rows"
      % (100 * (~corr[has_marker]).sum() / (~corr).sum(), 100 * np.mean([subj[i] in multi_sub for i in np.where(has_marker)[0]])))
print("marker rows under NON-multi subjects: %d  acc %.3f" % ((has_marker & ~np.array([s in multi_sub for s in subj])).sum(),
      corr[has_marker & ~np.array([s in multi_sub for s in subj])].mean() if (has_marker & ~np.array([s in multi_sub for s in subj])).any() else float('nan')))

# 2) vague single-issue subjects: look at bodies
vague1 = ["not working", "help needed", "issue with my account", "question", "please help", "urgent", "problem"]
vi = [i for i, s in enumerate(subj) if s in vague1 and not has_marker[i]]
print("\nvague single-issue subjects, no marker: %d rows, acc %.3f, errors %d (%.1f%% of all errors)"
      % (len(vi), corr[vi].mean(), (~corr[vi]).sum(), 100 * (~corr[vi]).sum() / (~corr).sum()))
import random; random.seed(2)
wrong = [i for i in vi if not corr[i]]
for i in random.sample(wrong, 6):
    print("  [%s] true=%s pred=%s conf=%.2f | %s" % (S[i]["subject"], Q[yr[i]], Q[pred[i]], conf[i], S[i]["body"][:170]))
right = [i for i in vi if corr[i]]
for i in random.sample(right, 3):
    print("  OK [%s] true=%s pred=%s conf=%.2f | %s" % (S[i]["subject"], Q[yr[i]], Q[pred[i]], conf[i], S[i]["body"][:170]))

# 3) noise ceiling from unambiguous subjects: minority-label share where the subject has a dominant route (>=85%)
by = defaultdict(Counter)
for s, y in zip(subj, yr): by[s][y] += 1
dom_rows = 0; minority = 0; model_wrong_on_minority = 0; model_wrong_on_majority = 0
for i, s in enumerate(subj):
    c = by[s]; t = sum(c.values()); k, v = c.most_common(1)[0]
    if t >= 20 and v / t >= 0.85 and not has_marker[i]:
        dom_rows += 1
        if yr[i] != k:
            minority += 1; model_wrong_on_minority += int(not corr[i])
        else:
            model_wrong_on_majority += int(not corr[i])
print("\nUNAMBIGUOUS subjects (dominant route>=85%%, n>=20, no marker): %d rows" % dom_rows)
print("  label differs from the subject's dominant route in %.1f%% of them (apparent label noise / rule we cannot see)" % (100 * minority / dom_rows))
print("  model errors there: %d, of which %d (%.0f%%) are rows labelled against the dominant route" %
      (model_wrong_on_minority + model_wrong_on_majority, model_wrong_on_minority,
       100 * model_wrong_on_minority / max(1, model_wrong_on_minority + model_wrong_on_majority)))
# do those minority-labelled rows share anything? tier, amount, other content
minor_idx = [i for i, s in enumerate(subj) if by[s].most_common(1)[0][0] != yr[i] and sum(by[s].values()) >= 20 and by[s].most_common(1)[0][1] / sum(by[s].values()) >= .85 and not has_marker[i]]
print("  minority-labelled rows: tier mix %s | amount==0 %.2f (base %.2f)" % (
    dict(Counter(S[i]["customer_tier"] for i in minor_idx)), np.mean([S[i]["amount"] == 0 for i in minor_idx]), np.mean([s["amount"] == 0 for s in S])))
print("  does the BODY text of a minority-labelled row differ from its subject's usual body? sample:")
for i in random.sample(minor_idx, 5):
    print("   [%s] label=%s dominant=%s model=%s conf=%.2f | %s" % (S[i]["subject"], Q[yr[i]], Q[by[subj[i]].most_common(1)[0][0]], Q[pred[i]], conf[i], S[i]["body"][:130]))
rep = {"marker_rows": int(has_marker.sum()), "marker_row_acc": float(corr[has_marker].mean()),
       "unambiguous_label_deviation_rate": minority / dom_rows,
       "markers": markers}
save_json("route_structure.json", rep)
