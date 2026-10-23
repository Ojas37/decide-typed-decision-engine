"""Section 3: GROUNDED error analysis on cross-fitted OOF probabilities (train only)."""
from collections import Counter, defaultdict
from common import *   # noqa: F401,F403

Z, yr, ya, yg, fold = load_cache(13)
rows = load_train()
S = [r["state"] for r in rows]
n = len(yg)
p = crossfit_binary_any(Z[:, 6], yg, "platt", seed=13)
dec = json.load(open("artifacts/meta.json"))["decision"]["grounded"]
THR = dec["threshold"]
pred = p >= THR
rep = {}
fp = pred & ~yg; fn = ~pred & yg
print("base rate grounded=%.3f | at thr %.2f: errors %d (%.2f%%)  FP(pred grounded, actually not)=%d  FN(pred not grounded, actually grounded)=%d"
      % (yg.mean(), THR, (pred != yg).sum(), 100 * (pred != yg).mean(), fp.sum(), fn.sum()))
print("macro-F1 full coverage %.4f | majority-class trivial %.4f" % (
    SC.macro_f1(list(zip(yg.tolist(), pred.tolist())), [True, False]),
    max(SC.macro_f1([(bool(v), True) for v in yg], [True, False]), SC.macro_f1([(bool(v), False) for v in yg], [True, False]))))
ab = np.asarray(M.abstain_binary(p, dec["abstain"]))
err = pred != yg
print("deployed abstention (conf<%.3f): abstains %.1f%%; error among abstained %.3f vs answered %.4f; catches %.1f%% of all errors"
      % (dec["abstain"]["thr"], 100 * ab.mean(), err[ab].mean(), err[~ab].mean(), 100 * (err & ab).sum() / err.sum()))
conf = np.maximum(p, 1 - p)
print("risk-coverage (conf=max(p,1-p)):", [(c["coverage"], round(c["risk"], 4), round(c["oracle_risk"], 4)) for c in selective_curve(conf, (~err).astype(float), (0.7, 0.8, 0.9, 0.95, 1.0))])

# ---- draft-vs-KB mismatch structure -----------------------------------------------
tok = M._tokens
def view(s):
    dt = M._merge_numbers(tok(s["draft_reply"], False)); kt = M._merge_numbers(tok(s["kb_article"], False))
    dn, dnu, dq = M._quantities(dt); kn, knu, kq = M._quantities(kt)
    return {"d_only": sorted(set(dt) - set(kt)), "k_only": sorted(set(kt) - set(dt)),
            "dn_only": sorted(dn - kn), "kn_only": sorted(kn - dn), "dq_only": sorted(dq - kq), "kq_only": sorted(kq - dq),
            "dq": dq, "kq": kq, "dn": dn, "kn": kn}
V = [view(s) for s in S]
def rate(mask, name):
    m = np.array(mask)
    print("  %-52s n=%4d  grounded-rate %.3f  model-error %.4f  (FP %d, FN %d)" % (name, m.sum(), yg[m].mean(), err[m].mean(), (fp & m).sum(), (fn & m).sum()))
print("\nERROR RATE BY MISMATCH TYPE")
rate([bool(v["dn_only"]) for v in V], "draft has numeral absent from KB")
rate([not v["dn_only"] for v in V], "all draft numerals appear in KB")
rate([bool(v["dq_only"]) for v in V], "draft time-span absent from KB")
rate([bool(v["dq"]) and not v["dq_only"] for v in V], "draft time-spans all match KB")
rate([not v["dq"] and not v["dn"] for v in V], "draft states no number at all")
rate([bool(v["dq_only"]) or bool(v["dn_only"]) for v in V], "any number/time mismatch")
rate([not (v["dq_only"] or v["dn_only"]) for v in V], "NO number/time mismatch")

# words that explain UNgrounded when there's no number mismatch (scope / plan / role substitutions)
no_num = [i for i, v in enumerate(V) if not (v["dq_only"] or v["dn_only"])]
c_un, c_gr = Counter(), Counter()
for i in no_num:
    (c_gr if yg[i] else c_un).update(V[i]["d_only"])
print("\nAmong rows with NO numeric mismatch: %d rows, ungrounded %d (%.1f%%), model errors %d"
      % (len(no_num), (~yg[no_num]).sum(), 100 * (~yg[no_num]).mean(), err[no_num].sum()))
lift = []
for w, c in c_un.items():
    if c >= 15:
        tot = c + c_gr.get(w, 0); lift.append((c / tot, w, c, tot))
print("  draft-only words most diagnostic of UNgrounded here:", [(w, "%d/%d" % (c, t)) for _, w, c, t in sorted(lift, reverse=True)[:14]])

print("\nFALSE POSITIVES (model says grounded, label says NOT)  n=%d  (sorted by confidence)" % fp.sum())
for i in [j for j in np.argsort(-p) if fp[j]][:8]:
    v = V[i]
    print("  p=%.3f | draft-only %s | kb-only(sample) %s | dq-only %s kq-only %s\n         DRAFT: %s\n         KB   : %s"
          % (p[i], v["d_only"][:10], v["k_only"][:10], v["dq_only"], v["kq_only"], S[i]["draft_reply"][:170], S[i]["kb_article"][:190]))
print("\nFALSE NEGATIVES (model says NOT grounded, label says grounded)  n=%d" % fn.sum())
for i in [j for j in np.argsort(p) if fn[j]][:8]:
    v = V[i]
    print("  p=%.3f | draft-only %s | kb-only(sample) %s | dq-only %s kq-only %s\n         DRAFT: %s\n         KB   : %s"
          % (p[i], v["d_only"][:10], v["k_only"][:10], v["dq_only"], v["kq_only"], S[i]["draft_reply"][:170], S[i]["kb_article"][:190]))

# systematic categories among errors
print("\nCATEGORY COUNTS among all %d errors:" % err.sum())
cat = Counter()
for i in np.where(err)[0]:
    v = V[i]
    cat[("FP" if fp[i] else "FN", "num/time mismatch" if (v["dn_only"] or v["dq_only"]) else "no number mismatch")] += 1
print(" ", dict(cat))
# FN with num mismatch: what do the mismatching numerals look like (units not normalised?)
print("\nFN rows with a numeric mismatch - mismatching items:")
for i in [j for j in np.where(fn)[0] if V[j]["dn_only"] or V[j]["dq_only"]][:10]:
    print("   dn_only=%s dq_only=%s | kn=%s kq=%s | draft: %s" % (V[i]["dn_only"], V[i]["dq_only"], sorted(V[i]["kn"])[:6], sorted(V[i]["kq"])[:4], S[i]["draft_reply"][:110]))
print("\nVocabulary the normaliser does not understand (words next to numerals in FN/FP rows):")
nxt = Counter()
for i in np.where(err)[0]:
    for t, nx in zip(M._merge_numbers(tok(S[i]["draft_reply"], False)), M._merge_numbers(tok(S[i]["draft_reply"], False))[1:]):
        if t[0].isdigit(): nxt[nx] += 1
print(" ", nxt.most_common(15))
save_json("grounded_diagnostics.json", {"errors": int(err.sum()), "FP": int(fp.sum()), "FN": int(fn.sum()), "categories": {"|".join(k): v for k, v in cat.items()}})
