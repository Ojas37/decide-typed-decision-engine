"""Route: do multi-issue bodies carry a learnable 'primary issue' rule? (train only)"""
import re
from collections import Counter, defaultdict
from common import *   # noqa: F401,F403

Z, yr, ya, yg, fold = load_cache(13)
rows = load_train()
S = [r["state"] for r in rows]
P = T.crossfit_temperature(Z[:, :5], yr, 13)
pred = P.argmax(1); corr = pred == yr
subj = [s["subject"].strip().lower() for s in S]
vague = {"two things", "help with 2 things", "few things to sort out", "a couple of issues", "several issues", "multiple problems"}
idx = [i for i, s in enumerate(subj) if s in vague]
print("multi-issue-subject rows: %d  acc %.3f" % (len(idx), corr[idx].mean()))
random_i = idx[:0]
import random
random.seed(5)
for i in random.sample(idx, 8):
    print("\n[%s] true=%s pred=%s conf=%.2f\n  %s" % (S[i]["subject"], Q[yr[i]], Q[pred[i]], P[i].max(), S[i]["body"]))

# connective frames used in multi-issue bodies
frames = Counter()
for i in idx:
    b = S[i]["body"]
    for m in re.finditer(r"(Priority for me is:|On a related note,|Separately, and this is less urgent,|My real question is this:|The part I need you to action is:|For background,|Also,|Second,|First,|Two things\.|Two issues:)", b):
        frames[m.group(0)] += 1
print("\nframe phrases:", frames.most_common(12))

# Build a sentence->route lexicon from single-issue rows (non-vague subjects) to find each issue's category
mon = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
def sents(b):
    return [mon.sub("$#", s.strip()) for s in re.split(r"(?<=[.!?])\s+", b) if s.strip()]
lex = defaultdict(Counter)
for i, s in enumerate(subj):
    if s in vague: continue
    for sent in sents(S[i]["body"])[:3]:
        lex[sent.lower()][yr[i]] += 1
def cat(sent):
    c = lex.get(sent.lower())
    if not c: return None
    k, v = c.most_common(1)[0]
    return k if v / sum(c.values()) > 0.8 and sum(c.values()) >= 3 else None

# split each multi-issue body at its frame; find categories of the two issue fragments
def fragments(b):
    b2 = re.sub(r"^(Hi there,|Hello,|Hi team,|Good morning,|hey|hi,)\s*", "", b, flags=re.I)
    return sents(b2)
stats = Counter(); n_ok = 0
pos_stats = Counter()
for i in idx:
    ss = fragments(S[i]["body"])
    cats = [(j, cat(s)) for j, s in enumerate(ss)]
    cats = [(j, c) for j, c in cats if c is not None]
    if len(cats) >= 2:
        n_ok += 1
        distinct = [c for _, c in cats]
        first, last = distinct[0], distinct[-1]
        stats["label==first_issue"] += yr[i] == first
        stats["label==last_issue"] += yr[i] == last
        stats["label in neither"] += yr[i] not in (first, last)
        stats["first==last"] += first == last
print("\nmulti-issue rows where >=2 fragments were categorised by the single-issue lexicon: %d" % n_ok)
print(dict(stats))
# how often is the label one of the categories present at all?
inany = 0; tot = 0
for i in idx:
    cs = {c for _, c in [(j, cat(s)) for j, s in enumerate(fragments(S[i]["body"]))] if c is not None}
    if len(cs) >= 1:
        tot += 1; inany += yr[i] in cs
print("label among categories detected in the body: %d / %d" % (inany, tot))

# frame-conditional: which issue-position follows each frame, label match
print("\nfragment-position vs label (per frame type):")
pf = defaultdict(Counter)
for i in idx:
    b = S[i]["body"]
    for fr in ("Priority for me is:", "On a related note,", "Separately, and this is less urgent,", "My real question is this:", "The part I need you to action is:", "For background,"):
        if fr in b:
            after = b.split(fr, 1)[1]
            first_sent = sents(after)[0] if sents(after) else ""
            c = cat(first_sent)
            before = b.split(fr, 1)[0]
            bs = sents(before)
            cb = cat(bs[-1]) if bs else None
            pf[fr]["n"] += 1
            if c is not None: pf[fr]["issue_after_frame_known"] += 1; pf[fr]["label==issue_after"] += int(yr[i] == c)
            if cb is not None: pf[fr]["issue_before_frame_known"] += 1; pf[fr]["label==issue_before"] += int(yr[i] == cb)
for fr, c in pf.items():
    print("  %-40s %s" % (fr, dict(c)))
