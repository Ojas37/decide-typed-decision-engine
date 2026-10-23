"""Stress-test the exported route calibrator and its use through DecisionModel (no dev data)."""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import model as M  # noqa: E402

art = sys.argv[1]
meta = json.load(open(os.path.join(art, "meta.json")))
cal = meta["calibration"]["route"]
print("artifact %s  route calibrator: %s  feature_version %s" % (art, cal["type"], meta["feature_version"]))
rng = np.random.RandomState(0)
cases = {
    "typical logits N(0,3)": rng.normal(0, 3, (20000, 5)),
    "extreme logits N(0,40)": rng.normal(0, 40, (20000, 5)),
    "all equal": np.zeros((50, 5)),
    "one huge": np.eye(5) * 500,
    "tiny": rng.normal(0, 1e-9, (50, 5)),
}
bad = 0
for name, L in cases.items():
    P = M.route_probs(L, cal)
    ok = np.all(np.isfinite(P)) and np.all(P >= 0) and np.allclose(P.sum(1), 1.0, atol=1e-9) and P.shape == L.shape
    print("  %-26s n=%-6d finite/nonneg/sum=1: %s   min p %.2e  max |sum-1| %.1e" % (name, len(L), ok, P.min(), np.abs(P.sum(1) - 1).max()))
    bad += (not ok)
# single-vector path must equal the batch path
v = rng.normal(0, 3, 5)
assert np.allclose(M.route_probs(v, cal), M.route_probs(v[None, :], cal)[0]), "1-d and 2-d paths differ"

m = M.DecisionModel(art)
assert m.ready, m.load_error
rows = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")][:600]
mism = 0
for r in rows:
    value, probs, conf, ab = m.route(r["state"])
    if abs(probs[value] - conf) > 1e-12 or value != max(probs, key=probs.get) or abs(sum(probs.values()) - 1) > 1e-9 or min(probs.values()) < 0:
        mism += 1
print("  served route outputs on %d train rows: contract violations = %d" % (len(rows), mism))
print("RESULT:", "PASS" if (bad == 0 and mism == 0) else "FAIL")
sys.exit(0 if (bad == 0 and mism == 0) else 1)
