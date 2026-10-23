"""Baseline parity: the modified model.py must reproduce the frozen baseline model.py exactly on the baseline artifacts."""
import importlib.util
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


old = load("backups/baseline_v1/model.py", "model_baseline_frozen")
new = load("model.py", "model_current")
art = sys.argv[1] if len(sys.argv) > 1 else "artifacts"
mo, mn = old.DecisionModel(art), new.DecisionModel(art)
assert mo.ready and mn.ready, (mo.load_error, mn.load_error)
rows = [json.loads(l) for l in open("data/train.jsonl", encoding="utf-8")][:1000]
odd = [{}, {"subject": None, "body": None, "customer_tier": None, "amount": None, "draft_reply": None, "kb_article": None},
       {"subject": "x", "body": "y" * 3000, "customer_tier": "platinum", "amount": -5.0},
       {"subject": "refund", "body": "refund please", "customer_tier": "Pro", "amount": "12.5", "extra": 1}]
states = [r["state"] for r in rows] + odd
bad = 0
for s in states:
    for fn in ("route", "approve", "grounded"):
        a, b = getattr(mo, fn)(s), getattr(mn, fn)(s)
        if a != b:
            bad += 1
print("artifacts=%s  compared %d states x 3 decisions: mismatches = %d" % (art, len(states), bad))
sys.exit(1 if bad else 0)
