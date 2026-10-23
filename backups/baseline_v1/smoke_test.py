#!/usr/bin/env python3
"""
Offline sanity checks for model.DecisionModel. Uses TRAIN rows only (never dev),
so it says nothing about accuracy - it checks contracts, robustness and speed.

    python smoke_test.py --artifacts artifacts --n 300
"""

import argparse
import json
import math
import statistics
import sys
import time

import model as M


def pct(xs, p):
    s = sorted(xs)
    return s[min(len(s) - 1, int(round((len(s) - 1) * p)))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--data", default="data/train.jsonl")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()

    failures = []

    def check(cond, msg):
        if not cond:
            failures.append(msg)

    t0 = time.perf_counter()
    m = M.DecisionModel(args.artifacts)
    print("load time %.0f ms, ready=%s %s" % ((time.perf_counter() - t0) * 1000, m.ready, m.load_error or ""))
    check(m.ready, "model did not load artifacts: %s" % m.load_error)

    with open(args.data, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh][:args.n]

    def validate(state):
        v, probs, conf, ab = m.route(state)
        check(v in M.QUEUES, "route value invalid")
        check(set(probs) == set(M.QUEUES), "route probs keys")
        check(abs(sum(probs.values()) - 1.0) <= 0.01, "route probs do not sum to 1")
        check(all(0.0 <= p <= 1.0 and math.isfinite(p) for p in probs.values()), "route prob out of range")
        check(abs(probs[v] - conf) <= 1e-9, "confidence != probs[value]")
        check(isinstance(ab, bool), "route abstain not bool")
        for name in ("approve", "grounded"):
            val, p, ab = getattr(m, name)(state)
            check(isinstance(val, bool) and isinstance(ab, bool), name + " types")
            check(0.0 <= p <= 1.0 and math.isfinite(p), name + " p_true range")

    for r in rows:
        validate(r["state"])

    # determinism: same input twice, cached and uncached, must agree exactly
    s = rows[0]["state"]
    check(m.route(s) == m.route(dict(s)), "route not deterministic")
    check(m._compute(s) == m._compute(s), "uncached compute not deterministic")

    # robustness: odd states must not raise and must stay valid
    odd = [{}, {"subject": None, "body": None, "customer_tier": None, "amount": None,
               "draft_reply": None, "kb_article": None},
           {"subject": "x", "body": "y" * 5000, "customer_tier": "platinum", "amount": -5.0,
            "draft_reply": "", "kb_article": ""},
           {"subject": "refund", "body": "refund please", "customer_tier": "Pro", "amount": "12.5",
            "draft_reply": "d", "kb_article": "k", "unexpected": [1, 2, 3]},
           {"subject": "z", "body": "z", "customer_tier": "free", "amount": 1e12,
            "draft_reply": "sixty minutes", "kb_article": "60 minutes"},
           {"amount": float("nan")}]
    for st in odd:
        try:
            validate(st)
        except Exception as exc:                                 # noqa: BLE001
            failures.append("raised %s on %r" % (exc, str(st)[:60]))

    # latency per call, UNCACHED (the realistic worst case), one request at a time
    lat = {"route": [], "approve": [], "grounded": []}
    for r in rows:
        for name in lat:
            m._cache.clear()
            t = time.perf_counter()
            getattr(m, name)(r["state"])
            lat[name].append((time.perf_counter() - t) * 1000.0)
    print("uncached latency ms (model only, excludes HTTP):")
    for name, xs in lat.items():
        print("  %-9s p50 %.3f  p95 %.3f  max %.3f" % (name, statistics.median(xs), pct(xs, 0.95), max(xs)))

    if failures:
        print("\nFAILED (%d):" % len(failures))
        for f in sorted(set(failures)):
            print("  -", f)
        return 1
    print("\nall contract/robustness checks passed on %d train rows" % len(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
