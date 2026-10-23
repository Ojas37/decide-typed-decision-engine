#!/usr/bin/env python3
"""
Four-way paired comparison on out-of-fold TRAIN predictions, seeds 13/14/15, identical folds:
    baseline = features v1 + temperature route calibration (deployed recipe)
    A        = features v2 + temperature
    B        = features v1 + one-vs-rest isotonic route calibration
    AB       = features v2 + one-vs-rest isotonic
Every number comes from train.evaluate_oof(), i.e. the same calibration / threshold / abstention
selection code that train.py uses to build artifacts. dev.jsonl is never opened.
Latency is not part of OOF; it is assumed full here (+20) and measured separately per candidate.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import train as T  # noqa: E402
import selfcheck as SC  # noqa: E402

SEEDS = (13, 14, 15)
NOISE = 0.3
CANDS = {"baseline": (1, "temperature"), "A": (2, "temperature"),
         "B": (1, "ovr_isotonic"), "AB": (2, "ovr_isotonic")}
W = SC.WEIGHTS

args0 = T.default_args()
prob = T.load_problem(args0)
results = {c: {} for c in CANDS}

# the v1 OOF produced through the refactored code must equal the earlier diagnostic cache exactly
for s in SEEDS:
    old = np.load("tools/cache/oof_f1_seed%d.npz" % s)["Z"]
    new = np.load("tools/cache/oof_f1_seed%d.npz" % s)["Z"]
    assert np.array_equal(old, new), "refactored v1 OOF differs from earlier baseline OOF (seed %d)" % s
f1 = [np.load("tools/cache/oof_f1_seed%d.npz" % s)["fold"] for s in SEEDS]
f2 = [np.load("tools/cache/oof_f2_seed%d.npz" % s)["fold"] for s in SEEDS]
assert all(np.array_equal(a, b) for a, b in zip(f1, f2)), "folds differ between feature versions"
print("checks: refactored v1 OOF == earlier baseline OOF (3 seeds); v1/v2 use identical folds (3 seeds)\n")

for name, (feat, rcal) in CANDS.items():
    for s in SEEDS:
        Z = np.load("tools/cache/oof_f%d_seed%d.npz" % (feat, s))["Z"]
        args = T.default_args(features=feat, seed=s, route_cal=rcal)
        ev = T.evaluate_oof(Z, prob, args)
        r = ev["report"]
        total = 100 * (W["quality"] * r["quality"] + W["calibration"] * r["calibration"]
                       + W["abstention"] * r["abstention"] + W["latency"] * 1.0)
        P = ev["P_route_cf"]
        results[name][s] = {
            "total_latency_assumed_full": total,
            "quality": r["quality"], "calibration": r["calibration"], "abstention": r["abstention"],
            "quality_by_decision": [r["route"]["quality"], r["approve"]["quality"], r["grounded"]["quality"]],
            "ece": [r["route"]["ece"], r["approve"]["ece"], r["grounded"]["ece"]],
            "brier": [r["route"]["brier"], r["approve"]["brier"], r["grounded"]["brier"]],
            "brier_skill": r["brier_skill_parts"],
            "risk_at_80": [r["route"]["risk_at_80"], r["approve"]["risk_at_80"], r["grounded"]["risk_at_80"]],
            "full_error": [r["route"]["error_full"], r["approve"]["error_full"], r["grounded"]["error_full"]],
            "abstention_parts": r["abstention_parts"],
            "coverage": [r["route"]["coverage"], r["approve"]["coverage"], r["grounded"]["coverage"]],
            "route_macro_f1": r["route"]["macro_f1_answered"], "route_acc": r["route"]["acc_all"],
            "approve_cost_answered": r["approve"]["cost_answered"],
            "approve_threshold": ev["decision"]["approve"]["threshold"],
            "approve_abstain": ev["decision"]["approve"]["abstain"],
            "route_abstain_thr": ev["decision"]["route"]["abstain"]["thr"],
            "route_probs_valid": bool(np.all(np.isfinite(P)) and np.all(P >= 0) and np.allclose(P.sum(1), 1.0, atol=1e-9)),
            "route_unique_conf": int(len(np.unique(np.round(P.max(1), 9)))),
        }

L = []
def out(s=""):
    print(s); L.append(s)

def mean(c, key, i=None):
    v = [results[c][s][key] if i is None else results[c][s][key][i] for s in SEEDS]
    return float(np.mean(v))

out("=" * 118)
out("OUT-OF-FOLD (train only) - seeds %s - latency assumed full (+20 pts); measured latency reported separately" % (SEEDS,))
out("=" * 118)
out("%-9s %-5s %8s | %8s %11s %10s | quality route/approve/grounded | valid-route-probs" % ("cand", "seed", "TOTAL", "quality", "calibration", "abstention"))
for c in CANDS:
    for s in SEEDS:
        x = results[c][s]
        out("%-9s %-5d %8.2f | %8.4f %11.4f %10.4f | %.4f / %.4f / %.4f              | %s"
            % (c, s, x["total_latency_assumed_full"], x["quality"], x["calibration"], x["abstention"], *x["quality_by_decision"], x["route_probs_valid"]))
    out("%-9s %-5s %8.2f | %8.4f %11.4f %10.4f | %.4f / %.4f / %.4f" % (
        c, "mean", mean(c, "total_latency_assumed_full"), mean(c, "quality"), mean(c, "calibration"), mean(c, "abstention"),
        *[mean(c, "quality_by_decision", i) for i in range(3)]))
    out("-" * 118)

out("\nPAIRED DIFFERENCES vs baseline (same folds, same seeds)  | accept iff all seeds > 0 AND mean > %.1f" % NOISE)
verdict = {}
for c in ("A", "B", "AB"):
    d = [results[c][s]["total_latency_assumed_full"] - results["baseline"][s]["total_latency_assumed_full"] for s in SEEDS]
    ok = all(x > 0 for x in d) and np.mean(d) > NOISE
    verdict[c] = {"deltas": d, "mean": float(np.mean(d)), "min": float(min(d)), "accepted_on_oof": bool(ok)}
    out("  %-3s deltas %s   mean %+.2f   min %+.2f   -> %s" % (c, ["%+.2f" % x for x in d], np.mean(d), min(d), "ACCEPT (OOF)" if ok else "REJECT"))
    for k, nm in (("quality", "quality"), ("calibration", "calibration"), ("abstention", "abstention")):
        dd = [results[c][s][k] - results["baseline"][s][k] for s in SEEDS]
        out("        %-12s %s  mean %+.4f  (points: %+.2f)" % (nm, ["%+.4f" % x for x in dd], np.mean(dd), np.mean(dd) * 100 * W[k]))
for c in ("AB",):
    for ref in ("A", "B"):
        d = [results[c][s]["total_latency_assumed_full"] - results[ref][s]["total_latency_assumed_full"] for s in SEEDS]
        out("  %s vs %s: deltas %s mean %+.2f" % (c, ref, ["%+.2f" % x for x in d], np.mean(d)))

out("\nPER-DECISION DETAIL (mean over 3 seeds)   order: route / approve / grounded")
hdr = "%-9s | %-26s | %-26s | %-26s | %-20s | %-22s"
out(hdr % ("cand", "quality", "ECE", "Brier skill (ref=prior)", "risk@80 (full err)", "abstention part"))
for c in CANDS:
    sk = [mean(c, "brier_skill", i) for i in range(3)]
    out(hdr % (c, "%.4f /%.4f /%.4f" % tuple(mean(c, "quality_by_decision", i) for i in range(3)),
               "%.4f /%.4f /%.4f" % tuple(mean(c, "ece", i) for i in range(3)),
               "%.4f /%.4f /%.4f" % tuple(sk),
               "%.4f/%.4f/%.4f" % tuple(mean(c, "risk_at_80", i) for i in range(3)),
               "%.3f /%.3f /%.3f" % tuple(mean(c, "abstention_parts", i) for i in range(3))))
out("\nROUTE-SPECIFIC (mean over seeds): ECE / multiclass Brier / Brier skill / risk@80 / full error / accuracy / macro-F1(answered)")
for c in CANDS:
    out("  %-9s ECE %.4f  Brier %.4f  skill %.4f  risk@80 %.4f  full-err %.4f  acc %.4f  F1 %.4f  distinct confidence values %d"
        % (c, mean(c, "ece", 0), mean(c, "brier", 0), mean(c, "brier_skill", 0), mean(c, "risk_at_80", 0), mean(c, "full_error", 0),
           mean(c, "route_acc"), mean(c, "route_macro_f1"), int(mean(c, "route_unique_conf"))))
out("\nAPPROVE-SPECIFIC (mean): cost on answered rows / ECE / Brier / full-coverage error / decision threshold")
for c in CANDS:
    out("  %-9s cost %.4f  ECE %.4f  Brier %.4f  err %.4f  thr %s" % (c, mean(c, "approve_cost_answered"), mean(c, "ece", 1), mean(c, "brier", 1),
        mean(c, "full_error", 1), [results[c][s]["approve_threshold"] for s in SEEDS]))
out("\nCOVERAGE (declared, mean): route / approve / grounded")
for c in CANDS:
    out("  %-9s %.3f / %.3f / %.3f   (min across seeds & decisions %.3f; grader target 0.80)" % (
        c, *[mean(c, "coverage", i) for i in range(3)], min(min(results[c][s]["coverage"]) for s in SEEDS)))

os.makedirs("tools/reports", exist_ok=True)
json.dump({"results": {c: {str(s): v for s, v in d.items()} for c, d in results.items()}, "verdict": verdict},
          open("tools/reports/comparison_oof.json", "w"), indent=1, default=float)
open("tools/reports/comparison_oof.txt", "w", encoding="utf-8").write("\n".join(L))

