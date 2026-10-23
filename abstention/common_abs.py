"""Helpers for the abstention study. TRAIN-ONLY. Never opens dev data."""
import os
import re
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import model as M          # noqa: E402
import selfcheck as SC     # noqa: E402
import train as T          # noqa: E402

Q = M.QUEUES
W = SC.WEIGHTS
SCREEN_SEEDS = (13, 14, 15)
CONFIRM_SEEDS = (16, 17, 18)
_PROB = None


def problem():
    global _PROB
    if _PROB is None:
        _PROB = T.load_problem(T.default_args())
    return _PROB


def load_oof(seed):
    z = np.load("experiments/cache/oof_f1_seed%d.npz" % seed)
    return z["Z"], z["fold"]


def part(conf, corr):
    """selfcheck abstention part: 1 - risk@80 / full error."""
    corr = np.asarray(corr, dtype=float)
    err = 1.0 - corr.mean()
    risk = SC.selective_risk(list(map(float, conf)), corr.tolist())
    return (max(0.0, 1.0 - risk / err) if err > 1e-9 else 1.0), risk, err


def baseline_B(Z, seed):
    """Candidate B's cross-fitted probabilities (route OvR isotonic, approve isotonic, grounded Platt)."""
    prob = problem()
    P = T.crossfit_route(Z[:, :5], prob.y_route, "ovr_isotonic", seed)
    pa = T.choose_binary_cal(Z[:, 5], prob.y_app.astype(float), seed)[1]
    pg = T.choose_binary_cal(Z[:, 6], prob.y_gr.astype(float), seed)[1]
    return P, pa, pg


def score(P, pa, pg, seed, coverage=0.83):
    """Same selection + scoring as train.evaluate_oof, for arbitrary probability arrays."""
    prob = problem()
    T.check_route_probs(P)
    assert np.all(np.isfinite(pa)) and pa.min() >= 0 and pa.max() <= 1
    r_rule = T.coverage_rule("min_conf", P.max(1), coverage)
    sel_a = T.select_approve(prob.y_app, pa, coverage)
    sel_g = T.select_grounded(prob.y_gr, pg, coverage)
    dec = {"route": {"abstain": r_rule},
           "approve": {"threshold": sel_a["threshold"], "abstain": sel_a["rule"]},
           "grounded": {"threshold": sel_g["threshold"], "abstain": sel_g["rule"]}}
    r = T.oof_pseudo_score(P, prob.y_route, pa, prob.y_app, pg, prob.y_gr, dec)
    total = 100 * (W["quality"] * r["quality"] + W["calibration"] * r["calibration"]
                   + W["abstention"] * r["abstention"] + W["latency"] * 1.0)
    return {"total": total, "quality": r["quality"], "calibration": r["calibration"], "abstention": r["abstention"],
            "q": [r["route"]["quality"], r["approve"]["quality"], r["grounded"]["quality"]],
            "ece": [r["route"]["ece"], r["approve"]["ece"], r["grounded"]["ece"]],
            "brier_skill": r["brier_skill_parts"], "abst_parts": r["abstention_parts"],
            "risk80": [r["route"]["risk_at_80"], r["approve"]["risk_at_80"], r["grounded"]["risk_at_80"]],
            "err": [r["route"]["error_full"], r["approve"]["error_full"], r["grounded"]["error_full"]],
            "cov": [r["route"]["coverage"], r["approve"]["coverage"], r["grounded"]["coverage"]],
            "route_brier": r["route"]["brier"], "route_acc": r["route"]["acc_all"],
            "approve_cost": r["approve"]["cost_answered"]}


def entropy(P):
    return -(P * np.log(np.clip(P, 1e-12, 1))).sum(1)


# discourse markers found in the train data in the earlier diagnostics (primary + secondary frames)
MARKER_RE = re.compile("Priority for me is:|My real question is this:|The part I need you to action is:|"
                       "The main thing I need resolved is:|What I actually need help with is this:|Most importantly:|"
                       "Less important, but while I am writing:|Some context first:|For background,|"
                       "Separately, and this is less urgent,|On a related note,|Also worth mentioning,", re.I)
