"""Shared helpers for the diagnostics. TRAIN-ONLY. Never imports or opens dev data."""
import json
import math
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import model as M          # noqa: E402
import selfcheck as SC     # noqa: E402
import train as T          # noqa: E402

Q = M.QUEUES
REPORTS = os.path.join(ROOT, "diagnostics", "reports")
os.makedirs(REPORTS, exist_ok=True)


def load_cache(seed=13):
    z = np.load(os.path.join(ROOT, "diagnostics", "cache", "oof_seed%d.npz" % seed))
    return z["Z"], z["y_route"], z["y_app"].astype(bool), z["y_gr"].astype(bool), z["fold"]


def load_train():
    return T.load_jsonl(os.path.join(ROOT, "data", "train.jsonl"))


def save_json(name, obj):
    with open(os.path.join(REPORTS, name), "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))


# ---------------------------------------------------------------------------
# calibrators, all cross-fitted (5-fold) so evaluation never sees its own fit
# ---------------------------------------------------------------------------

def fit_temp_binary(z, y):
    best, best_nll = 1.0, 1e9
    for t in np.exp(np.linspace(math.log(0.3), math.log(6.0), 241)):
        p = np.clip(M.sigmoid(z / t), 1e-12, 1 - 1e-12)
        nll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        if nll < best_nll:
            best, best_nll = float(t), nll
    return best


def crossfit_binary_any(z, y, kind, seed=101, k=5):
    y = y.astype(float)
    if kind == "raw":
        return np.clip(M.sigmoid(z), M.P_EPS, 1 - M.P_EPS)
    p = np.zeros(len(z))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(z, y):
        if kind == "temperature":
            p[te] = np.clip(M.sigmoid(z[te] / fit_temp_binary(z[tr], y[tr])), M.P_EPS, 1 - M.P_EPS)
        else:
            p[te] = M.apply_binary_cal(z[te], T.fit_binary_cal(z[tr], y[tr], kind))
    return p


def crossfit_route_any(Zr, y, kind, seed=101, k=5):
    n = len(y)
    if kind == "raw":
        return M.softmax_T(Zr, 1.0)
    P = np.zeros((n, 5))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(Zr, y):
        if kind == "temperature":
            P[te] = M.softmax_T(Zr[te], T.fit_temperature(Zr[tr], y[tr]))
            continue
        base_tr, base_te = M.softmax_T(Zr[tr], 1.0), M.softmax_T(Zr[te], 1.0)
        out = np.zeros((len(te), 5))
        for c in range(5):
            yc = (y[tr] == c).astype(float)
            if kind == "ovr_platt":
                lg = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
                cal = T.fit_binary_cal(lg(base_tr[:, c]), yc, "platt")
                out[:, c] = M.apply_binary_cal(lg(base_te[:, c]), cal)
            else:   # ovr_isotonic
                cal = T.fit_binary_cal(np.log(np.clip(base_tr[:, c], 1e-6, 1)) - np.log(np.clip(1 - base_tr[:, c], 1e-6, 1)), yc, "isotonic")
                zz = np.log(np.clip(base_te[:, c], 1e-6, 1)) - np.log(np.clip(1 - base_te[:, c], 1e-6, 1))
                out[:, c] = M.apply_binary_cal(zz, cal)
        P[te] = out / out.sum(axis=1, keepdims=True)
    return P


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for i in range(bins):
        m = (p > edges[i]) & (p <= edges[i + 1]) if i else (p <= edges[1])
        if m.sum():
            out.append({"bin": "%.1f-%.1f" % (edges[i], edges[i + 1]), "n": int(m.sum()),
                        "mean_p": float(p[m].mean()), "observed": float(y[m].mean())})
    return out


def equal_mass_ece(conf, corr, bins=15):
    order = np.argsort(conf)
    parts = np.array_split(order, bins)
    return float(sum(len(ix) / len(conf) * abs(conf[ix].mean() - corr[ix].mean()) for ix in parts if len(ix)))


def binary_metrics(p, y):
    y = y.astype(float)
    conf = np.maximum(p, 1 - p)
    corr = ((p > 0.5) == (y > 0.5)).astype(float)
    err = 1 - corr.mean()
    risk80 = SC.selective_risk(conf.tolist(), corr.tolist())
    base = y.mean()
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return {
        "ece15": SC.ece(p.tolist(), y.tolist()),
        "ece15_equal_mass": equal_mass_ece(p, y),
        "brier": SC.brier(p.tolist(), y.tolist()),
        "brier_skill": SC.brier_skill(SC.brier(p.tolist(), y.tolist()), base * (1 - base)),
        "nll": float(-np.mean(y * np.log(pc) + (1 - y) * np.log(1 - pc))),
        "auroc_label": float(roc_auc_score(y, p)),
        "full_error": float(err),
        "risk80": risk80,
        "abstention_part": max(0.0, 1 - risk80 / err) if err > 1e-9 else 1.0,
        "auroc_conf_vs_correct": float(roc_auc_score(corr, conf)) if 0 < corr.mean() < 1 else None,
        "unique_p": int(len(np.unique(np.round(p, 9)))),
    }


def route_metrics(P, y):
    top = P.argmax(1)
    conf = P.max(1)
    corr = (top == y).astype(float)
    err = 1 - corr.mean()
    truth = [Q[i] for i in y]
    rows = [{q: float(P[i, j]) for j, q in enumerate(Q)} for i in range(len(y))]
    prior = {q: float(np.mean(y == i)) for i, q in enumerate(Q)}
    br = SC.multiclass_brier(rows, truth, Q)
    ref = SC.multiclass_brier([prior] * len(y), truth, Q)
    risk80 = SC.selective_risk(conf.tolist(), corr.tolist())
    nll = -float(np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, 1))))
    return {
        "acc": float(corr.mean()),
        "ece15_top": SC.ece(conf.tolist(), corr.tolist()),
        "ece15_equal_mass": equal_mass_ece(conf, corr),
        "brier": br, "brier_skill": SC.brier_skill(br, ref), "nll": nll,
        "risk80": risk80, "full_error": float(err),
        "abstention_part": max(0.0, 1 - risk80 / err) if err > 1e-9 else 1.0,
        "auroc_conf_vs_correct": float(roc_auc_score(corr, conf)),
        "macro_f1": SC.macro_f1([(truth[i], Q[top[i]]) for i in range(len(y))], Q),
    }


def oracle_risk(err, cov):
    """Selective risk if the confidence ranking were perfect (all errors ranked last)."""
    return max(0.0, err - (1 - cov)) / cov


def selective_curve(conf, corr, covs=(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0)):
    err = 1 - float(np.mean(corr))
    return [{"coverage": c, "risk": SC.selective_risk(conf.tolist(), corr.tolist(), c),
             "oracle_risk": oracle_risk(err, c)} for c in covs]
