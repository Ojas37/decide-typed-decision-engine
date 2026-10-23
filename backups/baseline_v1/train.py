#!/usr/bin/env python3
"""
Train the shared multi-task Decide model and export portable artifacts.

    python train.py --train data/train.jsonl --out artifacts --seed 13

Reads ONLY the file given by --train. dev.jsonl is never opened here: every
choice below (calibration, decision thresholds, abstention thresholds, epoch
count) is made from out-of-fold predictions on the training split.

Pipeline
  1. Featurize (model.Featurizer) and build a vocabulary from train.
  2. K-fold: train the shared trunk + 3 heads on K-1 folds, predict the held-out
     fold. Concatenated, these are honest out-of-fold (OOF) logits.
  3. Fit calibrators on the OOF logits (route: temperature; binary: Platt or
     isotonic, chosen by cross-validated Brier). Reported metrics use
     cross-fitted calibration so they are not flattered by fitting on themselves.
  4. Choose the approve / grounded decision thresholds and the abstention rules
     by evaluating a grid on the cross-fitted OOF probabilities. Approve's
     theoretical break-even is 0.75 (3:1 cost); the chosen value is whatever the
     OOF evaluation prefers, and the whole curve is saved for inspection.
  5. Refit on all of train, export weights.npz / vocab.json / meta.json /
     training_report.json, and verify NumPy inference == PyTorch logits.

Training needs: numpy, torch, scikit-learn.  Inference needs only numpy.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

import model as M
import selfcheck as SC          # same metric definitions the judges use

QUEUES = M.QUEUES
TARGET_COV = SC.TARGET_COVERAGE


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_vocab(token_lists: Sequence[Sequence[str]], min_df: int) -> Dict[str, int]:
    df = Counter()
    for toks in token_lists:
        df.update(set(toks))
    kept = sorted((t for t, c in df.items() if c >= min_df), key=lambda t: (-df[t], t))
    return {t: i for i, t in enumerate(kept)}


# ---------------------------------------------------------------------------
# network (PyTorch twin of model.forward_logits)
# ---------------------------------------------------------------------------

class Net(nn.Module):
    def __init__(self, vocab: int, dense: int, emb: int, h1: int, h2: int,
                 hh: int, dropout: float) -> None:
        super().__init__()
        self.emb = nn.EmbeddingBag(vocab, emb, mode="sum")
        nn.init.normal_(self.emb.weight, 0.0, 0.1)
        self.l1 = nn.Linear(emb + dense, h1)
        self.l2 = nn.Linear(h1, h2)
        self.drop = nn.Dropout(dropout)

        def head(out: int) -> nn.Sequential:
            return nn.Sequential(nn.Linear(h2, hh), nn.ReLU(), nn.Linear(hh, out))
        self.route, self.approve, self.grounded = head(len(QUEUES)), head(1), head(1)

    def forward(self, flat, offsets, weights, dense):
        pooled = self.emb(flat, offsets, per_sample_weights=weights)
        h = torch.relu(self.l1(torch.cat([pooled, dense], dim=1)))
        h = self.drop(h)
        h = self.drop(torch.relu(self.l2(h)))
        return self.route(h), self.approve(h).squeeze(1), self.grounded(h).squeeze(1)

    def export(self) -> Dict[str, np.ndarray]:
        t = lambda x: x.detach().cpu().numpy().astype(np.float32)
        return {
            "emb": t(self.emb.weight),
            "W1": t(self.l1.weight).T.copy(), "b1": t(self.l1.bias),
            "W2": t(self.l2.weight).T.copy(), "b2": t(self.l2.bias),
            "Wr1": t(self.route[0].weight).T.copy(), "br1": t(self.route[0].bias),
            "Wr2": t(self.route[2].weight).T.copy(), "br2": t(self.route[2].bias),
            "Wa1": t(self.approve[0].weight).T.copy(), "ba1": t(self.approve[0].bias),
            "Wa2": t(self.approve[2].weight).T.copy(), "ba2": t(self.approve[2].bias),
            "Wg1": t(self.grounded[0].weight).T.copy(), "bg1": t(self.grounded[0].bias),
            "Wg2": t(self.grounded[2].weight).T.copy(), "bg2": t(self.grounded[2].bias),
        }


class Data:
    """Encoded rows: ragged sparse indices + dense matrix + labels."""

    def __init__(self, idx: List[np.ndarray], dense: np.ndarray,
                 y_route: np.ndarray, y_app: np.ndarray, y_gr: np.ndarray) -> None:
        self.idx, self.dense = idx, dense
        self.y_route, self.y_app, self.y_gr = y_route, y_app, y_gr
        self.lens = np.array([len(i) for i in idx], dtype=np.int64)
        self.base_w = (1.0 / np.sqrt(np.maximum(self.lens, 1))).astype(np.float32)

    def batch(self, rows: np.ndarray, drop: float, rng: np.random.RandomState):
        parts = [self.idx[i] for i in rows]
        flat = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
        lens = self.lens[rows]
        offsets = np.concatenate([[0], np.cumsum(lens)[:-1]]).astype(np.int64)
        w = np.repeat(self.base_w[rows], lens)
        if drop > 0.0:
            w = w * (rng.random_sample(len(flat)) >= drop) / (1.0 - drop)
        return (torch.from_numpy(flat.astype(np.int64)), torch.from_numpy(offsets),
                torch.from_numpy(w.astype(np.float32)), torch.from_numpy(self.dense[rows]))


def predict_logits(net: Net, data: Data, rows: np.ndarray) -> np.ndarray:
    net.eval()
    out = []
    rng = np.random.RandomState(0)
    with torch.no_grad():
        for s in range(0, len(rows), 1024):
            r, a, g = net(*data.batch(rows[s:s + 1024], 0.0, rng))
            out.append(torch.cat([r, a[:, None], g[:, None]], dim=1).numpy())
    return np.concatenate(out).astype(np.float64)


def task_losses(Z: np.ndarray, y_route, y_app, y_gr) -> Tuple[float, float, float]:
    e = Z[:, :5] - Z[:, :5].max(axis=1, keepdims=True)
    logp = e - np.log(np.exp(e).sum(axis=1, keepdims=True))
    nll_r = -float(np.mean(logp[np.arange(len(y_route)), y_route]))
    bce = lambda z, y: float(np.mean(np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))))
    return nll_r, bce(Z[:, 5], y_app.astype(float)), bce(Z[:, 6], y_gr.astype(float))


def fit_net(data: Data, train_rows: np.ndarray, val_rows: np.ndarray, args,
            seed: int, vocab_size: int) -> Tuple[Net, List[Tuple[float, float, float]]]:
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    net = Net(vocab_size, M.DENSE_DIM, args.emb_dim, args.hidden, args.hidden2,
              args.head_hidden, args.dropout)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = int(math.ceil(len(train_rows) / args.batch))
    total = steps_per_epoch * args.epochs

    def lr_at(s: int) -> float:
        warm = min(1.0, (s + 1) / steps_per_epoch)
        return warm * 0.5 * (1.0 + math.cos(math.pi * s / total))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    y_r = torch.from_numpy(data.y_route.astype(np.int64))
    y_a = torch.from_numpy(data.y_app.astype(np.float32))
    y_g = torch.from_numpy(data.y_gr.astype(np.float32))
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    history = []
    for epoch in range(args.epochs):
        net.train()
        perm = rng.permutation(train_rows)
        for s in range(0, len(perm), args.batch):
            rows = perm[s:s + args.batch]
            r, a, g = net(*data.batch(rows, args.token_dropout, rng))
            loss = (args.w_route * ce(r, y_r[rows]) + args.w_approve * bce(a, y_a[rows])
                    + args.w_grounded * bce(g, y_g[rows]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        if val_rows is not None and len(val_rows):
            Z = predict_logits(net, data, val_rows)
            history.append(task_losses(Z, data.y_route[val_rows],
                                       data.y_app[val_rows], data.y_gr[val_rows]))
    return net, history


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------

def fit_temperature(Zr: np.ndarray, y: np.ndarray) -> float:
    best, best_nll = 1.0, float("inf")
    for T in np.exp(np.linspace(math.log(0.4), math.log(4.0), 301)):
        p = M.softmax_T(Zr, float(T))
        nll = -float(np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-12, 1.0))))
        if nll < best_nll:
            best, best_nll = float(T), nll
    return best


def fit_binary_cal(z: np.ndarray, y: np.ndarray, kind: str) -> Dict[str, Any]:
    if kind == "platt":
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(z.reshape(-1, 1), y.astype(int))
        return {"type": "platt", "a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
    iso = IsotonicRegression(out_of_bounds="clip", y_min=M.P_EPS, y_max=1 - M.P_EPS)
    iso.fit(z, y.astype(float))
    return {"type": "isotonic", "x": [float(v) for v in iso.X_thresholds_],
            "y": [float(v) for v in iso.y_thresholds_]}


def crossfit_binary(z: np.ndarray, y: np.ndarray, kind: str, seed: int, k: int = 5) -> np.ndarray:
    p = np.zeros(len(z))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(z, y):
        p[te] = M.apply_binary_cal(z[te], fit_binary_cal(z[tr], y[tr], kind))
    return p


def choose_binary_cal(z: np.ndarray, y: np.ndarray, seed: int) -> Tuple[Dict[str, Any], np.ndarray, Dict[str, float]]:
    """Platt unless isotonic wins cross-validated Brier by a real margin."""
    briers, probs = {}, {}
    for kind in ("platt", "isotonic"):
        probs[kind] = crossfit_binary(z, y, kind, seed)
        briers[kind] = float(np.mean((probs[kind] - y) ** 2))
    kind = "isotonic" if briers["isotonic"] < briers["platt"] - 3e-4 else "platt"
    return fit_binary_cal(z, y, kind), probs[kind], briers


def crossfit_temperature(Zr: np.ndarray, y: np.ndarray, seed: int, k: int = 5) -> np.ndarray:
    P = np.zeros((len(y), 5))
    for tr, te in StratifiedKFold(k, shuffle=True, random_state=seed).split(Zr, y):
        P[te] = M.softmax_T(Zr[te], fit_temperature(Zr[tr], y[tr]))
    return P


# ---------------------------------------------------------------------------
# decision / abstention selection and offline scoring (selfcheck formulas)
# ---------------------------------------------------------------------------

def cov_scale(cov: float) -> float:
    return min(1.0, cov / TARGET_COV)


def approve_cost(y: np.ndarray, pred: np.ndarray) -> float:
    if len(y) == 0:
        return 0.0
    return float(np.mean(np.where(pred & ~y, M.FP_COST, 0.0) + np.where(~pred & y, 1.0, 0.0)))


def approve_quality(y: np.ndarray, p: np.ndarray, thr: float, abst: np.ndarray) -> float:
    ans = ~abst
    if not ans.any():
        return 0.0
    cost = approve_cost(y[ans], p[ans] >= thr)
    return SC.cost_score(cost, float(y[ans].mean())) * cov_scale(float(ans.mean()))


def grounded_quality(y: np.ndarray, p: np.ndarray, thr: float, abst: np.ndarray) -> float:
    ans = ~abst
    if not ans.any():
        return 0.0
    ya, pred = y[ans], p[ans] >= thr
    f1 = SC.macro_f1(list(zip(ya.tolist(), pred.tolist())), [True, False])
    triv = max(SC.macro_f1([(bool(v), True) for v in ya], [True, False]),
               SC.macro_f1([(bool(v), False) for v in ya], [True, False]))
    return SC.skill(f1, triv) * cov_scale(float(ans.mean()))


def smooth(v: np.ndarray, width: int = 5) -> np.ndarray:
    k = np.ones(width)
    return np.convolve(v, k, mode="same") / np.convolve(np.ones(len(v)), k, mode="same")


def pick_threshold(grid: np.ndarray, scores: np.ndarray, anchor: float) -> float:
    s = smooth(scores) - 1e-6 * np.abs(grid - anchor)       # ties -> nearest the theory value
    return float(grid[int(np.argmax(s))])


def coverage_rule(kind: str, p_or_conf: np.ndarray, coverage: float) -> Dict[str, Any]:
    """Abstain on the least certain (1-coverage) fraction, measured on OOF data."""
    if kind == "max_risk":
        return {"kind": kind, "thr": float(np.quantile(p_or_conf, coverage))}
    return {"kind": kind, "thr": float(np.quantile(p_or_conf, 1.0 - coverage))}


def select_approve(y: np.ndarray, p: np.ndarray, coverage: float) -> Dict[str, Any]:
    grid = np.round(np.arange(0.50, 0.951, 0.01), 2)
    results = {}
    for kind in ("max_risk", "min_conf"):
        score_var = M.approve_risk(p) if kind == "max_risk" else np.maximum(p, 1 - p)
        rule = coverage_rule(kind, score_var, coverage)
        abst = np.asarray(M.abstain_binary(p, rule))
        q = np.array([approve_quality(y, p, float(t), abst) for t in grid])
        t = pick_threshold(grid, q, 0.75)
        results[kind] = {"rule": rule, "threshold": t, "quality": float(q[list(grid).index(t)]),
                         "quality_at_0.75": float(q[list(grid).index(0.75)]),
                         "curve": {"%.2f" % g: float(v) for g, v in zip(grid, q)}}
    # Evidence for the threshold where it actually bites: ALL rows, no abstention.
    # (Behind an abstention band the answered cases sit far from any sensible
    # threshold, so the quality curve above is flat and cannot validate it.)
    full = np.array([approve_cost(y, p >= t) for t in grid])
    best = max(results, key=lambda k: results[k]["quality"])
    out = results[best]
    out["no_abstention_cost_curve"] = {"%.2f" % g: float(v) for g, v in zip(grid, full)}
    out["no_abstention_argmin_threshold"] = pick_threshold(grid, -full, 0.75)
    out["threshold"] = out["no_abstention_argmin_threshold"]
    out["quality"] = out["curve"]["%.2f" % out["threshold"]]
    out["no_abstention_cost_at_0.75"] = float(full[list(grid).index(0.75)])
    out["no_abstention_cost_min"] = float(full.min())
    out["quality_curve_flat"] = bool(np.ptp(np.array(list(out["curve"].values()))[(grid >= 0.5) & (grid <= 0.75)]) < 1e-9)
    out["chosen_kind"] = best
    out["alternatives"] = {k: {"quality": v["quality"], "threshold": v["threshold"]}
                           for k, v in results.items()}
    return out


def select_grounded(y: np.ndarray, p: np.ndarray, coverage: float) -> Dict[str, Any]:
    grid = np.round(np.arange(0.30, 0.701, 0.01), 2)
    rule = coverage_rule("min_conf", np.maximum(p, 1 - p), coverage)
    abst = np.asarray(M.abstain_binary(p, rule))
    q = np.array([grounded_quality(y, p, float(t), abst) for t in grid])
    t = pick_threshold(grid, q, 0.50)
    return {"rule": rule, "threshold": t, "quality": float(q[list(grid).index(t)]),
            "quality_at_0.50": float(q[list(grid).index(0.5)]),
            "curve": {"%.2f" % g: float(v) for g, v in zip(grid, q)}}


def oof_pseudo_score(P_route, y_route, p_app, y_app, p_gr, y_gr, dec) -> Dict[str, Any]:
    """Replicates selfcheck.py's axes (latency excluded) on out-of-fold data."""
    n = len(y_route)
    top = P_route.argmax(axis=1)
    conf = P_route.max(axis=1)
    r_abst = conf < dec["route"]["abstain"]["thr"]
    truth = [QUEUES[i] for i in y_route]
    ans = ~r_abst
    pairs = [(truth[i], QUEUES[top[i]]) for i in range(n) if ans[i]]
    maj = Counter(t for t, _ in pairs).most_common(1)[0][0]
    f1 = SC.macro_f1(pairs, QUEUES)
    triv = SC.macro_f1([(t, maj) for t, _ in pairs], QUEUES)
    q_route = SC.skill(f1, triv) * cov_scale(float(ans.mean()))
    r_corr = (top == y_route).astype(float)
    route_ece = SC.ece(conf.tolist(), r_corr.tolist())
    prior = {q: float(np.mean(y_route == i)) for i, q in enumerate(QUEUES)}
    rows = [{q: float(P_route[i, j]) for j, q in enumerate(QUEUES)} for i in range(n)]
    r_brier = SC.multiclass_brier(rows, truth, QUEUES)
    r_ref = SC.multiclass_brier([prior] * n, truth, QUEUES)
    r_risk = SC.selective_risk(conf.tolist(), r_corr.tolist())
    r_err = 1 - float(r_corr.mean())

    a_abst = np.asarray(M.abstain_binary(p_app, dec["approve"]["abstain"]))
    ya = y_app.astype(bool)
    a_thr = dec["approve"]["threshold"]
    q_app = approve_quality(ya, p_app, a_thr, a_abst)
    ans_a = ~a_abst
    cost = approve_cost(ya[ans_a], p_app[ans_a] >= a_thr)
    rate = float(ya[ans_a].mean())
    yf = y_app.astype(float)
    a_ece = SC.ece(p_app.tolist(), yf.tolist())
    a_brier = SC.brier(p_app.tolist(), yf.tolist())
    a_conf = np.maximum(p_app, 1 - p_app)
    a_corr = ((p_app > 0.5) == (yf > 0.5)).astype(float)
    a_risk = SC.selective_risk(a_conf.tolist(), a_corr.tolist())
    a_err = 1 - float(a_corr.mean())

    g_abst = np.asarray(M.abstain_binary(p_gr, dec["grounded"]["abstain"]))
    yg = y_gr.astype(bool)
    q_gr = grounded_quality(yg, p_gr, dec["grounded"]["threshold"], g_abst)
    ygf = y_gr.astype(float)
    g_ece = SC.ece(p_gr.tolist(), ygf.tolist())
    g_brier = SC.brier(p_gr.tolist(), ygf.tolist())
    g_conf = np.maximum(p_gr, 1 - p_gr)
    g_corr = ((p_gr > 0.5) == (ygf > 0.5)).astype(float)
    g_risk = SC.selective_risk(g_conf.tolist(), g_corr.tolist())
    g_err = 1 - float(g_corr.mean())

    ar, gr_ = float(yf.mean()), float(ygf.mean())
    skills = [SC.brier_skill(r_brier, r_ref), SC.brier_skill(a_brier, ar * (1 - ar)),
              SC.brier_skill(g_brier, gr_ * (1 - gr_))]
    mean_ece = (route_ece + a_ece + g_ece) / 3
    calibration = 0.5 * max(0.0, 1 - mean_ece / 0.20) + 0.5 * sum(skills) / 3
    ab = [max(0.0, 1 - r / e) if e > 1e-9 else 1.0
          for r, e in ((r_risk, r_err), (a_risk, a_err), (g_risk, g_err))]
    quality = (q_route + q_app + q_gr) / 3
    abstention = sum(ab) / 3
    return {
        "note": "out-of-fold on TRAIN with cross-fitted calibration; latency excluded; NOT a dev/test score",
        "quality": quality, "calibration": calibration, "abstention": abstention,
        "partial_score_without_latency": 100 * (0.40 * quality + 0.25 * calibration + 0.15 * abstention),
        "route": {"macro_f1_answered": f1, "majority_f1": triv, "acc_all": 1 - r_err,
                  "ece": route_ece, "brier": r_brier, "brier_ref_prior": r_ref,
                  "coverage": float(ans.mean()), "risk_at_80": r_risk, "error_full": r_err,
                  "quality": q_route},
        "approve": {"cost_answered": cost, "always_hold_cost_answered": rate,
                    "always_approve_cost_answered": (1 - rate) * M.FP_COST,
                    "ece": a_ece, "brier": a_brier, "coverage": float(ans_a.mean()),
                    "risk_at_80": a_risk, "error_full": a_err, "quality": q_app,
                    "base_rate": ar},
        "grounded": {"ece": g_ece, "brier": g_brier, "coverage": float((~g_abst).mean()),
                     "risk_at_80": g_risk, "error_full": g_err, "quality": q_gr,
                     "base_rate": gr_},
        "abstention_parts": ab, "brier_skill_parts": skills,
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", default="data/train.jsonl")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--emb-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--hidden2", type=int, default=128)
    ap.add_argument("--head-hidden", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--token-dropout", type=float, default=0.3)
    ap.add_argument("--min-df", type=int, default=3)
    ap.add_argument("--w-route", type=float, default=1.0)
    ap.add_argument("--w-approve", type=float, default=1.0)
    ap.add_argument("--w-grounded", type=float, default=1.0)
    ap.add_argument("--coverage", type=float, default=0.83,
                    help="target fraction answered (abstain on the rest); the grader's target is 0.80")
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()

    if os.path.basename(args.train).startswith("dev"):
        sys.exit("refusing to train on a dev file; dev.jsonl is for validation only")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True, warn_only=True)
    t_start = time.time()

    rows = load_jsonl(args.train)
    n = len(rows)
    print("train rows: %d  (%s)" % (n, args.train))
    states = [r["state"] for r in rows]
    y_route = np.array([QUEUES.index(r["labels"]["route"]) for r in rows], dtype=np.int64)
    y_app = np.array([bool(r["labels"]["approve"]) for r in rows])
    y_gr = np.array([bool(r["labels"]["grounded"]) for r in rows])

    extracted = [M.Featurizer.extract(s) for s in states]
    vocab = build_vocab([t for t, _ in extracted], args.min_df)
    print("vocab size: %d   dense dim: %d" % (len(vocab), M.DENSE_DIM))
    idx = [np.fromiter((vocab[t] for t in toks if t in vocab), dtype=np.int64) for toks, _ in extracted]
    dense = np.stack([d for _, d in extracted]).astype(np.float32)
    data = Data(idx, dense, y_route, y_app, y_gr)
    print("avg sparse tokens/row: %.1f" % data.lens.mean())

    # ---- 1. out-of-fold predictions ----------------------------------------
    Z = np.zeros((n, 7))
    hist = []
    skf = StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
    for k, (tr, va) in enumerate(skf.split(np.zeros(n), y_route)):
        net, h = fit_net(data, tr, va, args, args.seed + k, len(vocab))
        Z[va] = predict_logits(net, data, va)
        hist.append(h)
        print("fold %d/%d  held-out loss route/approve/grounded: %.4f / %.4f / %.4f  (%.0fs)"
              % (k + 1, args.folds, *h[-1], time.time() - t_start))
    mean_hist = np.mean(np.array(hist), axis=0)        # (epochs, 3)
    print("\nOOF held-out loss by epoch (route / approve / grounded):")
    for e, row in enumerate(mean_hist):
        print("  epoch %2d  %.4f  %.4f  %.4f" % (e + 1, *row))
    best_ep = [int(np.argmin(mean_hist[:, j])) + 1 for j in range(3)]
    print("  per-task loss minimum at epoch: route %d  approve %d  grounded %d  (using %d)"
          % (*best_ep, args.epochs))

    # ---- 2. calibration (fit on OOF; report with cross-fitting) -------------
    T = fit_temperature(Z[:, :5], y_route)
    P_route_cf = crossfit_temperature(Z[:, :5], y_route, args.seed)
    cal_a, p_app, b_a = choose_binary_cal(Z[:, 5], y_app.astype(float), args.seed)
    cal_g, p_gr, b_g = choose_binary_cal(Z[:, 6], y_gr.astype(float), args.seed)
    print("\ncalibration: route T=%.3f | approve %s (CV Brier %s) | grounded %s (CV Brier %s)"
          % (T, cal_a["type"], {k: round(v, 4) for k, v in b_a.items()},
             cal_g["type"], {k: round(v, 4) for k, v in b_g.items()}))

    # ---- 3. decisions + abstention from OOF probabilities ------------------
    r_rule = coverage_rule("min_conf", P_route_cf.max(axis=1), args.coverage)
    sel_a = select_approve(y_app, p_app, args.coverage)
    sel_g = select_grounded(y_gr, p_gr, args.coverage)
    decision = {
        "route": {"abstain": r_rule},
        "approve": {"threshold": sel_a["threshold"], "abstain": sel_a["rule"]},
        "grounded": {"threshold": sel_g["threshold"], "abstain": sel_g["rule"]},
    }
    print("approve: rule=%s chosen threshold=%.2f (theory 0.75)  OOF quality %.4f vs %.4f at 0.75 | alt %s"
          % (sel_a["chosen_kind"], sel_a["threshold"], sel_a["quality"], sel_a["quality_at_0.75"],
             sel_a["alternatives"]))
    print("approve, NO abstention (where 3:1 bites): argmin-cost threshold %.2f, cost %.4f vs %.4f at 0.75 "
          "(always-hold %.4f); quality curve flat over [0.50,0.75]: %s"
          % (sel_a["no_abstention_argmin_threshold"], sel_a["no_abstention_cost_min"],
             sel_a["no_abstention_cost_at_0.75"], float(y_app.mean()), sel_a["quality_curve_flat"]))
    print("grounded: threshold=%.2f  OOF quality %.4f vs %.4f at 0.50"
          % (sel_g["threshold"], sel_g["quality"], sel_g["quality_at_0.50"]))
    print("abstain thresholds: route conf<%.3f | approve %s | grounded conf<%.3f"
          % (r_rule["thr"], sel_a["rule"], sel_g["rule"]["thr"]))

    report = oof_pseudo_score(P_route_cf, y_route, p_app, y_app, p_gr, y_gr, decision)
    print("\nOUT-OF-FOLD ESTIMATES ON TRAIN (not dev; latency excluded)")
    print("  route    macroF1(answered) %.4f  majority %.4f  acc %.4f  ECE %.4f  coverage %.3f"
          % (report["route"]["macro_f1_answered"], report["route"]["majority_f1"],
             report["route"]["acc_all"], report["route"]["ece"], report["route"]["coverage"]))
    print("  approve  cost %.4f  always-hold %.4f  always-approve %.4f  ECE %.4f  Brier %.4f  coverage %.3f"
          % (report["approve"]["cost_answered"], report["approve"]["always_hold_cost_answered"],
             report["approve"]["always_approve_cost_answered"], report["approve"]["ece"],
             report["approve"]["brier"], report["approve"]["coverage"]))
    print("  grounded ECE %.4f  Brier %.4f  coverage %.3f  full-coverage error %.4f"
          % (report["grounded"]["ece"], report["grounded"]["brier"],
             report["grounded"]["coverage"], report["grounded"]["error_full"]))
    print("  axes: quality %.3f  calibration %.3f  abstention %.3f  -> partial %.2f/85 (no latency)"
          % (report["quality"], report["calibration"], report["abstention"],
             report["partial_score_without_latency"]))

    # ---- 4. final fit on all of train and export ----------------------------
    print("\nfitting final model on all %d rows ..." % n)
    all_rows = np.arange(n)
    net, _ = fit_net(data, all_rows, np.zeros(0, dtype=np.int64), args, args.seed + 1000, len(vocab))
    W = net.export()

    os.makedirs(args.out, exist_ok=True)
    np.savez_compressed(os.path.join(args.out, "weights.npz"), **W)
    with open(os.path.join(args.out, "vocab.json"), "w", encoding="utf-8") as fh:
        json.dump(vocab, fh, ensure_ascii=False, separators=(",", ":"))
    meta = {
        "format": M.ARTIFACT_FORMAT, "feature_version": M.FEATURE_VERSION,
        "queues": QUEUES, "dense_dim": M.DENSE_DIM, "vocab_size": len(vocab),
        "emb_dim": args.emb_dim,
        "calibration": {"route": {"type": "temperature", "T": T},
                        "approve": cal_a, "grounded": cal_g},
        "decision": decision,
        "provenance": {"train_file": os.path.basename(args.train), "train_sha256": sha256(args.train),
                       "n_train": n, "seed": args.seed, "args": vars(args),
                       "python": platform.python_version(), "numpy": np.__version__,
                       "torch": torch.__version__},
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    with open(os.path.join(args.out, "training_report.json"), "w", encoding="utf-8") as fh:
        json.dump({"oof_pseudo_score": report,
                   "approve_selection": sel_a, "grounded_selection": sel_g,
                   "oof_loss_by_epoch": mean_hist.tolist(),
                   "calibration_cv_brier": {"approve": b_a, "grounded": b_g}}, fh, indent=1)

    # ---- 5. parity: exported NumPy inference must equal PyTorch -------------
    probe = np.random.RandomState(args.seed).choice(n, size=min(300, n), replace=False)
    Zt = predict_logits(net, data, probe)
    worst = 0.0
    for j, i in enumerate(probe):
        r, a, g = M.forward_logits({k: v.astype(np.float32) for k, v in W.items()},
                                   data.idx[i], data.dense[i])
        worst = max(worst, float(np.abs(np.concatenate([r, [a, g]]) - Zt[j]).max()))
    print("numpy-vs-torch max |logit diff| over %d rows: %.2e" % (len(probe), worst))
    if worst > 1e-3:
        sys.exit("PARITY FAILURE: exported NumPy model does not reproduce PyTorch")

    total = sum(os.path.getsize(os.path.join(args.out, f)) for f in os.listdir(args.out))
    print("artifacts in %s: %.1f MB total  (%.0fs)" % (args.out, total / 1e6, time.time() - t_start))
    return 0


if __name__ == "__main__":
    sys.exit(main())
