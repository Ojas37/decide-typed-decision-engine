#!/usr/bin/env python3
"""calibrate.py — temperature scaling, cost threshold, abstention thresholds."""

import argparse, json, os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from train import MultiTaskModel, DecideDataset, collate, QUEUES, QUEUE2ID


def ece(probs, labels, n_bins=15):
    """Expected calibration error over equal-width bins."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    bins = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (probs > lo) & (probs <= hi) if lo > 0 else (probs >= lo) & (probs <= hi)
        if m.sum() == 0:
            continue
        total += abs(probs[m].mean() - labels[m].mean()) * (m.sum() / len(probs))
    return total


@torch.no_grad()
def collect(model, dl, device):
    """Return per-task (logits, labels, binary_probability, confidence)."""
    model.eval()
    out = {t: {"logits": [], "labels": [], "probs": [], "conf": []}
           for t in ["route", "approve", "grounded"]}
    for batch in dl:
        ids = batch["input_ids"].to(device)
        am = batch["attention_mask"].to(device)
        logits = model(ids, am)
        labels = batch["labels"].cpu().numpy()
        for i, task in enumerate(batch["tasks"]):
            if task == "route":
                z = logits["route_logits"][i].cpu().numpy()
                out[task]["logits"].append(z)
                out[task]["labels"].append(labels[i])
                p = torch.softmax(torch.tensor(z), -1).numpy()
                out[task]["probs"].append(p)
                out[task]["conf"].append(float(p.max()))
            else:
                key = "approve_logit" if task == "approve" else "grounded_logit"
                z = float(logits[key][i].cpu())
                out[task]["logits"].append(z)
                out[task]["labels"].append(labels[i])
                p = float(torch.sigmoid(torch.tensor(z)))
                out[task]["probs"].append(p)
                out[task]["conf"].append(max(p, 1 - p))
    for t in out:
        for k in out[t]:
            out[t][k] = np.array(out[t][k])
    return out


def fit_temperature(logits, labels, is_multiclass):
    """Grid-search T minimising NLL."""
    z = torch.tensor(logits, dtype=torch.float32)
    y = torch.tensor(labels, dtype=torch.long)
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(0.5, 5.0, 91):
        scaled = z / t
        if is_multiclass:
            nll = F.cross_entropy(scaled, y).item()
        else:
            nll = F.binary_cross_entropy_with_logits(
                scaled.squeeze(-1), y.float()).item()
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t, best_nll


def fit_approve_threshold(probs, labels, fp_cost=3.0):
    """Search threshold minimising 3:1 cost."""
    best_thr, best_cost = 0.5, float("inf")
    for thr in np.linspace(0.05, 0.95, 181):
        pred = probs > thr
        cost = np.mean((pred & (labels == 0)) * fp_cost +
                       (~pred & (labels == 1)) * 1.0)
        if cost < best_cost:
            best_cost, best_thr = cost, float(thr)
    return best_thr, best_cost


def abstain_threshold(conf, target_coverage=0.80):
    """Threshold so that the top `target_coverage` fraction is answered."""
    return float(np.quantile(conf, 1.0 - target_coverage))


def main(args):
    tok = AutoTokenizer.from_pretrained(os.path.join(args.ckpt, "tokenizer"))
    cfg = json.load(open(os.path.join(args.ckpt, "config.json")))
    model = MultiTaskModel(cfg["backbone"])
    model.load_state_dict(torch.load(os.path.join(args.ckpt, "model.pt"),
                                     map_location="cpu"))
    device = torch.device("cpu")
    model.to(device)

    ds = DecideDataset(os.path.join(args.data_dir, "dev.jsonl"), tok,
                       max_len=cfg["max_len"])
    dl = DataLoader(ds, batch_size=32, shuffle=False,
                    collate_fn=lambda b: collate(b, tok))
    out = collect(model, dl, device)

    calib = {"backbone": cfg["backbone"], "max_len": cfg["max_len"],
             "queues": QUEUES, "temperatures": {},
             "abstain_thresholds": {}, "target_coverage": 0.80}

    # ---- route (multiclass) ----
    t_route, _ = fit_temperature(out["route"]["logits"],
                                 out["route"]["labels"], True)
    calib["temperatures"]["route"] = t_route
    probs = torch.softmax(
        torch.tensor(out["route"]["logits"]) / t_route, -1).numpy()
    conf = probs.max(1)
    pred = probs.argmax(1)
    print(f"route    T={t_route:.3f}  ECE before={ece(out['route']['conf'], pred==out['route']['labels']):.4f}  "
          f"after={ece(conf, pred==out['route']['labels']):.4f}")
    calib["abstain_thresholds"]["route"] = abstain_threshold(conf)

    # ---- approve (binary, cost-sensitive) ----
    t_app, _ = fit_temperature(out["approve"]["logits"],
                               out["approve"]["labels"], False)
    calib["temperatures"]["approve"] = t_app
    p = torch.sigmoid(
        torch.tensor(out["approve"]["logits"]) / t_app).numpy()
    thr, cost = fit_approve_threshold(p, out["approve"]["labels"])
    calib["approve_threshold"] = thr
    print(f"approve  T={t_app:.3f}  threshold={thr:.3f}  cost={cost:.4f}")
    conf = np.maximum(p, 1 - p)
    corr = (p > 0.5) == (out["approve"]["labels"] > 0.5)
    print(f"         ECE after={ece(conf, corr):.4f}")
    calib["abstain_thresholds"]["approve"] = abstain_threshold(conf)

    # ---- grounded (binary) ----
    t_gr, _ = fit_temperature(out["grounded"]["logits"],
                              out["grounded"]["labels"], False)
    calib["temperatures"]["grounded"] = t_gr
    p = torch.sigmoid(
        torch.tensor(out["grounded"]["logits"]) / t_gr).numpy()
    conf = np.maximum(p, 1 - p)
    corr = (p > 0.5) == (out["grounded"]["labels"] > 0.5)
    print(f"grounded T={t_gr:.3f}  ECE after={ece(conf, corr):.4f}")
    calib["abstain_thresholds"]["grounded"] = abstain_threshold(conf)

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "calibration.json"), "w") as f:
        json.dump(calib, f, indent=2)
    print(f"\nwrote {os.path.join(args.out, 'calibration.json')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out", default="artifacts")
    main(ap.parse_args())