#!/usr/bin/env python3
"""
Regenerate out-of-fold (OOF) logits on TRAIN ONLY, with the exact hyper-parameters
recorded in artifacts/meta.json (read-only). Writes diagnostics/cache/oof_seed<S>.npz.
Never opens dev.jsonl, never writes to artifacts/.

    python diagnostics/make_oof.py --seed 13
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import model as M          # noqa: E402
import train as T          # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=13)
    a = ap.parse_args()

    with open("artifacts/meta.json", encoding="utf-8") as fh:
        meta = json.load(fh)
    args = argparse.Namespace(**meta["provenance"]["args"])
    args.seed = a.seed
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True, warn_only=True)

    rows = T.load_jsonl("data/train.jsonl")          # TRAIN only
    n = len(rows)
    y_route = np.array([M.QUEUES.index(r["labels"]["route"]) for r in rows], dtype=np.int64)
    y_app = np.array([bool(r["labels"]["approve"]) for r in rows])
    y_gr = np.array([bool(r["labels"]["grounded"]) for r in rows])
    extracted = [M.Featurizer.extract(r["state"]) for r in rows]
    vocab = T.build_vocab([t for t, _ in extracted], args.min_df)
    idx = [np.fromiter((vocab[t] for t in toks if t in vocab), dtype=np.int64) for toks, _ in extracted]
    dense = np.stack([d for _, d in extracted]).astype(np.float32)
    data = T.Data(idx, dense, y_route, y_app, y_gr)

    Z = np.zeros((n, 7))
    fold = np.zeros(n, dtype=np.int64)
    t0 = time.time()
    for k, (tr, va) in enumerate(StratifiedKFold(args.folds, shuffle=True, random_state=args.seed)
                                 .split(np.zeros(n), y_route)):
        net, _ = T.fit_net(data, tr, va, args, args.seed + k, len(vocab))
        Z[va] = T.predict_logits(net, data, va)
        fold[va] = k
        print("seed %d fold %d done (%.0fs)" % (args.seed, k + 1, time.time() - t0), flush=True)

    os.makedirs("diagnostics/cache", exist_ok=True)
    np.savez("diagnostics/cache/oof_seed%d.npz" % args.seed, Z=Z, y_route=y_route,
             y_app=y_app, y_gr=y_gr, fold=fold)
    T_est = T.fit_temperature(Z[:, :5], y_route)
    print("seed %d: OOF route temperature %.4f (artifact meta: %.4f)"
          % (args.seed, T_est, meta["calibration"]["route"]["T"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
