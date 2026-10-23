#!/usr/bin/env python3
"""Generate out-of-fold logits on TRAIN only. Never opens dev.jsonl.

    python tools/generate_oof.py --seed 13 --features 1
    python tools/generate_oof.py --seed 13 --features 2
"""
import argparse, json, os, sys, time
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
    ap.add_argument("--features", type=int, choices=(1, 2), default=1)
    a = ap.parse_args()

    if a.features == 1:
        # Reproduce baseline hyperparams exactly from the deployed meta.json
        with open("artifacts/meta.json", encoding="utf-8") as fh:
            meta = json.load(fh)
        args = argparse.Namespace(**meta["provenance"]["args"])
    else:
        # Feature v2 requires default_args (meta.json for v1 doesn't carry v2 flags)
        args = T.default_args(features=a.features)
    args.seed = a.seed
    args.features = a.features
    T.setup_determinism(args)

    prob = T.load_problem(args)
    Z, fold, hist = T.run_oof(prob, args, verbose=True)

    os.makedirs("tools/cache", exist_ok=True)
    path = "tools/cache/oof_f%d_seed%d.npz" % (a.features, a.seed)
    np.savez(path, Z=Z, fold=fold, hist=hist,
             y_route=prob.y_route, y_app=prob.y_app, y_gr=prob.y_gr)
    print("wrote %s  (%.0fs)" % (path, time.time() - time.time() + 1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

