#!/usr/bin/env python3
"""
Out-of-fold logits for one (feature version, seed) on TRAIN ONLY, using the baseline hyper-parameters.
Folds depend only on the seed, so feature versions 1 and 2 are evaluated on identical folds (paired).

    python experiments/oof.py --features 2 --seed 13
"""
import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
sys.path.insert(0, ROOT)
import train as T  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", type=int, choices=(1, 2), required=True)
    ap.add_argument("--seed", type=int, required=True)
    a = ap.parse_args()
    args = T.default_args(features=a.features, seed=a.seed)
    T.setup_determinism(args)
    t0 = time.time()
    prob = T.load_problem(args)
    Z, fold, hist = T.run_oof(prob, args, verbose=False)
    os.makedirs("experiments/cache", exist_ok=True)
    path = "experiments/cache/oof_f%d_seed%d.npz" % (a.features, a.seed)
    np.savez(path, Z=Z, fold=fold, hist=hist, y_route=prob.y_route, y_app=prob.y_app, y_gr=prob.y_gr)
    print("features v%d seed %d: wrote %s (%.0fs); final-epoch OOF loss route/approve/grounded %.4f / %.4f / %.4f"
          % (a.features, a.seed, path, time.time() - t0, *hist[-1]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
