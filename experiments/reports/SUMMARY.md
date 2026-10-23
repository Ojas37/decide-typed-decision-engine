# Experiments A / B: decision record

Protocol: baseline hyper-parameters, 5-fold OOF on train.jsonl, seeds 13/14/15, identical folds per seed (paired).
All numbers from `train.evaluate_oof()` (the same code that builds artifacts). Accept iff paired delta > 0 on all
3 seeds AND mean delta > 0.3 AND no hard constraint violated. dev.jsonl untouched until selection; then ONE selfcheck run.

| candidate | features | route calibration | paired delta total (13/14/15) | mean | verdict |
|---|---|---|---|---|---|
| baseline | v1 | temperature | - | OOF 81.66 | reference |
| A  | v2 (fine amount x tier) | temperature | -0.68 / -0.06 / -0.25 | -0.33 | reject |
| B  | v1 | one-vs-rest isotonic | +0.57 / +0.68 / +0.50 | +0.58 | ACCEPT |
| AB | v2 | one-vs-rest isotonic | -0.14 / +0.42 / +0.42 | +0.23 | reject (< 0.3, seed 13 negative) |

(OOF totals assume full latency marks; latency is measured separately: HTTP mean p95 baseline 1.67 / A 1.88 / B 1.83 / AB 2.43 ms.)

Selected: **B**. One dev selfcheck run: total 82.53 (quality 33.26, calibration 21.60, latency 20.00, abstention 7.66);
schema 4500/4500 valid; mean p95 2.36 ms. Public-dev score of the baseline reported earlier: 82.07.

Files: `comparison_oof.txt/json` (full tables), `candidate_checks.txt` (constraints), `selfcheck_dev_candidate_B.*`,
`baseline_parity.txt` / `baseline_regression.txt` (baseline reproduced bit-for-bit through the refactored code).

Restore baseline: see `backups/baseline_v1/RESTORE.md`. Promote B: copy `experiments/candidates/B/artifacts/*` into `artifacts/`
(requires the current model.py, which supports both artifact kinds).
