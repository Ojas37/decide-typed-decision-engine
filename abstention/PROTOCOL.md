# Abstention study: protocol (written BEFORE running any variant)

Baseline = promoted Candidate B (features v1, shared trunk, route = one-vs-rest isotonic, approve isotonic, grounded Platt).
Data: data/train.jsonl only, cross-fitted out-of-fold (OOF) predictions. dev.jsonl is not opened by anything in this directory.

Scored quantity: the same pseudo-score as `train.evaluate_oof` / selfcheck formulas (total = 40 quality + 25 calibration + 15 abstention + 20 latency;
latency assumed full and measured separately for any exported candidate).
Abstention axis per decision = 1 - risk@80 / full error, ranked by the confidence the API returns (route: max prob; approve: max(p, 1-p)).

Constraints every variant must keep: shared multi-task trunk unchanged (post-hoc stages operate only on the trunk's own outputs
and on inputs the service already has); probabilities stay calibrated (calibration axis may not drop); declared coverage >= 0.80
(we keep the 0.83 target); route probabilities finite / non-negative / sum to 1 and value == argmax(probs).

Procedure
1. Diagnose (why confident predictions are wrong; how much ranking headroom exists at all).
2. Screen a FIXED list of variants on seeds 13, 14, 15 (paired vs B on identical folds).
3. Pick at most ONE winner (highest mean paired delta) and re-evaluate it, unchanged, on FRESH seeds 16, 17, 18
   (new folds, new initialisations; OOF generated before screening results were known).
4. Acceptance (strict): paired delta total > +0.30 on EVERY one of the six seeds, with no calibration/quality regression beyond noise
   and no hard-constraint violation. Otherwise: no candidate is promoted; findings are reported.
5. Only then: export as a separate candidate directory (experiments/candidates/<name>/), verify serving, and report.
   The promoted artifacts/ are never modified by this study. Backup of the promoted state: backups/promoted_B/.
