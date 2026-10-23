# Abstention study (Route + Approve): result

Baseline: promoted Candidate B. Data: train.jsonl OOF only; dev.jsonl never opened. Protocol: `PROTOCOL.md` (pre-registered).
Strict bar: paired delta total > +0.30 on EVERY seed (screen 13/14/15, confirm 16/17/18). Outcome: **no candidate qualifies; nothing exported or promoted.**

## Why wrong predictions are confident (seed 13, B probabilities)
Route (full error 0.095, risk@80 0.060, part 0.367, AUROC(conf->correct) 0.688):
* 28% of errors are label flips on clear-subject rows (model follows the subject; conf 0.935 vs 0.940 on correct clear rows;
  160/173 at conf >= 0.90) -> unrankable by construction. Marker bodies (30% of errors, conf 0.84) and vague subjects (21%, conf 0.76)
  are already ranked low. Remaining "other" errors are mostly confident noise too.
* Headroom probes (GBM predicting correctness from calibrated probs, logit gaps, input flags, fold-safe subject-entropy encoding): AUROC 0.667-0.688,
  part 0.331-0.361: no better than the existing confidence. Ceiling if every non-noise error were ranked last: part 0.68 (not reachable with available information).
Approve (full error 0.109, risk@80 0.066, part 0.394):
* 69% of errors are in the approvable core (p >= 0.7; hold label is noise, in-core AUROC 0.54); the 80% cut already drops 529/584 soft-zone rows and the
  lowest-confidence core rows. Only amount/tier-ceiling information helps ranking (GBM probe: part 0.455, +0.31 pts at most, not deployable).

## Variants (paired vs B; all cross-fitted)
Screen (13/14/15) mean / min: R1 temperature -0.58/-0.68; R2 matrix scaling -0.55/-0.66; R3 own-logit confidence recal -0.03/-0.15;
R4 R3+input flags +0.11/+0.03; R5 blend -0.25/-0.30; R6 tie-break 0.00/-0.01; A1 amount/tier-ceiling calibration layer +0.30/+0.03;
A2 A1+other heads +0.20/-0.12; A3 tie-break +0.02/+0.01; A4 (reference only, separate GBM, not eligible) +0.58/+0.49.
Fresh seeds (16/17/18): A1 -0.06 mean (+0.01/-0.17/-0.02), abstention axis -0.012: does not reproduce. Others: R3 +0.05, R4 +0.09, R5 -0.20, R6 0.00.
(Deviation: confirm also ran the route variants because of a filter bug; nothing was selected from them.)
