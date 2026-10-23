# Decide: diagnostic findings (TRAIN-ONLY, out-of-fold)

Method: 5-fold OOF logits regenerated with the exact deployed hyper-parameters (`make_oof.py`; seed 13
reproduces the artifact's route temperature 1.1987 exactly). Seeds 14 and 15 give a noise band.
All calibrators are cross-fitted (never scored on their own fit). Metrics reuse `selfcheck.py`'s functions.
dev.jsonl was never read. No file in `artifacts/`, `app.py`, `selfcheck.py`, `model.py`, `train.py` or the data was changed.

Method validation: OOF total (latency assumed full) = 81.87 / 81.59 / 81.39 over three seeds (mean 81.62, std 0.24);
your reported public-dev score is 82.07. Differences below ~0.3 points are inside seed noise.

## Score headroom (seed 13)
| axis | value | points | headroom |
|---|---|---|---|
| quality (40%) | 0.822 (route 0.925 / approve 0.553 / grounded 0.990) | 32.9 | 7.1 (approve 6.0, route 1.0, grounded 0.1) |
| calibration (25%) | 0.844 (ECE 0.037/0.009/0.007; Brier skill 0.794/0.644/0.890) | 21.1 | 3.9 |
| abstention (15%) | 0.526 (route 0.367 / approve 0.394 / grounded 0.816) | 7.9 | 7.1 (mostly label-noise-limited) |
| latency (20%) | 1.0 | 20.0 | 0 |

## Route
* acc 0.9055, macro-F1 0.898. Worst class: general (F1 0.820). Largest flows: account<->technical (94+89), general<->billing (79+64).
* ~5.2% of rows with an unambiguous subject are labelled against the obvious route; the model follows the subject in 99% of those errors -> irreducible label noise (~26% of errors).
* 22% of rows contain discourse markers; their accuracy is 0.861 vs 0.918. Label = the issue after a PRIMARY marker ("Priority for me is:", "My real question is this:", "The part I need you to action is:", "The main thing I need resolved is:", "Most importantly:", "What I actually need help with is this:"); never the issue after a SECONDARY marker ("For background,", "Separately, and this is less urgent,", "On a related note,", "Less important, but while I am writing:", "Some context first:", "Also worth mentioning,"). Current sentence-level tokens glue marker+issue so this is not learnable.
* 334 rows have vague subject AND vague body ("Can someone help with my account please"): accuracy 0.647, 19% of errors, largely unrecoverable.
* Confidence: wrong mean 0.777 vs correct 0.903, AUROC(conf->correct) 0.690. 41% of errors are made at conf >= 0.90 (noise). Selective risk (full 0.0945): 70%: .0585, 75%: .0593, 80%: .0598, 85%: .0619, 90%: .0658, 95%: .0755.
* Calibration shape: mid bins underconfident by 6-11 pts, top bin overconfident by 1 pt.

## Approve
* Hold side is clean (p<0.1: 3233 rows, 0.8% approved). The approvable side (p>=0.7: 2683 rows) is 18.2% hold-labelled and that residual is unpredictable (5-fold LR AUROC 0.48; model's own p 0.54; flat across route, tier, amount, grounded label, folds) -> label noise. False approvals are 82-89% of cost at every sensible threshold.
* Full-coverage cost by threshold: 0.50: .278, 0.60: .267, 0.70: .2655, 0.73: .2632, 0.75: .2655, 0.80: .273, 0.84: .307; always-hold .377. Bowl is shallow and centred near the 0.75 theory value.
* With the deployed abstention band the quality curve is flat on 0.50-0.78, so the threshold is only identifiable on the no-abstention curve (done in train.py). Going to 0.80 costs -0.27 pts.
* Hard approved-amount maxima: free 31.1, pro 121.3, enterprise 499.5. Approval odds decay before the cap (free 0.75/0.49/0.24/0.09/0.00 at 15/20/25/30/32.5; pro 0.56 -> 0.30 over 95-115). The model's amount bins (5/10/15/20/25/30/40... ) are too coarse there. Most confident false holds are free $26-31 and pro $96-119.
* Most confident false approvals (p 0.92-0.96) are indistinguishable from approved rows (noise).

## Grounded
* 211 errors (3.25%): FP 131, FN 80. Deployed abstention catches 84% of them; answered error 0.63%.
* 29.5% of errors sit on rows with number-format gaps ("quarter of an hour" 13%, "N hundred and M" 12%, hyphenated numbers 9%, "thousand" 5%): those rows have 12.3% error vs 2.5% elsewhere. Also draft numerals flagged as "missing from KB" even though the unit-normalised span matches ("sixty days" vs "two months").

## Calibration comparison (cross-fitted; ECE15 / Brier / Brier skill)
| task | raw | temperature | Platt | isotonic |
|---|---|---|---|---|
| route (top-label; OvR for Platt/iso) | .0344/.1599/.796 | .0380/.1617/.794 (deployed) | .0362/.1617/.794 | **.0120/.1586/.798** |
| approve | .0458/.0882/.625 | .0340/.0859/.634 | .0307/.0848/.639 | **.0064/.0839/.643** (deployed) |
| grounded | .0180/.0274/.886 | .0111/.0270/.888 | .0082/.0264/.890 (deployed) | .0062/.0265/.890 |

## What-ifs (same OOF methodology, paired over 3 seeds where stated)
* Route OvR-isotonic calibration: +0.57 / +0.68 / +0.50 (robust).
* Approve with finer amount/tier-cap resolution (stacking diagnostic): +0.66 / +0.91 / +0.71.
* Grounded: oracle halving errors +0.31, removing all +1.59 (upper bounds).
* Route marker rows fixed (oracle, recalibrated): -0.30 (50%) / -0.57 (100%) with temperature; +0.48 / +0.43 with OvR-iso, i.e. about the same or slightly below OvR-iso alone (+0.57).
* Coverage target: 0.75 -1.83, 0.78 -0.49, 0.80 +0.01, 0.85 -0.23, 0.90 -0.78, 1.00 -2.07.
* Approve calibrators: Platt -0.35, temperature -0.38, raw -0.63 (isotonic is best). Approve abstain rule: max_risk -0.10, band-below-threshold -1.01.
* Isotonic tie-breaking: ~+0.03; not worth it.
