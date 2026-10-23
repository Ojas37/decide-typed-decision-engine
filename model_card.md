# Model Card — Decide

## What I built
One shared `distilbert-base-uncased` encoder with three heads:
- **route**: 5-way linear head over the `[CLS]` representation of `subject + body`
- **approve**: binary head over `[CLS]` of `subject + body + tier + amount`
- **grounded**: binary head over the `[CLS]` of `[CLS] draft [SEP] kb [SEP]`

The encoder is the same for all three decisions. Only the head output read by
`model.py` changes. Training is single-task per batch; the deployed artifact is
one ONNX graph containing all three heads.

## What I trained on
`train.jsonl` provided at kickoff (~8k rows, 3 labels per row). No external
data, no hosted API, no generated data.

## Measured accuracy and calibration
(Fill in from `results.json` on the full dev split.)
- route macro-F1:
- approve cost-weighted error (3:1):
- grounded macro-F1:
- mean ECE after temperature scaling:
- Brier skill:

## Where it fails
Routing labels are noisy by design; the model is not near-certain on ambiguous
tickets and its probabilities reflect that. Approval depends partly on account
history that is not in the ticket, so a ceiling below 100% is expected. Grounded
fails when the draft paraphrases the KB so heavily that lexical overlap is low.

## When it abstains
The abstain threshold is chosen on dev so that declared coverage stays at or
above 80%. Abstention is based on `max(p, 1-p)` for binary decisions and on the
top probability for routing. The flag removes the case from quality scoring;
the abstention score still reads the reported probabilities.