# Decide — Typed Decision Engine

One model. Three typed decisions. Calibrated probabilities. Sub-2ms p95 on CPU.

---

## Start the service (single command, clean 4-core Linux machine)

```bash
pip install -r requirements.txt && uvicorn app:app --host 0.0.0.0 --port 8000
```

Artifacts (`artifacts/weights.npz`, `artifacts/vocab.json`, `artifacts/meta.json`)
are committed to the repo — no download step needed. The service is ready to
accept requests as soon as uvicorn finishes loading.

Verify it is alive:

```bash
curl http://localhost:8000/health
# {"status":"ok","loaded":true}
```

Run the full benchmark against the dev split:

```bash
python selfcheck.py --url http://localhost:8000/v1/decide
```

---

## Where the sharing happens

> The problem statement requires this section. Here it is.

Every request — regardless of whether it asks for `route`, `approve`, or
`grounded` — is processed by **one shared forward pass** through one shared
set of weights. Nothing is routed to a separate model.

**In `model.py`:**

```
Featurizer.encode(state)
    → (sparse token indices, dense float32 vector)
        ↓
forward_logits(W, idx, dense)          # one call, one set of weights
    │
    ├─ EmbeddingBag lookup + sum-pool  → pooled (128-d)   ← shared
    ├─ concat(pooled, dense)
    ├─ Linear W1 (→256) + ReLU        ← shared trunk layer 1
    ├─ Linear W2 (→128) + ReLU        ← shared trunk layer 2
    │
    ├─ route head:    Linear Wr1 (→64) + ReLU + Linear Wr2 (→5)
    ├─ approve head:  Linear Wa1 (→64) + ReLU + Linear Wa2 (→1)
    └─ grounded head: Linear Wg1 (→64) + ReLU + Linear Wg2 (→1)
```

`forward_logits` returns all three outputs in a single pass. `DecisionModel`
caches that result, so `route(state)`, `approve(state)`, `grounded(state)` on
the same state each read the cached trunk output — the trunk runs exactly once.

The trunk weights (`emb`, `W1`, `b1`, `W2`, `b2`) are trained jointly on all
three labels simultaneously. Each head adds one private two-layer block
(`64-d hidden + output`). Adding a fourth decision type means adding one more
such block and one column to the training loop — the trunk and vocabulary do not
change.

---

## Pretrained components and licences

No external pretrained model is used. The embedding table and all weights are
trained from scratch on the provided `train.jsonl` dataset.

Libraries used at inference:

| Library | Version | Licence |
|---|---|---|
| NumPy | ≥ 1.26 | BSD 3-Clause |
| FastAPI | ≥ 0.115 | MIT |
| uvicorn | ≥ 0.32 | BSD 3-Clause |

Libraries used at training only (not needed to serve):

| Library | Version | Licence |
|---|---|---|
| PyTorch | ≥ 2.1 | BSD-style (Meta) |
| scikit-learn | ≥ 1.3 | BSD 3-Clause |

---

## Benchmark results (dev split — 1,500 records × 3 decisions)

Results file: `results.json` (produced by `selfcheck.py`).

| Axis | Score | Weight | Contribution |
|---|---|---|---|
| Quality | 0.8316 | 40% | 33.26 |
| Calibration | 0.8641 | 25% | 21.60 |
| Latency | 1.0000 | 20% | 20.00 |
| Abstention | 0.5106 | 15% | 7.66 |
| **Total** | | | **82.53 / 100** |

Per-decision latency (client p95, one request at a time, warm):

| Decision | p50 (ms) | p95 (ms) |
|---|---|---|
| route | 0.82 | 1.01 |
| approve | 0.81 | 1.02 |
| grounded | 0.81 | 1.01 |
| **mean p95** | | **1.01 ms** |

Hard gate: 100 ms. Full-marks threshold: 25 ms. Measured mean p95: **1.01 ms**.

Schema: 4,500 / 4,500 valid. Artifact size: 4.45 MB (limit: 500 MB).

---

## Architecture overview

```
state dict  →  Featurizer  →  (sparse token ids, dense float32 vector [104-d])
                                    ↓
                            Shared trunk (NumPy)
                        EmbeddingBag sum-pool (128-d)
                        concat dense
                        Linear(384→256) + ReLU        [W1, b1]
                        Linear(256→128) + ReLU        [W2, b2]
                                    ↓
            ┌───────────────────────┼──────────────────────┐
            ▼                       ▼                       ▼
    route head               approve head            grounded head
 Linear(128→64)+ReLU     Linear(128→64)+ReLU     Linear(128→64)+ReLU
 Linear(64→5)  softmax   Linear(64→1)  sigmoid   Linear(64→1)  sigmoid
 OvR isotonic calibr.    Platt scaling           Isotonic calibr.
 argmax → queue label    τ = 0.73 (3:1 cost)     τ = 0.50
```

Inference is pure NumPy — no PyTorch, no ONNX runtime, no compilation step.
Weights are loaded from `artifacts/weights.npz` at startup.

---

## The three decisions

**`route`** — one of five queues: `billing`, `technical`, `account`, `shipping`,
`general`. Returns a full probability distribution. Calibrated with one-vs-rest
isotonic regression; the model abstains when `max(probs) < 0.52`.

**`approve`** — yes/no auto-approval. Wrongly approving costs 3× wrongly
holding, so the decision threshold is 0.73, not 0.5. Returns `p_true` — the
calibrated probability that approval is safe. Abstains when the model's
confidence is below 0.79 (too close to the asymmetric break-even to act).

**`grounded`** — yes/no: is the draft reply supported by the KB article?
Driven by token-overlap, bigram-overlap, and quantity-agreement features between
`draft_reply` and `kb_article`. The easiest of the three (dev F1 0.994). Sets a
high abstention bar at 0.95 confidence — grounding errors are high-stakes.

---

## Training the model yourself

Training requires PyTorch and scikit-learn (`requirements-train.txt`).

```bash
pip install -r requirements-train.txt

# Place train.jsonl in data/ first, then:
python train.py \
    --train data/train.jsonl \
    --out   artifacts \
    --seed  13 \
    --route_cal ovr_isotonic
```

Full set of options used to produce the committed artifacts:

```bash
python train.py \
    --train data/train.jsonl --out artifacts --seed 13 \
    --folds 5 --epochs 20 --batch 128 --lr 0.0005 --wd 0.1 \
    --emb_dim 128 --hidden 256 --hidden2 128 --head_hidden 64 \
    --dropout 0.2 --token_dropout 0.3 --min_df 3 \
    --coverage 0.83 --route_cal ovr_isotonic
```

Training on CPU takes roughly 8–12 minutes for 20 epochs on 6,500 examples.
GPU is faster but not required. The script writes `weights.npz`, `vocab.json`,
`meta.json`, and `training_report.json` into `--out` and verifies that NumPy
inference matches PyTorch logits before exiting.

---

## Repo layout

```
app.py              HTTP layer (do not modify — schema is fixed)
model.py            Featurizer, forward pass, calibration, DecisionModel
train.py            Full training pipeline: featurize → OOF → calibrate → export
selfcheck.py        Scoring code (identical to what the judges run)
smoke_test.py       Offline contract + latency check (no service needed)
requirements.txt    Inference dependencies only (NumPy, FastAPI, uvicorn)
requirements-train.txt  Adds PyTorch and scikit-learn for training
artifacts/
    weights.npz     Trained weights (~4.1 MB)
    vocab.json      Token vocabulary (~200 KB)
    meta.json       Calibration parameters, decision thresholds, provenance
data/               Place train.jsonl and dev.jsonl here (gitignored)
tools/              Optional analysis scripts (not needed to serve or judge)
```

---

## Endpoint

```
POST /v1/decide
GET  /health
```

Request:

```json
{
  "request_id": "abc-123",
  "decision": "route",
  "state": {
    "subject": "...", "body": "...", "customer_tier": "pro",
    "amount": 49.99, "draft_reply": "...", "kb_article": "..."
  }
}
```

Route response:

```json
{
  "request_id": "abc-123", "decision": "route", "type": "choice",
  "value": "billing",
  "probs": {"billing": 0.82, "technical": 0.09, "account": 0.06,
            "shipping": 0.02, "general": 0.01},
  "confidence": 0.82, "abstain": false, "latency_ms": 0.9
}
```

Binary response (approve / grounded):

```json
{
  "request_id": "abc-123", "decision": "approve", "type": "binary",
  "value": false, "p_true": 0.13, "abstain": false, "latency_ms": 0.8
}
```
