# Model Card: Decide Decision Engine

---

## 1. Model Details

- **Model Name:** Decide Multi-Task Customer Support Decision Engine
- **Model Version:** Candidate B (Production Deployment)
- **Model Type:** Multi-task deep neural network with pure NumPy CPU execution
- **Developers:** Decide Hackathon Team
- **Release Date:** October 2026
- **License:** MIT License
- **Inference Runtime:** Pure NumPy (`numpy>=1.26`, `fastapi>=0.115`, `uvicorn>=0.32`) — zero PyTorch or ONNX dependencies at serving time.
- **Artifact Footprint:** 4.47 MB total (`weights.npz` 4.24 MB, `vocab.json` 205 KB, `meta.json` 17 KB) — well under the 500 MB hard limit.

### Architecture Specification
- **Shared Input Representation:**
  - **Sparse Bag-of-Words / Subwords:** Hash-indexed unigrams, word bigrams (`s2:`, `b2:`), sentence patterns (`bs:`), and lexical overlap indicators between draft replies and knowledge-base articles (`dm:`, `km:`, `db:`, `dn:`, `dnu:`, `dq:`). Vocabulary size: 7,994 tokens.
  - **Sparse Embedding Layer:** `EmbeddingBag` sum-pooling into a 128-dimensional dense vector.
  - **Dense Feature Block (104 dimensions):** Customer tier one-hot encoding (`free`, `pro`, `enterprise`, unknown), continuous log-scaled amount ($\frac{\log(1+\text{amount})}{8.0}$), zero-amount indicator, 19 one-hot dollar threshold bins, tier $\times$ bin cross-terms, text length metrics, uppercase ratios, and draft/article overlap percentages.
- **Shared Trunk (Trunk forward pass executed exactly once per state):**
  - Layer 1: Linear ($232 \to 256$) + ReLU activation + LayerNorm.
  - Layer 2: Linear ($256 \to 128$) + ReLU activation + LayerNorm.
- **Specialized Decision Heads (64-dimensional hidden layer each):**
  - **`route` Head:** Linear ($128 \to 64$) + ReLU $\to$ Linear ($64 \to 5$) output logits across 5 queues (`billing`, `technical`, `account`, `shipping`, `general`).
  - **`approve` Head:** Linear ($128 \to 64$) + ReLU $\to$ Linear ($64 \to 1$) output logit for refund/action auto-approval.
  - **`grounded` Head:** Linear ($128 \to 64$) + ReLU $\to$ Linear ($64 \to 1$) output logit for drafted reply factual grounding.

---

## 2. Intended Use & Constraints

- **Primary Intended Use:** Sub-10ms synchronous decision automation for customer support tickets on standard 4-core CPU hardware without GPU acceleration.
- **Out of Scope & Prohibited Uses:**
  - Token-by-token generative text synthesis or freeform dialogue.
  - External network calls, hosted embedding lookups, or third-party LLM APIs in the request loop.
  - Processing non-text media (audio, images, attachments).

---

## 3. Training Data & Methodology

- **Training Corpus:** `train.jsonl` (6,500 labeled enterprise support records with queue assignments, approval actions, customer tiers, refund amounts, drafted agent responses, and knowledge base reference texts).
- **Evaluation Set:** `dev.jsonl` (1,500 records; 4,500 evaluated decisions). Dev data was never opened or leaked during training or hyperparameter search.
- **Optimization Strategy:**
  - Joint multi-task training using AdamW optimizer ($\text{lr} = 0.0005$, $\text{weight\_decay} = 0.1$, batch size 128, 20 epochs).
  - Multi-task loss: $\mathcal{L} = 1.0 \cdot \mathcal{L}_{\text{route}} + 1.0 \cdot \mathcal{L}_{\text{approve}} + 1.0 \cdot \mathcal{L}_{\text{grounded}}$.
  - Stratified 5-fold cross-validation on train splits to compute out-of-fold (OOF) logits for unbiased threshold selection and calibration fitting.
- **Probability Calibration Protocol:**
  - **`route`:** One-vs-Rest (OvR) Isotonic Regression fitted over raw logits with softmax probability normalization. Prevents extreme overconfidence on noisy tickets while maintaining top-1 class ordering.
  - **`approve`:** Isotonic regression fitted over OOF margin scores.
  - **`grounded`:** Platt scaling (logistic sigmoid regression) fitted over margin distances.
- **Cost-Sensitive Decision Thresholding:**
  - **`approve`:** Configured decision threshold $\tau = 0.75$ (strictly derived from the 3:1 asymmetric cost penalty where wrongly approving costs $3.0$ vs $1.0$ for holding for human review).
  - **`grounded`:** Symmetric decision threshold $\tau = 0.50$.

---

## 4. Empirical Evaluation & Benchmark Results

The model was evaluated using the official `selfcheck.py` benchmark harness over all 1,500 dev records (4,500 sequential warm HTTP calls).

### Official Hackathon Score Breakdown

| Evaluation Dimension | Weight | Raw Score | Weighted Points | Status |
|---|:---:|:---:|:---:|:---:|
| **Decision Quality** | 40% | 0.8259 | **33.04** | Exceeds trivial policies on all 3 heads |
| **Probability Calibration** | 25% | 0.8635 | **21.59** | Mean ECE: 0.0128, Brier Skill: 0.7910 |
| **Latency Gate & Scale** | 20% | 1.0000 | **20.00** | Full marks (Client p95: 2.16 ms $\le$ 25 ms) |
| **Abstention & Error Ranking** | 15% | 0.5338 | **8.01** | Risk at 80% coverage: 5.67% / 7.00% / 0.33% |
| **OVERALL TOTAL** | **100%** | — | **82.63 / 100** | **PASS on all schema and latency gates** |

### Per-Decision Quality & Calibration Metrics

| Decision Type | Primary Metric | Accuracy / Cost | Coverage | ECE | Brier Score | Trivial Baseline |
|---|---|---|---|---|---|---|
| **`route`** (5 queues) | **0.9394** Macro-F1 | 90.73% Acc | 83.27% | 0.0122 | 0.1530 | 0.0848 Macro-F1 (majority) |
| **`approve`** (binary) | **0.2076** Cost Error | 0.9094 Macro-F1 | 84.47% | 0.0161 | 0.0846 | 0.3504 (Hold) / 1.9487 (Approve) |
| **`grounded`** (binary) | **0.9927** Macro-F1 | 99.47% Acc | 86.87% | 0.0102 | 0.0198 | 0.3816 Macro-F1 |

### Calibration Summary
- **Mean Expected Calibration Error (ECE):** **0.0128** (probabilities are accurate to within ~1.3 percentage points across confidence bins).
- **Brier Skill Score:** **0.7910** (outperforms climatological base-rate prior prediction by ~79.1%).

### Latency Performance (Single-request, persistent HTTP/1.1, client-measured)
- **`route`:** p50 = 0.99 ms, p95 = **2.18 ms**
- **`approve`:** p50 = 0.96 ms, p95 = **2.21 ms**
- **`grounded`:** p50 = 0.94 ms, p95 = **2.10 ms**
- **Mean Client p95:** **2.16 ms** (11.5x faster than the 25 ms full-marks threshold; 46x faster than the 100 ms hard gate).
- **Server Internal Processing p95:** **0.83 ms** (model execution + response serialization).

---

## 5. Abstention & Selective Prediction Policy

The engine implements calibrated selective prediction (`abstain: true`), passing low-certainty cases to human agents:

| Decision Head | Full-Coverage Error | Risk @ 80% Coverage | Declared Coverage | Abstention Policy & Threshold |
|---|:---:|:---:|:---:|---|
| **`route`** | 9.27% | **5.67%** | 83.27% | Abstain if $\max(P_{\text{route}}) < 0.902$ |
| **`approve`** | 10.80% | **7.00%** | 84.47% | Abstain if prediction risk score $> 0.496$ |
| **`grounded`** | 2.40% | **0.33%** | 86.87% | Abstain if $\text{confidence} < 0.953$ |

- **Grader Floor Compliance:** Grader scales quality down if coverage falls below 80%. All three heads comfortably exceed this target ($83.3\%$, $84.5\%$, $86.9\%$), maximizing quality retention while eliminating the most error-prone tail cases.

---

## 6. Known Failure Modes & Mitigations

1. **Category Ambiguity in Support Tickets:**
   - *Failure Mode:* Real-world tickets often touch multiple categories (e.g., requesting a refund for a billing error caused by a technical account lock).
   - *Mitigation:* The OvR Isotonic calibrator distributes probability mass across overlapping categories, triggering selective abstention rather than producing an overconfident guess.
2. **Latent Account State in Auto-Approvals:**
   - *Failure Mode:* A small percentage of customer approval decisions depend on unrecorded account transaction histories not provided in the ticket text.
   - *Mitigation:* The decision threshold is locked at $\tau = 0.75$, and cases near the decision boundary are flagged for human review.
3. **Draft Reply Omission on Terse KB Articles:**
   - *Failure Mode:* Extremely concise knowledge-base snippets can produce low unigram overlap even when semantically valid.
   - *Mitigation:* Quantity matchers (`dq:`, `kq:`, `dn:`) verify numeric amounts, dates, and units independently of general vocabulary.

---

## 7. Pass/Fail Gates Verification

- [x] **Schema Compliance:** 4,500 / 4,500 valid responses (100.00%), zero violations.
- [x] **Latency Hard Floor:** Mean p95 = 2.16 ms (hard gate: 100 ms) — **PASS**.
- [x] **Reachable & Live:** Passes automated keep-alive probes with zero connection drops.
- [x] **One Shared Backbone:** Single forward pass in NumPy executes shared trunk for all three heads.
- [x] **Artifact Size:** 4.47 MB (limit: 500 MB) — **PASS**.
- [x] **Model Card Present:** Fully documented submission model card.
