# Model Card: Decide Decision Engine

## Model Details
- **Model Name:** Decide Multi-Task Support Decision Engine (Candidate B)
- **Architecture:** Shared-backbone multi-task neural network with NumPy CPU inference engine
  - **Shared Representation:** Bag-of-words / subword hash embeddings (dimension 128, vocab 7,994) concatenated with 104 engineered dense features (character lengths, token counts, uppercase ratios, ticket amount bins, customer tier one-hot encodings, draft/article overlap lexical features).
  - **Trunk:** 2-layer MLP backbone (Dense 256 -> ReLU -> LayerNorm -> Dense 128 -> ReLU -> LayerNorm).
  - **Decision Heads:** Three specialized classification heads (64 hidden units each):
    - `route`: 5-class distribution (`billing`, `technical`, `account`, `shipping`, `general`).
    - `approve`: Binary logit for auto-approval decision under asymmetric cost.
    - `grounded`: Binary logit for citation / factual support verification.
- **Serving Runtime:** Pure NumPy inference with zero torch dependency at runtime.

---

## Intended Use
- **Primary Domain:** High-throughput, sub-10ms inbound customer support automation running on 4-core CPU machines.
- **Out of Scope:** Multi-turn conversational generation, open-ended freeform answering, external API calling.

---

## Training Data & Methodology
- **Training Corpus:** `train.jsonl` (6,500 labeled customer support requests across 5 queues with approval status, ground-truth grounding labels, customer tier, refund amounts, draft replies, and knowledge base snippets).
- **Validation:** 5-fold cross-fitting on train out-of-fold (OOF) predictions (`seed=13`, `seed=14`, `seed=15`) across all architectural iterations; zero leakage from `dev.jsonl`.
- **Calibration Protocol:**
  - `route`: One-vs-Rest (OvR) Isotonic Regression over logits with softmax normalization, eliminating overconfidence while preserving rank-order.
  - `approve` / `grounded`: Platt scaling (logistic sigmoid calibration) tuned on cross-validated OOF margins.
- **Asymmetric Cost Thresholding:**
  - `approve`: Configured decision threshold `tau = 0.73` to account for 3:1 false-positive cost ratio ($C_{FP} = 3.0$ vs $C_{FN} = 1.0$).
  - `grounded`: Decision threshold `tau = 0.50`.

---

## Evaluation Results (`data/dev.jsonl`, 1,500 rows, 4,500 calls)

| Evaluation Metric | Measured Value | Benchmark / Target |
|---|---|---|
| **TOTAL SCORE** | **82.53** | Baseline ~75.0 |
| **Quality Score** | **0.8316** | Weight 40% |
| **Calibration Score** | **0.8641** | Weight 25% |
| **Latency Score** | **1.0000** | Weight 20% (Full score < 25ms) |
| **Abstention Score** | **0.5106** | Weight 15% |

### Per-Decision Metrics
- **Routing (`route`):**
  - Macro-F1: **0.9386**
  - Accuracy: **0.9140** (vs trivial majority 0.0842)
  - ECE: **0.0153**
  - Brier score: **0.1501**
- **Approval (`approve`):**
  - Cost-weighted error: **0.2144** (vs always-hold 0.3758, always-approve 1.8726)
  - Macro-F1: **0.9239**
  - ECE: **0.0129**
  - Brier score: **0.0846**
- **Grounding (`grounded`):**
  - Macro-F1: **0.9936** (vs trivial 0.3796)
  - ECE: **0.0105**
  - Brier score: **0.0194**

### Overall Calibration & Latency
- **Mean ECE:** **0.0129**
- **Brier Skill Score:** **0.7928**
- **Latency (Client p95):**
  - `route`: 1.01 ms
  - `approve`: 1.02 ms
  - `grounded`: 1.01 ms
  - **Mean p95:** **1.01 ms** (Well below the 25.0 ms ceiling for full marks; ~100x faster than the 100 ms gate)
- **Model-Only Latency (Excluding HTTP):** p95 ~0.48 ms

---

## Abstention & Selective Prediction
The model declines low-certainty requests by returning `abstain: true`, routing them to human specialists:
- **Coverage Target:** Configured for >= 80% coverage on every head.
  - Route coverage: **83.47%**
  - Approve coverage: **84.27%**
  - Grounded coverage: **87.40%**
- **Risk at 80% Coverage:**
  - Route: **0.0567** (vs full-coverage error 0.0860)
  - Approve: **0.0692** (vs full-coverage error 0.1127)
  - Grounded: **0.0042** (vs full-coverage error 0.0213)
- **Mechanisms:**
  - `route`: Minimum top probability threshold ($\tau_{conf} = 0.784$).
  - `approve`: Minimum prediction confidence distance from threshold ($\tau_{conf} = 0.787$).
  - `grounded`: Minimum prediction confidence ($\tau_{conf} = 0.953$).

---

## Failure Modes & Mitigations
- **Ambiguous & Label-Flipped Tickets:** Approximately 28% of routing residual errors originate from inherently noisy support desk category conventions (e.g., billing inquiries discussing account reactivation). The OvR isotonic calibrator dampens confidence on these ambiguous texts, triggering selective abstention.
- **Unobservable Account History in Approvals:** Under-the-hood policies that hinge on unrecorded customer dispute history cannot be deduced purely from state payload; the calibrated threshold $\tau=0.73$ conservatively holds requests without high certainty.
- **Short or Truncated Draft Replies:** Lexical overlap drops on extremely short sentences; dense feature engineering normalizes character length and token ratios to stabilize logit scales.

