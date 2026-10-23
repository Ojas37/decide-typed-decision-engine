"""
Decide model: one shared multi-task backbone, three task-specific heads.

Where the sharing happens
-------------------------
Every request is turned into ONE input (sparse text features + dense numeric
features, see `Featurizer`) and pushed through ONE shared trunk:

    embedding-bag over sparse tokens  ->  concat(dense features)
        -> Linear+ReLU -> Linear+ReLU            (shared trunk, 128-d)
            |-> route head     (5-way softmax)
            |-> approve head   (binary logit)
            |-> grounded head  (binary logit)

The trunk is trained jointly on all three labels (see train.py). Each head only
adds a small private layer; the representation they read is the same one. A
request's trunk forward pass is computed once and cached, so route/approve/
grounded for the same state reuse it.

Inference is pure NumPy (no PyTorch, no network). Artifacts live in
`artifacts/`: weights.npz, vocab.json, meta.json (see train.py for how they are
produced). Featurization, calibration and abstention helpers below are imported
by train.py so training and serving cannot drift apart.
"""

import os

# Matrices here are tiny; multi-threaded BLAS only adds latency and jitter.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import logging
import math
import re
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

QUEUES: List[str] = ["billing", "technical", "account", "shipping", "general"]

logger = logging.getLogger("decide.model")

FP_COST = 3.0                 # wrongly approving costs 3x wrongly holding
P_EPS = 0.002                 # probabilities are kept inside [P_EPS, 1-P_EPS]
FEATURE_VERSION = 1                       # default for new training runs; artifacts carry their own
SUPPORTED_FEATURE_VERSIONS = (1, 2)
ARTIFACT_FORMAT = 1
TIERS = ["free", "pro", "enterprise"]
AMOUNT_EDGES = [0.001, 5, 10, 15, 20, 25, 30, 40, 50, 75, 100, 125, 150,
                200, 300, 500, 750, 1000]
N_AMOUNT_BINS = len(AMOUNT_EDGES) + 1

# Feature version 2 adds, to the version-1 dense vector, a fine-grained amount block:
# amount as a fraction of its tier's approval ceiling (learned from train, stored in
# meta.json), expanded with triangular "hat" functions every 0.05 of the ceiling.
HAT_KNOTS = np.linspace(0.0, 1.3, 27)
HAT_WIDTH = 0.05
N_HATS = len(HAT_KNOTS)
V2_DIM = len(TIERS) * N_HATS + len(TIERS) * 3 + 3


# ---------------------------------------------------------------------------
# text normalisation
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"\d+(?:\.\d+)?|[a-z]+")
_MONEY_RE = re.compile(r"\$\s?\d[\d,]*(?:\.\d+)?")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")

_UNITS = {w: i for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve "
    "thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60,
         "seventy": 70, "eighty": 80, "ninety": 90}
_TIME_MIN = {"minute": 1, "hour": 60, "day": 1440, "week": 10080,
             "month": 43200, "year": 525600}


def _as_str(v: Any) -> str:
    if v is None:
        return ""
    return v if isinstance(v, str) else str(v)


def _as_amount(v: Any) -> float:
    try:
        a = float(v)
    except (TypeError, ValueError):
        return 0.0
    if a != a or a in (float("inf"), float("-inf")):
        return 0.0
    return max(a, 0.0)


def _tokens(text: str, mask_numbers: bool) -> List[str]:
    t = _MONEY_RE.sub(" moneyamt ", text.lower())
    toks = _TOKEN_RE.findall(t)
    if mask_numbers:
        toks = ["<n>" if (x[0].isdigit() and len(x) > 2) else x for x in toks]
    return toks


def _merge_numbers(toks: Sequence[str]) -> List[str]:
    """'sixty' -> '60', 'one hundred eighty' -> '180'; everything else unchanged."""
    out: List[str] = []
    i, n = 0, len(toks)
    while i < n:
        t = toks[i]
        if t in _UNITS or t in _TENS or t == "hundred":
            cur, j = 0, i
            while j < n:
                w = toks[j]
                if w in _UNITS:
                    cur += _UNITS[w]
                elif w in _TENS:
                    cur += _TENS[w]
                elif w == "hundred":
                    cur = (cur or 1) * 100
                else:
                    break
                j += 1
            out.append(str(cur))
            i = j
        else:
            out.append(t)
            i += 1
    return out


def _stem(w: str) -> str:
    return w[:-1] if len(w) > 3 and w.endswith("s") and not w.endswith("ss") else w


def _quantities(toks: Sequence[str]) -> Tuple[set, set, set]:
    """
    Returns (numerals, numeral+following-word pairs, canonical time spans).
    Time spans are normalised to minutes so '2 weeks' == '14 days' and
    'half an hour' == '30 minutes'; business days are kept apart.
    """
    nums, numu, qty = set(), set(), set()
    n = len(toks)
    for i, t in enumerate(toks):
        if t[0].isdigit():
            nums.add(t)
            if i + 1 < n:
                numu.add(t + "_" + _stem(toks[i + 1]))
        u = _TIME_MIN.get(_stem(t))
        if u is None or i == 0:
            continue
        prev = toks[i - 1]
        if prev == "business":
            if i > 1 and toks[i - 2][0].isdigit():
                qty.add("bd" + toks[i - 2])
            continue
        if prev[0].isdigit():
            val = float(prev)
        elif prev in ("a", "an"):
            val = 0.5 if (i > 1 and toks[i - 2] == "half") else 1.0
        elif prev == "half":
            val = 0.5
        else:
            continue
        qty.add("%g" % (val * u))
    return nums, numu, qty


# ---------------------------------------------------------------------------
# featurizer (shared by train.py and inference)
# ---------------------------------------------------------------------------

class Featurizer:
    """
    state -> (sparse token strings, dense float32 vector).

    Sparse tokens (looked up in the trained vocabulary):
        s:/s2:/sf:   subject unigrams, bigrams, whole subject
        b:/b2:/bs:   body unigrams, bigrams, whole templated sentences
        dm:/km:      words only in the draft / only in the KB article
        db:          draft bigrams absent from the KB article
        dn:/kn:      numerals only in draft / only in KB (number words folded in)
        dnu:/knu:    same, with the following word ('14_day')
        dq:/kq:      time spans only in draft / only in KB (unit-normalised)
    Dense: customer tier, amount (log, zero flag, bins, tier x bins) and
    draft-vs-KB overlap / quantity-agreement statistics.
    """

    def __init__(self, vocab: Optional[Dict[str, int]] = None, version: int = 1,
                 tier_caps: Optional[Dict[str, float]] = None) -> None:
        if version not in SUPPORTED_FEATURE_VERSIONS:
            raise ValueError("unsupported feature version %r" % version)
        if version == 2 and not tier_caps:
            raise ValueError("feature version 2 needs tier_caps")
        self.vocab = vocab
        self.version = version
        self.tier_caps = tier_caps

    @staticmethod
    def extract(state: Dict[str, Any]) -> Tuple[List[str], np.ndarray]:
        subject = _as_str(state.get("subject"))
        body = _as_str(state.get("body"))
        tier = _as_str(state.get("customer_tier")).strip().lower()
        draft = _as_str(state.get("draft_reply"))
        kb = _as_str(state.get("kb_article"))
        amount = _as_amount(state.get("amount"))

        toks = set()

        st = _tokens(subject, True)
        toks.update("s:" + t for t in st)
        toks.update("s2:%s_%s" % p for p in zip(st, st[1:]))
        if st:
            toks.add("sf:" + " ".join(st))

        bt = _tokens(body, True)
        toks.update("b:" + t for t in bt)
        toks.update("b2:%s_%s" % p for p in zip(bt, bt[1:]))
        for sent in _SENT_RE.split(_MONEY_RE.sub(" moneyamt ", body.lower())):
            s = _tokens(sent, True)
            if 1 <= len(s) <= 30:
                toks.add("bs:" + " ".join(s))

        dt = _merge_numbers(_tokens(draft, False))
        kt = _merge_numbers(_tokens(kb, False))
        ds, ks = set(dt), set(kt)
        d_bi, k_bi = set(zip(dt, dt[1:])), set(zip(kt, kt[1:]))
        toks.update("dm:" + t for t in ds - ks)
        toks.update("km:" + t for t in ks - ds)
        toks.update("db:%s_%s" % p for p in d_bi - k_bi)

        dn, dnu, dq = _quantities(dt)
        kn, knu, kq = _quantities(kt)
        toks.update("dn:" + t for t in dn - kn)
        toks.update("kn:" + t for t in kn - dn)
        toks.update("dnu:" + t for t in dnu - knu)
        toks.update("knu:" + t for t in knu - dnu)
        toks.update("dq:" + t for t in dq - kq)
        toks.update("kq:" + t for t in kq - dq)

        # ---- dense ---------------------------------------------------------
        tier_oh = [1.0 if tier == t else 0.0 for t in TIERS]
        tier_oh.append(0.0 if tier in TIERS else 1.0)
        b = 0
        while b < len(AMOUNT_EDGES) and amount >= AMOUNT_EDGES[b]:
            b += 1
        bin_oh = [0.0] * N_AMOUNT_BINS
        bin_oh[b] = 1.0
        cross = [0.0] * (len(TIERS) * N_AMOUNT_BINS)
        if tier in TIERS:
            cross[TIERS.index(tier) * N_AMOUNT_BINS + b] = 1.0

        inter = len(ds & ks)
        union = len(ds | ks)
        d_n, k_n = len(ds), len(ks)
        pair = [
            inter / union if union else 0.0,
            inter / d_n if d_n else 0.0,
            inter / k_n if k_n else 0.0,
            len(d_bi & k_bi) / len(d_bi) if d_bi else 0.0,
            len(dn) / 5.0, len(dn & kn) / 5.0, len(dn - kn) / 5.0, len(kn - dn) / 5.0,
            len(dq) / 3.0, len(dq & kq) / 3.0, len(dq - kq) / 3.0, len(kq - dq) / 3.0,
            1.0 if dq else 0.0,
            1.0 if (dq and dq <= kq) else 0.0,
            1.0 if (dn and dn <= kn) else 0.0,
            len(dt) / 40.0, len(kt) / 40.0, len(bt) / 80.0,
            1.0 if not draft.strip() else 0.0,
            1.0 if not kb.strip() else 0.0,
            1.0 if not body.strip() else 0.0,
            1.0 if not subject.strip() else 0.0,
        ]
        dense = np.asarray(
            tier_oh + [math.log1p(amount) / 8.0, 1.0 if amount == 0.0 else 0.0]
            + bin_oh + cross + pair, dtype=np.float32)
        return sorted(toks), dense

    def encode(self, state: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        toks, dense = self.extract(state)
        if self.version == 2:
            tier = _as_str(state.get("customer_tier")).strip().lower()
            block = amount_block_v2([_as_amount(state.get("amount"))], [tier], self.tier_caps)[0]
            dense = np.concatenate([dense, block])
        vocab = self.vocab or {}
        idx = np.fromiter((vocab[t] for t in toks if t in vocab), dtype=np.int64)
        return idx, dense


DENSE_DIM = int(Featurizer.extract({})[1].shape[0])          # version-1 dense width


def dense_dim(version: int) -> int:
    return DENSE_DIM + (V2_DIM if version == 2 else 0)


def amount_block_v2(amounts: Sequence[float], tiers: Sequence[str],
                    tier_caps: Dict[str, float]) -> np.ndarray:
    """
    Fine amount features, one row per request.

    r = amount / (tier's approval ceiling), clipped to [0, 2]. Per tier (so each
    tier can learn its own decay curve): 27 triangular hats over r in [0, 1.3],
    plus [r, r^2, beyond-ceiling flag]. Then the same three terms globally.
    Zero amounts (non-monetary requests) get no hats: the v1 zero flag covers them.
    """
    a = np.asarray(amounts, dtype=np.float64)
    tidx = np.array([TIERS.index(t) if t in TIERS else -1 for t in tiers])
    default_cap = float(np.median([tier_caps[t] for t in TIERS]))
    cap = np.array([tier_caps[TIERS[j]] if j >= 0 else default_cap for j in tidx])
    r = np.minimum(a / np.maximum(cap, 1e-6), 2.0)
    pos = (a > 0).astype(np.float64)
    hats = np.maximum(0.0, 1.0 - np.abs(r[:, None] - HAT_KNOTS[None, :]) / HAT_WIDTH) * pos[:, None]
    lin = np.stack([r * pos, np.minimum(r, 1.5) ** 2 * pos, (r > 1.0) * pos], axis=1)
    out = np.zeros((len(a), V2_DIM), dtype=np.float32)
    base = len(TIERS) * N_HATS
    for j in range(len(TIERS)):
        m = tidx == j
        out[m, j * N_HATS:(j + 1) * N_HATS] = hats[m]
        out[m, base + j * 3: base + j * 3 + 3] = lin[m]
    out[:, base + len(TIERS) * 3:] = lin
    return out


# ---------------------------------------------------------------------------
# network forward (NumPy twin of the PyTorch module in train.py)
# ---------------------------------------------------------------------------

def forward_logits(W: Dict[str, np.ndarray], idx: np.ndarray,
                   dense: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """Shared trunk once, then the three heads. Returns (route[5], approve, grounded)."""
    emb = W["emb"]
    if idx.size:
        pooled = emb[idx].sum(axis=0) * (1.0 / math.sqrt(idx.size))
    else:
        pooled = np.zeros(emb.shape[1], dtype=np.float32)
    x = np.concatenate([pooled.astype(np.float32, copy=False), dense])
    h = np.maximum(x @ W["W1"] + W["b1"], 0.0)
    h = np.maximum(h @ W["W2"] + W["b2"], 0.0)
    r = np.maximum(h @ W["Wr1"] + W["br1"], 0.0) @ W["Wr2"] + W["br2"]
    a = np.maximum(h @ W["Wa1"] + W["ba1"], 0.0) @ W["Wa2"] + W["ba2"]
    g = np.maximum(h @ W["Wg1"] + W["bg1"], 0.0) @ W["Wg2"] + W["bg2"]
    return r.astype(np.float64), float(a[0]), float(g[0])


WEIGHT_KEYS = ["emb", "W1", "b1", "W2", "b2",
               "Wr1", "br1", "Wr2", "br2",
               "Wa1", "ba1", "Wa2", "ba2",
               "Wg1", "bg1", "Wg2", "bg2"]


# ---------------------------------------------------------------------------
# calibration and abstention helpers (also used by train.py)
# ---------------------------------------------------------------------------

def softmax_T(logits: np.ndarray, T: float) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / T
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def route_probs(logits: np.ndarray, cal: Dict[str, Any]) -> np.ndarray:
    """
    Calibrated route distribution from raw head logits (last axis = 5 queues).

    temperature:   softmax(logits / T).
    ovr_isotonic:  per-queue isotonic map applied to logit(softmax prob), then
                   clipped to [P_EPS, 1-P_EPS] and renormalised, so the result is
                   strictly positive, finite, and sums to 1.
    """
    if cal["type"] == "temperature":
        return softmax_T(logits, float(cal["T"]))
    if cal["type"] != "ovr_isotonic":
        raise ValueError("unknown route calibrator %r" % cal["type"])
    base = np.clip(softmax_T(logits, 1.0), 1e-6, 1.0)
    z = np.log(base) - np.log(np.clip(1.0 - base, 1e-6, 1.0))
    out = np.empty_like(base)
    for k in range(base.shape[-1]):
        out[..., k] = np.interp(z[..., k], np.asarray(cal["x"][k], dtype=np.float64),
                                np.asarray(cal["y"][k], dtype=np.float64))
    out = np.clip(out, P_EPS, 1.0 - P_EPS)
    return out / out.sum(axis=-1, keepdims=True)


def sigmoid(z: Any) -> Any:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def apply_binary_cal(z: Any, cal: Dict[str, Any]) -> Any:
    """Map a raw head logit to a calibrated probability of 'yes'."""
    if cal["type"] == "platt":
        p = sigmoid(cal["a"] * np.asarray(z, dtype=np.float64) + cal["b"])
    elif cal["type"] == "isotonic":
        p = np.interp(z, np.asarray(cal["x"], dtype=np.float64),
                      np.asarray(cal["y"], dtype=np.float64))
    else:
        raise ValueError("unknown calibrator %r" % cal["type"])
    return np.clip(p, P_EPS, 1.0 - P_EPS)


def approve_risk(p: Any) -> Any:
    """Expected cost of the better of {approve, hold} at calibrated P(yes)=p."""
    return np.minimum(FP_COST * (1.0 - p), p)


def abstain_binary(p: Any, rule: Dict[str, Any]) -> Any:
    if rule["kind"] == "max_risk":
        return approve_risk(p) > rule["thr"]
    return np.maximum(p, 1.0 - p) < rule["thr"]          # kind == "min_conf"


# ---------------------------------------------------------------------------
# the model served by app.py
# ---------------------------------------------------------------------------

class DecisionModel:
    """
    One model, three decision types.

    Loaded once at startup. route/approve/grounded all read the same cached
    trunk pass for a given state.
    """

    CACHE_SIZE = 2048

    def __init__(self, artifacts_dir: str = "artifacts") -> None:
        self.artifacts_dir = artifacts_dir
        self.ready = False
        self.load_error: Optional[str] = None
        self._cache: "OrderedDict[tuple, Dict[str, Any]]" = OrderedDict()
        self._lock = threading.Lock()
        try:
            self._load(artifacts_dir)
            self.ready = True
        except Exception as exc:                      # keep the endpoint alive
            self.load_error = "%s: %s" % (type(exc).__name__, exc)
            logger.error("DecisionModel could not load artifacts from %r (%s). "
                         "Serving uninformative priors and abstaining.",
                         artifacts_dir, self.load_error)

    # -- loading ------------------------------------------------------------
    def _load(self, d: str) -> None:
        with open(os.path.join(d, "meta.json"), encoding="utf-8") as fh:
            meta = json.load(fh)
        version = meta.get("feature_version")
        if meta.get("format") != ARTIFACT_FORMAT or version not in SUPPORTED_FEATURE_VERSIONS:
            raise ValueError("artifact/feature version mismatch (got format=%r feature_version=%r)"
                             % (meta.get("format"), version))
        if meta.get("queues") != QUEUES:
            raise ValueError("artifact queue order differs from model.QUEUES")
        with open(os.path.join(d, "vocab.json"), encoding="utf-8") as fh:
            vocab = json.load(fh)
        with np.load(os.path.join(d, "weights.npz"), allow_pickle=False) as z:
            W = {k: np.ascontiguousarray(z[k], dtype=np.float32) for k in WEIGHT_KEYS}
        if W["W1"].shape[0] != W["emb"].shape[1] + dense_dim(version):
            raise ValueError("weights do not match the dense feature layout")
        if W["emb"].shape[0] != len(vocab):
            raise ValueError("vocab size does not match the embedding table")
        self.W = W
        caps = meta.get("featurizer", {}).get("tier_caps")
        self.featurizer = Featurizer(vocab, version, caps)
        self.meta = meta
        self.route_cal = meta["calibration"]["route"]
        route_probs(np.zeros(len(QUEUES)), self.route_cal)       # fail at load, not per request
        self.cal_approve = meta["calibration"]["approve"]
        self.cal_grounded = meta["calibration"]["grounded"]
        dec = meta["decision"]
        self.route_abstain = dec["route"]["abstain"]
        self.approve_thr = float(dec["approve"]["threshold"])
        self.approve_abstain = dec["approve"]["abstain"]
        self.grounded_thr = float(dec["grounded"]["threshold"])
        self.grounded_abstain = dec["grounded"]["abstain"]

    # -- shared forward pass ------------------------------------------------
    def _compute(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Uncached trunk pass + all three calibrated outputs."""
        idx, dense = self.featurizer.encode(state)
        r, a, g = forward_logits(self.W, idx, dense)
        probs = route_probs(r, self.route_cal)
        pa = float(apply_binary_cal(a, self.cal_approve))
        pg = float(apply_binary_cal(g, self.cal_grounded))
        if not (np.all(np.isfinite(probs)) and math.isfinite(pa) and math.isfinite(pg)):
            return self._fallback()
        k = int(np.argmax(probs))
        conf = float(probs[k])
        return {
            "route": (QUEUES[k], {q: float(probs[i]) for i, q in enumerate(QUEUES)},
                      conf, bool(conf < self.route_abstain["thr"])),
            "approve": (bool(pa >= self.approve_thr), pa,
                        bool(abstain_binary(pa, self.approve_abstain))),
            "grounded": (bool(pg >= self.grounded_thr), pg,
                         bool(abstain_binary(pg, self.grounded_abstain))),
        }

    @staticmethod
    def _fallback() -> Dict[str, Any]:
        u = 1.0 / len(QUEUES)
        return {"route": (QUEUES[0], {q: u for q in QUEUES}, u, True),
                "approve": (False, 0.4, True),
                "grounded": (False, 0.5, True)}

    def _predict(self, state: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ready:
            return self._fallback()
        key = (_as_str(state.get("subject")), _as_str(state.get("body")),
               _as_str(state.get("customer_tier")), _as_str(state.get("amount")),
               _as_str(state.get("draft_reply")), _as_str(state.get("kb_article")))
        with self._lock:
            hit = self._cache.get(key)
            if hit is not None:
                self._cache.move_to_end(key)
                return hit
        out = self._compute(state)
        with self._lock:
            self._cache[key] = out
            while len(self._cache) > self.CACHE_SIZE:
                self._cache.popitem(last=False)
        return out

    # -- routing ------------------------------------------------------------
    def route(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, float], float, bool]:
        """
        Which queue should this request go to?

        Returns (value, probs, confidence, abstain):
            value       one of QUEUES
            probs       a probability for EVERY queue, summing to 1.0 (+/- 0.01)
            confidence  your probability for `value`, in [0, 1]
            abstain     True if you would rather not commit to an answer
        """
        value, probs, conf, abstain = self._predict(state)["route"]
        return value, dict(probs), conf, abstain

    # -- approval -----------------------------------------------------------
    def approve(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        """
        Can this request be auto-approved, or does it need a human?

        Returns (value, p_true, abstain). `value` applies a cost-aware threshold
        (a wrong approval costs 3x a wrong hold), so it is not simply p_true > 0.5.
        """
        return self._predict(state)["approve"]

    # -- grounding ----------------------------------------------------------
    def grounded(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        """
        Is state["draft_reply"] supported by state["kb_article"]?

        Returns (value, p_true, abstain), same shape as approve().
        """
        return self._predict(state)["grounded"]
