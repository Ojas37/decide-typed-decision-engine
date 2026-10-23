"""
model.py — Decide inference layer.

One shared ONNX encoder, three heads. Loaded once in __init__.
"""

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

QUEUES: List[str] = ["billing", "technical", "account", "shipping", "general"]


class DecisionModel:
    def __init__(self, artifacts_dir: str = "artifacts") -> None:
        self.artifacts_dir = artifacts_dir
        cfg_path = os.path.join(artifacts_dir, "config.json")
        calib_path = os.path.join(artifacts_dir, "calibration.json")

        if not os.path.exists(cfg_path):
            raise FileNotFoundError(f"missing {cfg_path}; run export_onnx.py first")
        if not os.path.exists(calib_path):
            raise FileNotFoundError(f"missing {calib_path}; run calibrate.py first")

        self.cfg = json.load(open(cfg_path))
        self.calib = json.load(open(calib_path))
        self.max_len = int(self.cfg.get("max_len", 256))

        self.tokenizer = AutoTokenizer.from_pretrained(
            os.path.join(artifacts_dir, "tokenizer"))

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 4          # match the 4-core judging machine
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            os.path.join(artifacts_dir, "model.onnx"),
            sess_options=so,
            providers=["CPUExecutionProvider"],
        )
        self._input_names = [i.name for i in self.session.get_inputs()]
        self._output_names = [o.name for o in self.session.get_outputs()]

    # ---------- internal helpers ----------
    def _encode(self, text: str, text_pair: str | None = None):
        enc = self.tokenizer(
            text, text_pair,
            truncation=True, max_length=self.max_len,
            padding=False, return_tensors="np",
        )
        return (enc["input_ids"].astype(np.int64),
                enc["attention_mask"].astype(np.int64))

    def _forward(self, input_ids, attention_mask) -> Dict[str, np.ndarray]:
        outputs = self.session.run(
            self._output_names,
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        return dict(zip(self._output_names, outputs))

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    @staticmethod
    def _sigmoid(x: float) -> float:
        return float(1.0 / (1.0 + np.exp(-x)))

    def _t(self, task: str) -> float:
        return float(self.calib["temperatures"].get(task, 1.0))

    def _abstain(self, task: str, confidence: float) -> bool:
        thr = float(self.calib["abstain_thresholds"].get(task, 0.0))
        return confidence < thr

    # ---------- public API ----------
    def route(self, state: Dict[str, Any]) -> Tuple[str, Dict[str, float], float, bool]:
        text = (state.get("subject", "") + " " + state.get("body", "")).strip()
        ids, mask = self._encode(text)
        logits = self._forward(ids, mask)["route_logits"][0].astype(np.float64)
        probs_arr = self._softmax(logits / self._t("route"))
        probs = {q: float(probs_arr[i]) for i, q in enumerate(QUEUES)}
        value = max(probs, key=probs.get)
        confidence = float(probs[value])
        return value, probs, confidence, self._abstain("route", confidence)

    def approve(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        text = (
            f"tier: {state.get('customer_tier','free')} "
            f"amount: {state.get('amount',0.0)} . "
            f"{state.get('subject','')} . {state.get('body','')}"
        ).strip()
        ids, mask = self._encode(text)
        logit = float(self._forward(ids, mask)["approve_logit"][0][0])
        p_true = self._sigmoid(logit / self._t("approve"))
        value = p_true > float(self.calib.get("approve_threshold", 0.75))
        confidence = p_true if value else 1.0 - p_true
        return value, p_true, self._abstain("approve", confidence)

    def grounded(self, state: Dict[str, Any]) -> Tuple[bool, float, bool]:
        draft = state.get("draft_reply", "")
        kb = state.get("kb_article", "")
        ids, mask = self._encode(draft, kb)
        logit = float(self._forward(ids, mask)["grounded_logit"][0][0])
        p_true = self._sigmoid(logit / self._t("grounded"))
        value = p_true > 0.5
        confidence = p_true if value else 1.0 - p_true
        return value, p_true, self._abstain("grounded", confidence)