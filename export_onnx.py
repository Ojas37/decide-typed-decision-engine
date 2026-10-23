#!/usr/bin/env python3
"""export_onnx.py — export one shared ONNX graph and INT8-quantize it."""

import argparse, json, os, shutil
import numpy as np
import torch
from transformers import AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType

from train import MultiTaskModel


def main(args):
    cfg = json.load(open(os.path.join(args.ckpt, "config.json")))
    tok = AutoTokenizer.from_pretrained(os.path.join(args.ckpt, "tokenizer"))

    model = MultiTaskModel(cfg["backbone"])
    model.load_state_dict(torch.load(os.path.join(args.ckpt, "model.pt"),
                                     map_location="cpu"))
    model.eval()

    dummy = tok("hello world", return_tensors="pt", truncation=True,
                max_length=cfg["max_len"])
    input_ids = dummy["input_ids"]
    attention_mask = dummy["attention_mask"]

    os.makedirs(args.out, exist_ok=True)
    fp32_path = os.path.join(args.out, "model_fp32.onnx")
    int8_path = os.path.join(args.out, "model.onnx")

    torch.onnx.export(
        model,
        (input_ids, attention_mask),
        fp32_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["route_logits", "approve_logit", "grounded_logit"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "seq"},
            "attention_mask": {0: "batch", 1: "seq"},
            "route_logits": {0: "batch"},
            "approve_logit": {0: "batch"},
            "grounded_logit": {0: "batch"},
        },
        opset_version=17,
        do_constant_folding=True,
    )

    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QUInt8)
    print(f"wrote {int8_path} ({os.path.getsize(int8_path)/1e6:.1f} MB)")

    # Copy tokenizer + config next to the ONNX file so model.py loads one dir.
    shutil.copytree(os.path.join(args.ckpt, "tokenizer"),
                    os.path.join(args.out, "tokenizer"), dirs_exist_ok=True)
    with open(os.path.join(args.out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints")
    ap.add_argument("--out", default="artifacts")
    main(ap.parse_args())