#!/usr/bin/env python3
"""train.py — shared encoder + three heads (route, approve, grounded)."""

import argparse, json, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

QUEUES = ["billing", "technical", "account", "shipping", "general"]
QUEUE2ID = {q: i for i, q in enumerate(QUEUES)}


class MultiTaskModel(nn.Module):
    def __init__(self, backbone_name="distilbert-base-uncased"):
        super().__init__()
        self.backbone = AutoModel.from_pretrained(backbone_name)
        H = self.backbone.config.hidden_size
        self.route_head = nn.Linear(H, len(QUEUES))
        self.approve_head = nn.Linear(H, 1)
        self.grounded_head = nn.Linear(H, 1)

    def forward(self, input_ids, attention_mask):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]          # shared [CLS] representation
        return {
            "route_logits": self.route_head(cls),   # (B, 5)
            "approve_logit": self.approve_head(cls),# (B, 1)
            "grounded_logit": self.grounded_head(cls)# (B, 1)
        }


class DecideDataset(Dataset):
    """Each row becomes three samples: route, approve, grounded."""

    def __init__(self, path, tokenizer, max_len=256):
        self.samples = []
        self.tok = tokenizer
        self.max_len = max_len
        with open(path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                s, lab = r["state"], r["labels"]

                # route: ticket text -> one of 5 queues
                self.samples.append({
                    "task": "route",
                    "text": (s.get("subject", "") + " " + s.get("body", "")).strip(),
                    "text_pair": None,
                    "label": QUEUE2ID[lab["route"]],
                })

                # approve: ticket + tier + amount -> binary
                approve_text = (
                    f"tier: {s.get('customer_tier','free')} "
                    f"amount: {s.get('amount',0.0)} . "
                    f"{s.get('subject','')} . {s.get('body','')}"
                ).strip()
                self.samples.append({
                    "task": "approve",
                    "text": approve_text,
                    "text_pair": None,
                    "label": int(bool(lab["approve"])),
                })

                # grounded: PAIRWISE draft vs kb -> binary
                self.samples.append({
                    "task": "grounded",
                    "text": s.get("draft_reply", ""),
                    "text_pair": s.get("kb_article", ""),
                    "label": int(bool(lab["grounded"])),
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        x = self.samples[i]
        if x["text_pair"] is not None:
            enc = self.tok(
                x["text"], x["text_pair"],
                truncation=True, max_length=self.max_len,
                padding=False, return_tensors=None,
            )
        else:
            enc = self.tok(
                x["text"],
                truncation=True, max_length=self.max_len,
                padding=False, return_tensors=None,
            )
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "task": x["task"],
            "label": x["label"],
        }


def collate(batch, tokenizer):
    max_len = max(len(b["input_ids"]) for b in batch)
    input_ids, attention_mask = [], []
    for b in batch:
        pad = max_len - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [tokenizer.pad_token_id] * pad)
        attention_mask.append(b["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "tasks": [b["task"] for b in batch],
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }


def train(args):
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    tok = AutoTokenizer.from_pretrained(args.backbone)
    model = MultiTaskModel(args.backbone)

    ds = DecideDataset(os.path.join(args.data_dir, "train.jsonl"), tok, args.max_len)
    dl = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, tok), num_workers=0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    model.train()
    for epoch in range(args.epochs):
        total, n = 0.0, 0
        for step, batch in enumerate(dl):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(input_ids, attention_mask)
            loss = 0.0
            for i, task in enumerate(batch["tasks"]):
                y = labels[i:i+1]
                if task == "route":
                    loss = loss + F.cross_entropy(out["route_logits"][i:i+1], y)
                elif task == "approve":
                    loss = loss + F.binary_cross_entropy_with_logits(
                        out["approve_logit"][i:i+1].squeeze(-1), y.float())
                elif task == "grounded":
                    loss = loss + F.binary_cross_entropy_with_logits(
                        out["grounded_logit"][i:i+1].squeeze(-1), y.float())
            loss = loss / max(1, len(batch["tasks"]))

            optim.zero_grad()
            loss.backward()
            optim.step()

            total += loss.item(); n += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{len(dl)} loss {total/n:.4f}")

    os.makedirs(args.out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pt"))
    tok.save_pretrained(os.path.join(args.out_dir, "tokenizer"))
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({"backbone": args.backbone, "max_len": args.max_len,
                   "queues": QUEUES}, f, indent=2)
    print(f"saved to {args.out_dir}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_dir", default="checkpoints")
    ap.add_argument("--backbone", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=256)
    train(ap.parse_args())