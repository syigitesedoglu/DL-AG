#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Table 11 baselines: fine-tune BERT-Base and DistilBERT for 3 seeds, eval on AdvGLUE.

Gives matched multi-seed mean±std for fair comparison vs DL-AG.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dlag.data import load_advglue_task
from dlag.utils import get_device, set_seed

SEEDS = [42, 43, 44]


class SST2Train(Dataset):
    def __init__(self, split, tokenizer, max_length=128, max_n=0):
        ds = load_dataset("glue", "sst2", split=split)
        if max_n and max_n > 0:
            ds = ds.select(range(min(max_n, len(ds))))
        self.ds = ds
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        enc = self.tok(
            ex["sentence"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(int(ex["label"]), dtype=torch.long),
        }


@torch.no_grad()
def eval_advglue(model, tokenizer, data, device):
    model.eval()
    preds, refs = [], []
    for item in tqdm(data, desc="AdvGLUE", leave=False):
        enc = tokenizer(
            item["text"],
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        pred = int(torch.argmax(model(**enc).logits, dim=-1).item())
        preds.append(pred)
        refs.append(item["label"])
    return 100.0 * accuracy_score(refs, preds)


def train_one(model_name: str, seed: int, args, device):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    train_ds = SST2Train("train", tok, max_n=args.max_train)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    optim = AdamW(model.parameters(), lr=args.lr)

    model.train()
    for ep in range(args.epochs):
        loop = tqdm(loader, desc=f"{model_name.split('/')[-1]} seed={seed} ep{ep+1}")
        for batch in loop:
            optim.zero_grad()
            out = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                labels=batch["labels"].to(device),
            )
            out.loss.backward()
            optim.step()
            loop.set_postfix(loss=float(out.loss.detach().cpu()))

    out_dir = Path(args.save_root) / model_name.replace("/", "_") / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    return out_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-train", type=int, default=15000, help="SST-2 train subset for CPU finishability")
    ap.add_argument("--advglue", type=str, default="dev.json")
    ap.add_argument("--save-root", type=str, default="./models/baselines_3seed")
    ap.add_argument("--out", type=str, default="./multiseed_results_full/baselines_summary.json")
    ap.add_argument(
        "--models",
        nargs="+",
        default=["bert-base-uncased", "distilbert-base-uncased"],
    )
    args = ap.parse_args()

    device = get_device()
    adv = load_advglue_task(args.advglue, "sst2")
    all_rows = []

    for model_name in args.models:
        scores = []
        for seed in args.seeds:
            print("=" * 60)
            print(f"BASELINE {model_name} seed={seed}")
            out_dir = train_one(model_name, seed, args, device)
            tok = AutoTokenizer.from_pretrained(out_dir)
            model = AutoModelForSequenceClassification.from_pretrained(out_dir).to(device)
            ra = eval_advglue(model, tok, adv, device)
            scores.append(ra)
            row = {"model": model_name, "seed": seed, "advglue_sst2_RA": ra, "path": str(out_dir)}
            all_rows.append(row)
            print(f"  AdvGLUE SST-2 RA = {ra:.2f}%")

        arr = np.array(scores, dtype=float)
        std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        print(f"\n{model_name}: {arr.mean():.2f} ± {std:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    summary = {"rows": all_rows}
    for model_name in args.models:
        vals = [r["advglue_sst2_RA"] for r in all_rows if r["model"] == model_name]
        arr = np.array(vals, dtype=float)
        summary[model_name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "values": vals,
            "seeds": args.seeds,
        }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
