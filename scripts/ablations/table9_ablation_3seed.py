#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Table 9 component ablation over 3 DL-AG seeds:
- L1-only: Layer1 + untuned DistilBERT (same backbone init per seed via seed)
- L2-only: ARD student without Layer1
- Hybrid: Layer1 + ARD student
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dlag.data import load_advglue_task
from dlag.layer1 import Layer1Purifier
from dlag.utils import get_device, set_seed
from repo_paths import ensure_import_path

ensure_import_path()

SEEDS = [42, 43, 44]


@torch.no_grad()
def score(model, tok, texts, labels, device, purifier=None):
    preds = []
    for text in texts:
        t = purifier.purify(text) if purifier is not None else text
        enc = tok(t, return_tensors="pt", truncation=True, max_length=128, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        preds.append(int(torch.argmax(model(**enc).logits, dim=-1).item()))
    return 100.0 * accuracy_score(labels, preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--dlag-root", type=str, default="./models/dl_ag_student_distilbert")
    ap.add_argument("--advglue", type=str, default="dev.json")
    ap.add_argument("--out", type=str, default="./multiseed_results_full/table9_ablation_summary.json")
    args = ap.parse_args()

    device = get_device()
    data = load_advglue_task(args.advglue, "sst2")
    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]
    purifier = Layer1Purifier(delta=2, lam=0.7)

    rows = []
    buckets = {"L1-only": [], "L2-only": [], "Hybrid": []}

    for seed in args.seeds:
        set_seed(seed)
        student_dir = Path(args.dlag_root) / f"seed_{seed}"
        if not student_dir.exists():
            raise FileNotFoundError(f"Missing DL-AG checkpoint: {student_dir}")

        # L1-only: fresh DistilBERT (seeded init) + purifier
        tok_b = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        base = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        ).to(device)
        # light SST-2 head is random — for ablation fairness use student tokenizer/model path
        # Better L1-only: use a standard SST-2 DistilBERT checkpoint
        tok_l1 = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
        l1_model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        ).to(device)
        ra_l1 = score(l1_model, tok_l1, texts, labels, device, purifier=purifier)

        # L2-only / Hybrid from ARD student
        tok = AutoTokenizer.from_pretrained(student_dir)
        student = AutoModelForSequenceClassification.from_pretrained(student_dir).to(device)
        ra_l2 = score(student, tok, texts, labels, device, purifier=None)
        ra_h = score(student, tok, texts, labels, device, purifier=purifier)

        for name, ra in [("L1-only", ra_l1), ("L2-only", ra_l2), ("Hybrid", ra_h)]:
            buckets[name].append(ra)
            rows.append({"seed": seed, "variant": name, "advglue_sst2_RA": ra})
            print(f"seed={seed} {name}: {ra:.2f}%")

    summary = {"rows": rows, "variants": {}}
    for name, vals in buckets.items():
        arr = np.array(vals, dtype=float)
        summary["variants"][name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "values": vals,
            "seeds": args.seeds,
        }
        print(f"{name}: {arr.mean():.2f} ± {summary['variants'][name]['std']:.2f}")

    # synergy over seeds: hybrid - max(L1,L2)
    syn = []
    for i, seed in enumerate(args.seeds):
        syn.append(buckets["Hybrid"][i] - max(buckets["L1-only"][i], buckets["L2-only"][i]))
    sarr = np.array(syn, dtype=float)
    summary["synergy"] = {
        "values": syn,
        "mean": float(sarr.mean()),
        "std": float(sarr.std(ddof=1)) if len(sarr) > 1 else 0.0,
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
