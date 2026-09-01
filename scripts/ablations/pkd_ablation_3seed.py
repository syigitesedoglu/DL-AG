#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Table 10 stress ablation: Logit-Only KD vs PKD-Skip (and PKD-Last), 3 seeds.

Matches paper stress setting spirit: small SST-2 slice, short train, then AdvGLUE RA.
Produces per-seed scores for t-test (PKD-Skip vs Logit-Only).
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
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dlag.data import load_advglue_task
from dlag.utils import get_device, set_seed
from repo_paths import ensure_import_path

ensure_import_path()

SEEDS = [42, 43, 44]

STRATEGIES = {
    "Logit-Only": {},
    # student_layer -> teacher_layer (hidden_states indices; 0=embeddings)
    "PKD-Last": {4: 10, 5: 11, 6: 12},
    "PKD-Skip": {1: 2, 3: 6, 5: 10, 6: 12},
}


def train_strategy(name, layer_map, teacher, teacher_name, seed, args, device):
    set_seed(seed)
    tok = AutoTokenizer.from_pretrained(teacher_name)
    student = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=2, output_hidden_states=True
    ).to(device)
    student.train()

    raw = load_dataset("glue", "sst2", split="train").shuffle(seed=seed).select(range(args.n_train))
    def tokenize(batch):
        return tok(batch["sentence"], padding="max_length", truncation=True, max_length=128)

    raw = raw.map(tokenize, batched=True)
    raw.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    loader = DataLoader(raw, batch_size=args.batch_size, shuffle=True)

    # project student 768 -> teacher hidden if needed
    t_dim = teacher.config.hidden_size
    s_dim = student.config.hidden_size
    projs = torch.nn.ModuleDict(
        {str(s): torch.nn.Linear(s_dim, t_dim).to(device) for s in layer_map.keys()}
    )
    optim = AdamW(list(student.parameters()) + list(projs.parameters()), lr=args.lr)
    T, alpha, lam_pkd = args.T, args.alpha, args.lambda_pkd

    # one epoch (stress)
    for batch in tqdm(loader, desc=f"{name} seed={seed}", leave=False):
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        optim.zero_grad()
        with torch.no_grad():
            t_out = teacher(input_ids=ids, attention_mask=mask)
        s_out = student(input_ids=ids, attention_mask=mask)
        loss_ce = F.cross_entropy(s_out.logits, labels)
        loss_kl = F.kl_div(
            F.log_softmax(s_out.logits / T, dim=-1),
            F.softmax(t_out.logits / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)
        loss_pkd = torch.tensor(0.0, device=device)
        if layer_map:
            for s_idx, t_idx in layer_map.items():
                hs = s_out.hidden_states[s_idx]
                ht = t_out.hidden_states[t_idx]
                hs_p = projs[str(s_idx)](F.normalize(hs, p=2, dim=-1))
                ht_n = F.normalize(ht, p=2, dim=-1)
                loss_pkd = loss_pkd + F.mse_loss(hs_p, ht_n)
        loss = (1 - alpha) * loss_ce + alpha * loss_kl + lam_pkd * loss_pkd
        loss.backward()
        optim.step()

    out = Path(args.save_root) / name / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    student.save_pretrained(out)
    tok.save_pretrained(out)
    return out, tok


@torch.no_grad()
def eval_advglue(model, tok, data, device):
    model.eval()
    preds, refs = [], []
    for item in data:
        enc = tok(item["text"], return_tensors="pt", truncation=True, max_length=128, padding=True)
        enc = {k: v.to(device) for k, v in enc.items()}
        pred = int(torch.argmax(model(**enc).logits, dim=-1).item())
        preds.append(pred)
        refs.append(item["label"])
    return 100.0 * accuracy_score(refs, preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--n-train", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--T", type=float, default=2.0)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lambda-pkd", type=float, default=0.1)
    ap.add_argument("--teacher", type=str, default="textattack/bert-base-uncased-SST-2")
    ap.add_argument("--advglue", type=str, default="dev.json")
    ap.add_argument("--save-root", type=str, default="./models/pkd_ablation_3seed")
    ap.add_argument("--out", type=str, default="./multiseed_results_full/pkd_ablation_summary.json")
    args = ap.parse_args()

    device = get_device()
    teacher = AutoModelForSequenceClassification.from_pretrained(
        args.teacher, output_hidden_states=True
    ).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    adv = load_advglue_task(args.advglue, "sst2")
    rows = []
    by_strat = {k: [] for k in STRATEGIES}

    for name, layer_map in STRATEGIES.items():
        for seed in args.seeds:
            print("=" * 60)
            print(f"PKD ABLATION {name} seed={seed}")
            out, tok = train_strategy(name, layer_map, teacher, args.teacher, seed, args, device)
            model = AutoModelForSequenceClassification.from_pretrained(out).to(device)
            ra = eval_advglue(model, tok, adv, device)
            by_strat[name].append(ra)
            rows.append({"strategy": name, "seed": seed, "advglue_sst2_RA": ra, "path": str(out)})
            print(f"  RA={ra:.2f}%")

    summary = {"rows": rows, "strategies": {}}
    for name, vals in by_strat.items():
        arr = np.array(vals, dtype=float)
        summary["strategies"][name] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "values": vals,
            "seeds": args.seeds,
        }
        print(f"{name}: {arr.mean():.2f} ± {arr.std(ddof=1) if len(arr)>1 else 0:.2f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
