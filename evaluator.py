#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DL-AG evaluator.py — paper-protocol full run.

Flow per example:
  clean accuracy on original text
  for each attack in {TextFooler, DeepWordBug, Pruthi}:
      generate adv -> Layer1 purify -> Layer2 student predict
  report RA, ASR=1-RA, mean latency ms

Also supports AdvGLUE pre-attacked eval mode (--advglue-only) without live attacks.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict

import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dlag.attacks import AttackNotAvailable, build_attacks, generate_adversarial
from dlag.data import load_advglue_task
from dlag.layer1 import Layer1Purifier
from dlag.utils import get_device, set_seed


def parse_args():
    p = argparse.ArgumentParser(description="DL-AG paper-protocol evaluator")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-path", type=str, required=True)
    p.add_argument("--advglue", type=str, default="dev.json")
    p.add_argument("--task", type=str, default="sst2", choices=["sst2", "mnli", "qnli", "qqp"])
    p.add_argument("--max-examples", type=int, default=0, help="0=all")
    p.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "advglue-only"],
        help="full=live TextAttack generation; advglue-only=score pre-attacked AdvGLUE with L1+L2",
    )
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--lambda-l1", type=float, default=0.7)
    p.add_argument("--delta", type=int, default=2)
    return p.parse_args()


@torch.no_grad()
def predict(model, tokenizer, text: str, device, max_length: int = 128) -> int:
    enc = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits
    return int(torch.argmax(logits, dim=-1).item())


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    print("=" * 60)
    print("DL-AG EVALUATOR (paper protocol)")
    print(f"mode={args.mode} model={args.model_path} task={args.task} device={device}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    print("Loading Layer1 purifier (Levenshtein + LM)...")
    purifier = Layer1Purifier(delta=args.delta, lam=args.lambda_l1)

    data = load_advglue_task(args.advglue, task=args.task)
    if args.max_examples and args.max_examples > 0:
        data = data[: args.max_examples]
    print(f"Examples: {len(data)}")

    results: Dict[str, Dict[str, float]] = {
        "clean": {"correct": 0, "total": 0, "time_ms": 0.0},
    }

    attacks = None
    if args.mode == "full":
        try:
            print("Building TextAttack recipes (TextFooler / DeepWordBug / Pruthi)...")
            attacks = build_attacks(model, tokenizer, device)
            for name in attacks:
                results[name] = {"correct": 0, "total": 0, "time_ms": 0.0}
        except AttackNotAvailable as e:
            print(str(e))
            print("Falling back to --mode advglue-only")
            args.mode = "advglue-only"

    if args.mode == "advglue-only":
        results["advglue_l1l2"] = {"correct": 0, "total": 0, "time_ms": 0.0}

    for item in tqdm(data, desc="eval"):
        text = item["text"]
        y = item["label"]

        # clean
        t0 = time.time()
        pred = predict(model, tokenizer, text, device)
        results["clean"]["time_ms"] += (time.time() - t0) * 1000
        results["clean"]["correct"] += int(pred == y)
        results["clean"]["total"] += 1

        if args.mode == "advglue-only":
            t0 = time.time()
            purified = purifier.purify(text)
            pred = predict(model, tokenizer, purified, device)
            results["advglue_l1l2"]["time_ms"] += (time.time() - t0) * 1000
            results["advglue_l1l2"]["correct"] += int(pred == y)
            results["advglue_l1l2"]["total"] += 1
            continue

        assert attacks is not None
        for name, attack in attacks.items():
            adv = generate_adversarial(attack, text, y)
            t0 = time.time()
            purified = purifier.purify(adv)
            pred = predict(model, tokenizer, purified, device)
            results[name]["time_ms"] += (time.time() - t0) * 1000
            results[name]["correct"] += int(pred == y)
            results[name]["total"] += 1

    summary = {}
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for name, m in results.items():
        if m["total"] <= 0:
            continue
        ra = 100.0 * m["correct"] / m["total"]
        asr = 100.0 - ra if name != "clean" else None
        lat = m["time_ms"] / m["total"]
        summary[name] = {
            "RA": ra,
            "ASR": asr,
            "latency_ms": lat,
            "n": m["total"],
            "correct": m["correct"],
        }
        if asr is None:
            print(f"[{name:<14}] CA/RA={ra:.2f}% | latency={lat:.2f} ms")
        else:
            print(f"[{name:<14}] RA={ra:.2f}% | ASR={asr:.2f}% | latency={lat:.2f} ms")

    # attack mean if live attacks present
    attack_names = [k for k in summary if k not in ("clean", "advglue_l1l2")]
    if attack_names:
        mean_ra = sum(summary[k]["RA"] for k in attack_names) / len(attack_names)
        summary["mean_attack_RA"] = mean_ra
        print(f"[mean_attack  ] RA={mean_ra:.2f}%")

    out = args.out or str(Path("multiseed_results_full") / f"seed_{args.seed}_{args.task}_{args.mode}.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "seed": args.seed,
        "model_path": args.model_path,
        "task": args.task,
        "mode": args.mode,
        "summary": summary,
        "raw": results,
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
