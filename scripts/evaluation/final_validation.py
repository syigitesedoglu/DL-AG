# -*- coding: utf-8 -*-
"""
DL-AG final validation for the legacy balanced (AdvGLUE + IMDB) student.

Targets (legacy path): AdvGLUE accuracy > 50% and IMDB accuracy > 80%.
"""

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from textblob import TextBlob
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from repo_paths import ADVGLUE_DEV_JSON, MODELS_BALANCED, ensure_import_path

ensure_import_path()

MODEL_PATH = str(MODELS_BALANCED)
ADV_DATA_PATH = str(ADVGLUE_DEV_JSON)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def layer1_defense(text):
    try:
        blob = TextBlob(text)
        return str(blob.correct())
    except Exception:
        return text


def load_advglue(filepath):
    if not os.path.exists(filepath):
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("sst2", [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    model_path = args.model_path or os.path.join(MODEL_PATH, f"seed_{args.seed}")
    if not os.path.isdir(model_path) and os.path.isdir(MODEL_PATH):
        # Backward compatibility: older single-folder layout
        model_path = MODEL_PATH

    print("=" * 60)
    print("DL-AG FINAL VALIDATION")
    print("Model: Balanced (Mixed) DistilBERT")
    print(f"Seed: {args.seed}")
    print(f"Path: {model_path}")
    print("=" * 60)

    print("\nLoading model...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(DEVICE)
        model.eval()
        print("Model ready.")
    except Exception as e:
        print(f"ERROR: could not load model: {e}")
        return

    print("\nTEST 1: AdvGLUE attack evaluation...")
    adv_data = load_advglue(ADV_DATA_PATH)
    preds_adv, refs_adv = [], []

    for item in tqdm(adv_data, desc="AdvGLUE"):
        text = layer1_defense(item["sentence"])  # Layer 1 on
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            preds_adv.append(torch.argmax(outputs.logits, dim=-1).item())
            refs_adv.append(item["label"])

    score_adv = accuracy_score(refs_adv, preds_adv) * 100

    print("\nTEST 2: IMDB generalization evaluation...")
    # 500 test examples is enough for a quick check
    imdb_data = load_dataset("imdb", split="test").shuffle(seed=args.seed).select(range(500))
    preds_imdb, refs_imdb = [], []

    for item in tqdm(imdb_data, desc="IMDB"):
        text = item["text"]  # no Layer 1 on clean IMDB (measure clean generalization)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            preds_imdb.append(torch.argmax(outputs.logits, dim=-1).item())
            refs_imdb.append(item["label"])

    score_imdb = accuracy_score(refs_imdb, preds_imdb) * 100

    print("\n" + "=" * 50)
    print("DL-AG evaluation summary")
    print("=" * 50)
    print(f"AdvGLUE accuracy (robustness)  : {score_adv:.2f}%")
    print(f"IMDB accuracy (generalization) : {score_imdb:.2f}%")
    print("-" * 50)

    result = {
        "seed": args.seed,
        "model_path": model_path,
        "advglue_acc": float(score_adv),
        "imdb_acc": float(score_imdb),
        "n_adv": len(refs_adv),
        "n_imdb": len(refs_imdb),
    }
    out_path = args.out or os.path.join("multiseed_results", f"seed_{args.seed}.json")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"Results written to: {out_path}")

    meets_adv = score_adv > 55
    meets_imdb = score_imdb > 80
    if meets_adv:
        print("AdvGLUE: meets the >55% robustness threshold.")
    else:
        print("AdvGLUE: below the >55% robustness threshold.")

    if meets_imdb:
        print("IMDB: meets the >80% clean-accuracy threshold.")
    else:
        print("IMDB: below the >80% clean-accuracy threshold (possible forgetting).")

    print("=" * 50)
    if meets_adv and meets_imdb:
        print("Summary: both evaluation thresholds are satisfied.")
    else:
        print("Summary: one or more thresholds are not satisfied; see scores above.")


if __name__ == "__main__":
    main()
