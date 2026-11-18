# -*- coding: utf-8 -*-
"""
DL-AG legacy path: mixed balanced training.

Goal: keep AdvGLUE performance (about 60-70%) while recovering IMDB
clean accuracy (80%+).

Method: teacher-student distillation (RoBERTa -> DistilBERT)
Data: AdvGLUE (attack) + IMDB (clean) mix
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from textblob import TextBlob
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from repo_paths import ADVGLUE_DEV_JSON, MODELS_BALANCED, ensure_import_path

ensure_import_path()

TEACHER_MODEL_NAME = "textattack/roberta-base-SST-2"
STUDENT_MODEL_NAME = "distilbert-base-uncased"
ADV_DATA_PATH = str(ADVGLUE_DEV_JSON)
SAVE_ROOT = str(MODELS_BALANCED)

# CPU-friendly defaults
BATCH_SIZE = 16
EPOCHS = 4
LR = 3e-5  # slightly lower LR to reduce forgetting
TEMPERATURE = 2.0
ALPHA = 0.5

# How many clean IMDB rows to mix in (balance knob).
# AdvGLUE SST-2 subset is ~150 examples; 150-200 clean rows works well.
CLEAN_SAMPLE_SIZE = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    print("CPU mode: training will be slower; used to mitigate catastrophic forgetting.")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def layer1_defense(text):
    try:
        blob = TextBlob(text)
        return str(blob.correct())
    except Exception:
        return text


class MixedDataset(Dataset):
    def __init__(self, adv_path, tokenizer, clean_sample_size, seed=42):
        self.data = []
        self.tokenizer = tokenizer

        print(f"Loading AdvGLUE: {adv_path}")
        with open(adv_path, "r", encoding="utf-8") as f:
            adv_json = json.load(f)
        adv_items = adv_json.get("sst2", [])

        for item in adv_items:
            # Pass attack text through TextBlob Layer-1 proxy, then add
            self.data.append(
                {
                    "text": layer1_defense(item["sentence"]),
                    "label": item["label"],
                    "type": "attack",
                }
            )
        print(f"Added {len(adv_items)} attack examples.")

        print(f"Loading IMDB clean data... ({clean_sample_size} examples)")
        try:
            imdb_data = (
                load_dataset("imdb", split="train")
                .shuffle(seed=seed)
                .select(range(clean_sample_size))
            )

            for item in imdb_data:
                # Do not apply Layer 1 on clean text (preserve original language signal)
                self.data.append(
                    {
                        "text": item["text"],
                        "label": item["label"],
                        "type": "clean",
                    }
                )
            print(f"Added {len(imdb_data)} clean examples.")

        except Exception as e:
            print(f"Error loading IMDB: {e}")

        rng = random.Random(seed)
        rng.shuffle(self.data)
        print(f"Shuffled {len(self.data)} total examples for training.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item["text"],
            truncation=True,
            padding="max_length",
            max_length=128,  # shorter sequences for faster CPU training
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    distillation = nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
    ) * (T * T)
    student_loss = F.cross_entropy(student_logits, labels)
    return alpha * distillation + (1.0 - alpha) * student_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-root", type=str, default=SAVE_ROOT)
    args = parser.parse_args()

    set_seed(args.seed)
    save_path = os.path.join(args.save_root, f"seed_{args.seed}")
    os.makedirs(save_path, exist_ok=True)

    print("=" * 60)
    print("DL-AG balanced training")
    print(f"Seed: {args.seed}")
    print(f"Save path: {save_path}")
    print("Objective: improve adversarial robustness with limited clean-accuracy loss.")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME)

    print("\nLoading teacher...")
    teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_MODEL_NAME)
    teacher.to(device)
    teacher.eval()

    # Student starts from pretrained DistilBERT (not random init)
    print("Loading student...")
    student = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_MODEL_NAME, num_labels=2
    )
    student.to(device)
    student.train()

    dataset = MixedDataset(ADV_DATA_PATH, tokenizer, CLEAN_SAMPLE_SIZE, seed=args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    optimizer = AdamW(student.parameters(), lr=LR)

    print(f"\nStarting training ({EPOCHS} epochs)...")

    for epoch in range(EPOCHS):
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in loop:
            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                teacher_outputs = teacher(input_ids, attention_mask=mask)
            student_outputs = student(input_ids, attention_mask=mask)

            loss = distillation_loss(
                student_outputs.logits,
                teacher_outputs.logits,
                labels,
                TEMPERATURE,
                ALPHA,
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch+1} done. Mean loss: {total_loss/len(loader):.4f}")

    print(f"\nSaving balanced model: {save_path}")
    student.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    with open(os.path.join(save_path, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "epochs": EPOCHS,
                "alpha": ALPHA,
                "temperature": TEMPERATURE,
            },
            f,
            indent=2,
        )
    print("Training finished. Evaluate the checkpoint on AdvGLUE and IMDB.")


if __name__ == "__main__":
    main()
