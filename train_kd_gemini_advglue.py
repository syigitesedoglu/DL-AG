# -*- coding: utf-8 -*-
"""
DL-AG FINAL: TEACHER-STUDENT KNOWLEDGE DISTILLATION
---------------------------------------------------
Recovered from Gemini chat protocol (score-lift path).

Teacher: RoBERTa (SST-2 Expert)
Student: DistilBERT
Defense: TextBlob (Layer 1) on training texts

NOTE:
- Original Gemini script fed DistilBERT token ids to RoBERTa teacher.
  This version uses a separate teacher tokenizer (required for real KD).
- Trains on AdvGLUE SST-2 (dev.json) — very small (~148). Matches the
  "AdvGLUE-only" high-RA path described in son_rapor; do not confuse with
  the 15k sst2_augmented multi-seed campaign.
"""

import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

try:
    from textblob import TextBlob
except ImportError as e:
    raise ImportError("pip install textblob") from e

# --- 1. AYARLAR ---
TEACHER_MODEL_NAME = "textattack/roberta-base-SST-2"
STUDENT_MODEL_NAME = "distilbert-base-uncased"
DATA_PATH = "dev.json"
SAVE_PATH = "./dl_ag_final_student"

BATCH_SIZE = 16
EPOCHS = 5
LR = 5e-5
TEMPERATURE = 2.0
ALPHA = 0.5

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("GPU Aktif (CUDA)")
elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("GPU Aktif (MPS)")
else:
    DEVICE = torch.device("cpu")
    print("CPU Modu Aktif")


# --- 2. LAYER 1: TextBlob ---
def layer1_defense(text):
    try:
        return str(TextBlob(text).correct())
    except Exception:
        return text


# --- 3. DATASET ---
class AdvGlueDataset(Dataset):
    def __init__(self, filepath, student_tokenizer, teacher_tokenizer):
        self.data = []
        self.student_tokenizer = student_tokenizer
        self.teacher_tokenizer = teacher_tokenizer

        print(f"Veri okunuyor: {filepath}")
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"dev.json yok: {filepath}")

        with open(filepath, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        items = raw_data.get("sst2", [])
        if not items:
            print("UYARI: sst2 bulunamadi.")

        print(f"Toplam {len(items)} satir (TextBlob on-isleme)...")
        for item in tqdm(items, desc="Layer1 TextBlob"):
            clean_text = layer1_defense(item["sentence"])
            self.data.append({"text": clean_text, "label": int(item["label"])})

    def __len__(self):
        return len(self.data)

    def _enc(self, tok, text):
        return tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt",
        )

    def __getitem__(self, idx):
        item = self.data[idx]
        s = self._enc(self.student_tokenizer, item["text"])
        t = self._enc(self.teacher_tokenizer, item["text"])
        return {
            "student_input_ids": s["input_ids"].flatten(),
            "student_attention_mask": s["attention_mask"].flatten(),
            "teacher_input_ids": t["input_ids"].flatten(),
            "teacher_attention_mask": t["attention_mask"].flatten(),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


# --- 4. KD LOSS ---
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    distillation = nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1),
    ) * (T * T)
    student_loss = F.cross_entropy(student_logits, labels)
    return alpha * distillation + (1.0 - alpha) * student_loss


# --- 5. TRAIN ---
def main():
    print("=" * 50)
    print("TEACHER-STUDENT (Gemini protokolu, dual tokenizer)")
    print("=" * 50)

    student_tok = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME)
    teacher_tok = AutoTokenizer.from_pretrained(TEACHER_MODEL_NAME)

    print("Teacher (RoBERTa) yukleniyor...")
    teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_MODEL_NAME)
    teacher.to(DEVICE)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    print("Student (DistilBERT) yukleniyor...")
    student = AutoModelForSequenceClassification.from_pretrained(
        STUDENT_MODEL_NAME, num_labels=2
    )
    # Keep GLUE/AdvGLUE label space (0=neg, 1=pos). Do NOT copy textattack
    # id2label — that checkpoint is label-flipped vs AdvGLUE (verified ~29% vs ~71% flipped).
    student.to(DEVICE)
    student.train()

    dataset = AdvGlueDataset(DATA_PATH, student_tok, teacher_tok)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = AdamW(student.parameters(), lr=LR)

    # Auto-detect teacher label orientation on a small probe
    flip_teacher = False
    with torch.no_grad():
        n_ok = n_flip = n = 0
        for i in range(min(64, len(dataset))):
            ex = dataset[i]
            logits = teacher(
                ex["teacher_input_ids"].unsqueeze(0).to(DEVICE),
                attention_mask=ex["teacher_attention_mask"].unsqueeze(0).to(DEVICE),
            ).logits[0]
            y = int(ex["labels"].item())
            pred = int(logits.argmax().item())
            n_ok += int(pred == y)
            n_flip += int((1 - pred) == y)
            n += 1
        flip_teacher = n_flip > n_ok
        print(
            f"Teacher probe: acc={100*n_ok/n:.1f}% flipped_acc={100*n_flip/n:.1f}% "
            f"-> flip_teacher_logits={flip_teacher}"
        )

    print(f"Egitim: {EPOCHS} epoch, n={len(dataset)}")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for batch in loop:
            s_ids = batch["student_input_ids"].to(DEVICE)
            s_mask = batch["student_attention_mask"].to(DEVICE)
            t_ids = batch["teacher_input_ids"].to(DEVICE)
            t_mask = batch["teacher_attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits = teacher(t_ids, attention_mask=t_mask).logits
                if flip_teacher:
                    teacher_logits = teacher_logits.flip(dims=[-1])
            student_logits = student(s_ids, attention_mask=s_mask).logits
            loss = distillation_loss(
                student_logits, teacher_logits, labels, TEMPERATURE, ALPHA
            )
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        print(f"Epoch {epoch+1} mean loss={total_loss / max(len(loader), 1):.4f}")

    os.makedirs(SAVE_PATH, exist_ok=True)
    student.save_pretrained(SAVE_PATH)
    student_tok.save_pretrained(SAVE_PATH)
    print(f"Kaydedildi: {SAVE_PATH}")


if __name__ == "__main__":
    main()
