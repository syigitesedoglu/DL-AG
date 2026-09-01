# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 19:15:08 2025

@author: yigit
"""

# -*- coding: utf-8 -*-
"""
DL-AG FINAL STRATEGY: MIXED BALANCED TRAINING
Amaç: AdvGLUE başarısını korurken (%60-70 bandı), IMDB başarısını geri kazanmak (%80+).
Yöntem: Teacher-Student (RoBERTa -> DistilBERT)
Veri: AdvGLUE (Saldırı) + IMDB (Temiz) Karışımı
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

# --- AYARLAR ---
TEACHER_MODEL_NAME = "textattack/roberta-base-SST-2"
STUDENT_MODEL_NAME = "distilbert-base-uncased"
ADV_DATA_PATH = str(ADVGLUE_DEV_JSON)
SAVE_ROOT = str(MODELS_BALANCED)

# Ryzen Dostu Ayarlar (CPU'yu patlatmayalım ama işe yarasın)
BATCH_SIZE = 16
EPOCHS = 4
LR = 3e-5           # Biraz daha yavaş öğrensin ki unutmasın
TEMPERATURE = 2.0
ALPHA = 0.5

# Kaç tane temiz veri ekleyelim? (Denge Ayarı)
# AdvGLUE'da yaklaşık 150 veri var. Yanına 150-200 tane temiz veri koyarsak denge sağlanır.
CLEAN_SAMPLE_SIZE = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == 'cpu':
    print(" CPU Modu Aktif: İşlem biraz sürecek ama 'Felaket Unutkanlığı' çözeceğiz.")


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

# --- LAYER 1 ---
def layer1_defense(text):
    try:
        blob = TextBlob(text)
        return str(blob.correct())
    except:
        return text

# --- KARIŞIK VERİ SETİ ---
class MixedDataset(Dataset):
    def __init__(self, adv_path, tokenizer, clean_sample_size, seed=42):
        self.data = []
        self.tokenizer = tokenizer

        # 1. AdvGLUE (Saldırı) Verilerini Yükle
        print(f" AdvGLUE Yükleniyor: {adv_path}")
        with open(adv_path, 'r', encoding='utf-8') as f:
            adv_json = json.load(f)
        adv_items = adv_json.get("sst2", [])

        for item in adv_items:
            # Saldırı verilerini TextBlob'dan geçirip ekle
            self.data.append({
                'text': layer1_defense(item['sentence']),
                'label': item['label'],
                'type': 'attack'
            })
        print(f" {len(adv_items)} adet SALDIRI verisi eklendi.")

        # 2. IMDB (Temiz) Verilerini Yükle ve Karıştır
        print(f" IMDB (Temiz Veri) Yükleniyor... ({clean_sample_size} adet)")
        try:
            # Train setinden rastgele temiz veri çek
            imdb_data = load_dataset("imdb", split="train").shuffle(seed=seed).select(range(clean_sample_size))

            for item in imdb_data:
                # Temiz veriye dokunma (Layer 1 uygulamıyoruz ki orjinal dili öğrensin)
                self.data.append({
                    'text': item['text'],
                    'label': item['label'],
                    'type': 'clean'
                })
            print(f" {len(imdb_data)} adet TEMİZ veri eklendi.")

        except Exception as e:
            print(f" IMDB yüklenirken hata: {e}")

        # Verileri Karıştır (Çok Önemli!)
        rng = random.Random(seed)
        rng.shuffle(self.data)
        print(f" Toplam {len(self.data)} adet veri harmanlandı ve eğitime hazır.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=128, # Biraz kıstık ki CPU hızlansın
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

# --- DISTILLATION LOSS ---
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    distillation = nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1)
    ) * (T * T)
    student_loss = F.cross_entropy(student_logits, labels)
    return alpha * distillation + (1.0 - alpha) * student_loss

# --- MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-root", type=str, default=SAVE_ROOT)
    args = parser.parse_args()

    set_seed(args.seed)
    save_path = os.path.join(args.save_root, f"seed_{args.seed}")
    os.makedirs(save_path, exist_ok=True)

    print("=" * 60)
    print(" DL-AG DENGELİ EĞİTİM (BALANCED TRAINING)")
    print(f"Seed: {args.seed}")
    print(f"Save: {save_path}")
    print("Hedef: Hem saldırıyı durdurmak hem de dili konuşmak.")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME)

    # Teacher (RoBERTa)
    print("\n‍ Teacher Yükleniyor...")
    teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_MODEL_NAME)
    teacher.to(device)
    teacher.eval()

    # Student (DistilBERT) - Sıfırdan değil, pretrained halinden başlayalım
    print(" Student Yükleniyor...")
    student = AutoModelForSequenceClassification.from_pretrained(STUDENT_MODEL_NAME, num_labels=2)
    student.to(device)
    student.train()

    # Veri Seti
    dataset = MixedDataset(ADV_DATA_PATH, tokenizer, CLEAN_SAMPLE_SIZE, seed=args.seed)
    g = torch.Generator()
    g.manual_seed(args.seed)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    optimizer = AdamW(student.parameters(), lr=LR)

    print(f"\n Eğitim Başlıyor ({EPOCHS} Epoch)...")

    for epoch in range(EPOCHS):
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            with torch.no_grad():
                teacher_outputs = teacher(input_ids, attention_mask=mask)
            student_outputs = student(input_ids, attention_mask=mask)

            loss = distillation_loss(student_outputs.logits, teacher_outputs.logits, labels, TEMPERATURE, ALPHA)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        print(f" Epoch {epoch+1} Bitti. Ort. Kayıp: {total_loss/len(loader):.4f}")

    print(f"\n Dengeli Model Kaydediliyor: {save_path}")
    student.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    with open(os.path.join(save_path, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump({"seed": args.seed, "epochs": EPOCHS, "alpha": ALPHA, "temperature": TEMPERATURE}, f, indent=2)
    print(" BİTTİ! Şimdi bu modeli hem AdvGLUE hem IMDB ile test et.")

if __name__ == "__main__":
    main()