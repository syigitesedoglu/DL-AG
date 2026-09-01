# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 19:23:54 2025

@author: yigit
"""

# -*- coding: utf-8 -*-
"""
DL-AG FINAL VALIDATION: THE GRAND FINALE
Model: Balanced Teacher-Student (AdvGLUE + IMDB Trained)
Hedef: AdvGLUE > %50  VE  IMDB > %80
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

# --- AYARLAR ---
MODEL_PATH = str(MODELS_BALANCED)
ADV_DATA_PATH = str(ADVGLUE_DEV_JSON)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def layer1_defense(text):
    try:
        blob = TextBlob(text)
        return str(blob.correct())
    except:
        return text

def load_advglue(filepath):
    if not os.path.exists(filepath): return []
    with open(filepath, 'r', encoding='utf-8') as f:
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
        # geriye uyum: eski tek klasör yapısı
        model_path = MODEL_PATH

    print("=" * 60)
    print("🚀 DL-AG FİNAL DOĞRULAMA TESTİ (GRAND FINALE)")
    print("      Model: Balanced (Mixed) DistilBERT")
    print(f"Seed: {args.seed}")
    print(f"Path: {model_path}")
    print("=" * 60)

    # 1. MODELİ YÜKLE
    print("\n📥 Model Yükleniyor...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(DEVICE)
        model.eval()
        print("✅ Model Hazır.")
    except Exception as e:
        print(f"❌ HATA: Model yüklenemedi! {e}")
        return

    # --- TEST 1: ADVGLUE (SALDIRI DİRENCİ) ---
    print("\n🥊 TEST 1: AdvGLUE Saldırı Testi Başlıyor...")
    adv_data = load_advglue(ADV_DATA_PATH)
    preds_adv, refs_adv = [], []

    for item in tqdm(adv_data, desc="AdvGLUE"):
        text = layer1_defense(item['sentence'])  # Layer 1 Aktif
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            preds_adv.append(torch.argmax(outputs.logits, dim=-1).item())
            refs_adv.append(item['label'])

    score_adv = accuracy_score(refs_adv, preds_adv) * 100

    # --- TEST 2: IMDB (GENEL DİL YETENEĞİ) ---
    print("\n📚 TEST 2: IMDB Genelleme Testi Başlıyor...")
    # Hız için 500 test verisi yeterli
    imdb_data = load_dataset("imdb", split="test").shuffle(seed=args.seed).select(range(500))
    preds_imdb, refs_imdb = [], []

    for item in tqdm(imdb_data, desc="IMDB"):
        text = item['text']  # Temiz veriye Layer 1 yapmıyoruz (veya yapabiliriz, şu an saf zekayı ölçelim)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        with torch.no_grad():
            outputs = model(**inputs)
            preds_imdb.append(torch.argmax(outputs.logits, dim=-1).item())
            refs_imdb.append(item['label'])

    score_imdb = accuracy_score(refs_imdb, preds_imdb) * 100

    # --- FİNAL KARNE ---
    print("\n" + "=" * 50)
    print("🏆 DL-AG FİNAL KARNESİ")
    print("=" * 50)
    print(f"🛡️  AdvGLUE Skoru (Robustness)   : %{score_adv:.2f}")
    print(f"🧠  IMDB Skoru (Generalization)  : %{score_imdb:.2f}")
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
    print(f"💾 Sonuç kaydedildi: {out_path}")

    # YORUM
    success = True
    if score_adv > 55:
        print("✅ SALDIRI BAŞARISI: Model saldırıları başarıyla engelliyor! (Hedef > %55)")
    else:
        print("⚠️ SALDIRI UYARISI: Saldırı skoru biraz düşük.")
        success = False

    if score_imdb > 80:
        print("✅ DİL BAŞARISI: Model İngilizceyi unutmamış! (Hedef > %80)")
    else:
        print("⚠️ DİL UYARISI: Modelde hala biraz unutkanlık var.")
        success = False

    print("=" * 50)
    if success:
        print("🌟 SONUÇ: TEBRİKLER! TEZİNİZ SAVUNMAYA HAZIR! 🌟")
        print("   Hem koruyan hem de konuşan dengeli bir model ürettiniz.")
    else:
        print("💡 SONUÇ: Dengeyi tam tutturamadık ama yine de iyi bir deneyim.")

if __name__ == "__main__":
    main()