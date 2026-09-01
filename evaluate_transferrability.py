# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 10:49:35 2025

@author: yigit
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from textblob import TextBlob
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os

# --- AYARLAR ---
# Senin eğittiğin ve 42.60 alan modelin yolu
MODEL_PATH = "./layer2_model_augmented/final_robust_model" 
BASE_MODEL_NAME = "distilbert-base-uncased"

def layer1_defense(text):
    """Layer 1: Spell Checker"""
    try:
        blob = TextBlob(text)
        return str(blob.correct())
    except:
        return text

def main():
    print("="*50)
    print("   DL-AG GENELLEME (TRANSFERABILITY) TESTİ")
    print("   Hedef: Modelin IMDB üzerindeki başarısını ölçmek")
    print("="*50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    if not os.path.exists(MODEL_PATH):
        print("❌ HATA: Model bulunamadı! Yol yanlış olabilir.")
        return

    # Modeli Yükle
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    print("✅ Model yüklendi.")

    # IMDB Test Verisini Yükle
    print("📥 IMDB Veri Seti Yükleniyor...")
    try:
        # Sadece test setinden rastgele 1000 örnek alalım (Hızlı olsun)
        dataset = load_dataset("imdb", split="test").shuffle(seed=42).select(range(1000))
    except Exception as e:
        print(f"Hata: {e}")
        return

    preds = []
    refs = []

    print("🚀 Test Başlıyor (Layer 1 + Layer 2)...")
    for example in tqdm(dataset):
        text = example['text']
        label = example['label']
        
        # 1. Layer 1 Uygula
        # IMDB metinleri çok uzun olduğu için Layer 1 çok yavaşlatabilir. 
        # İstersen burayı yorum satırı yapıp sadece Layer 2 gücüne bakabiliriz.
        # clean_text = layer1_defense(text) 
        clean_text = text # Şimdilik hız için direkt verelim, Layer 2 gücünü görelim.

        # 2. Tokenize & Tahmin
        inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=-1).item()
            
        preds.append(predicted_class_id)
        refs.append(label)

    acc = accuracy_score(refs, preds)
    print("\n" + "="*40)
    print(f"IMDB SKORU: %{acc*100:.2f}")
    print("="*40)
    
    if acc > 0.80:
        print("🌟 MÜKEMMEL: Model sadece AdvGLUE'yu ezberlememiş, genel İngilizceyi çözmüş!")
    elif acc > 0.70:
        print("✅ İYİ: Modelin genelleme yeteneği kabul edilebilir seviyede.")
    else:
        print("⚠️ DÜŞÜK: Model SST-2 verisine biraz fazla odaklanmış (Overfitting).")

if __name__ == "__main__":
    main()