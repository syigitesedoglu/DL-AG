# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 19:07:55 2025

@author: yigit
"""

# -*- coding: utf-8 -*-
"""
DL-AG FINAL CHECK: GENERALIZATION TEST
Model: Teacher-Student (AdvGLUE ile eğitilmiş)
Hedef: IMDB skorunun %80 üzerinde kalması
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from textblob import TextBlob
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os

# --- AYARLAR ---
MODEL_PATH = "./dl_ag_final_student" # Yeni modelin yolu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def layer1_defense(text):
    try:
        blob = TextBlob(text)
        return str(blob.correct())
    except:
        return text

def main():
    print("="*60)
    print("🚀 DL-AG 'SAĞLAMALAMA' TESTİ: IMDB GENELLEME")
    print("      AdvGLUE Skoru: %75.00 (Harika)")
    print("      Soru: Peki ya normal İngilizce?")
    print("="*60)

    # Modeli Yükle
    print("📥 Yeni Model Yükleniyor...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
    except:
        print("❌ Model bulunamadı! Yol yanlış olabilir.")
        return

    # IMDB Test Verisini Yükle
    print("📥 IMDB (Test Seti) İndiriliyor...")
    # Rastgele 500 örnek yeterli, hızlı olsun
    dataset = load_dataset("imdb", split="test").shuffle(seed=42).select(range(500))

    preds = []
    refs = []

    print("🚀 Test Başlıyor (Clean Data Performance)...")
    for example in tqdm(dataset):
        text = example['text']
        label = example['label']
        
        # Layer 1'i de dahil edelim ki gerçek senaryo olsun
        # (İstersen hız için kapatabilirsin ama doğrusu açık olmasıdır)
        # clean_text = layer1_defense(text) 
        clean_text = text # Hız için kapalı tutuyorum, modelin saf zekasına bakalım
        
        inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs)
            prediction = torch.argmax(outputs.logits, dim=-1).item()
            
        preds.append(prediction)
        refs.append(label)

    acc = accuracy_score(refs, preds) * 100
    
    print("\n" + "="*40)
    print(f"📊 IMDB SKORU: %{acc:.2f}")
    print("="*40)
    
    if acc > 80:
        print("🌟 MUCİZE: Model hem saldırıları öğrendi (%75) hem de dili unutmadı!")
        print("   Bu, tezin EN GÜÇLÜ sonucudur.")
    elif acc > 70:
        print("✅ KABUL EDİLEBİLİR: Biraz düşüş var ama hala çalışıyor.")
    else:
        print("⚠️ UYARI: Model saldırıları ezberlerken normal dili biraz unutmuş (Overfitting).")

if __name__ == "__main__":
    main()