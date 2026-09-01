# -*- coding: utf-8 -*-
"""
Created on Fri Dec 26 19:05:18 2025

@author: yigit
"""

# -*- coding: utf-8 -*-
"""
DL-AG FINAL TEST: MOMENT OF TRUTH
Model: Teacher-Student ile Eğitilmiş DistilBERT
Defense: TextBlob (Layer 1) + Knowledge Distillation (Layer 2)
Hedef: %40.80 ve %42.60 skorlarını geçmek
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from textblob import TextBlob
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import json
import os

# --- AYARLAR ---
# Yeni eğittiğimiz süper modelin yolu
MODEL_PATH = "./dl_ag_final_student"
DATA_PATH = "dev.json"

# Cihaz
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- LAYER 1: SAVUNMA (Aynısı Testte de Olmalı) ---
def layer1_defense(text):
    try:
        blob = TextBlob(text)
        return str(blob.correct())
    except:
        return text

def load_advglue(filepath):
    print(f"📂 Veri okunuyor: {filepath}")
    if not os.path.exists(filepath):
        print("❌ Dosya bulunamadı!")
        return []
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data.get("sst2", [])

def main():
    print("="*60)
    print("🚀 DL-AG FINAL TESTİ: ADVGLUE SALDIRISI")
    print(f"Model: {MODEL_PATH}")
    print("="*60)

    # 1. Modeli Yükle
    print("📥 Model Yükleniyor...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        print("✅ Model Başarıyla Yüklendi!")
    except Exception as e:
        print(f"❌ HATA: Model yüklenemedi. Klasör yolu doğru mu? {e}")
        return

    # 2. Veriyi Yükle
    dataset = load_advglue(DATA_PATH)
    print(f"✅ {len(dataset)} adet saldırı verisi test edilecek.")

    preds = []
    refs = []

    print("\n🥊 Maç Başlıyor...")
    
    for item in tqdm(dataset):
        text = item['sentence']
        label = item['label']
        
        # ADIM 1: Layer 1 Temizliği
        clean_text = layer1_defense(text)
        
        # ADIM 2: Layer 2 Tahmini
        inputs = tokenizer(clean_text, return_tensors="pt", truncation=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item()
            
        preds.append(prediction)
        refs.append(label)

    # SONUÇ
    acc = accuracy_score(refs, preds)
    score = acc * 100
    
    print("\n" + "="*40)
    print(f"🏆 FİNAL SKORU: %{score:.2f}")
    print("="*40)
    
    print("\n--- ANALİZ ---")
    if score > 42.60:
        print("🔥🔥🔥 REKOR KIRILDI! Tezin en yüksek skoruna ulaştın!")
    elif score > 40.80:
        print("✅ BAŞARILI: TextBlob (%40.80) bariyerini aştık. Distillation işe yaradı!")
    elif score > 39.19:
        print("👍 İYİ: Eski resmi skoru (%39.19) geçtik, gelişme var.")
    else:
        print("⚠️ HMM: Skor beklenen kadar artmadı. Analiz etmemiz gerek.")

if __name__ == "__main__":
    main()