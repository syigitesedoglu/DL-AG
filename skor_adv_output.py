# -*- coding: utf-8 -*-
"""
Created on Mon Dec 22 18:29:54 2025

@author: yigit
"""

# -*- coding: utf-8 -*-
"""
DL-AG: Intermediate Baseline Check
Layer 1: MLM Purification (BERT)
Layer 2: Robust DistilBERT (Current Trained Model)
Dataset: AdvGLUE (dev.json) - SST-2 Task
"""

import json
import torch
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForMaskedLM

# --- AYARLAR ---
# AdvGLUE dev.json dosyasının yolu (Klasöründe olduğundan emin ol)
ADVGLUE_PATH = "dev.json" 
# Eğittiğin Layer 2 modeli
MODEL_PATH = "./layer2_model_augmented/final_robust_model"
# Layer 1 için kullanılacak model (Hız için 'distilbert' seçebilirsin, şimdilik 'bert' kalsın güçlü olsun)
PURIFIER_MODEL_NAME = "bert-base-uncased"
BASE_MODEL_NAME = "distilbert-base-uncased"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. ADIM: VERİ SETİNİ YÜKLEME ---
def load_advglue_sst2(filepath):
    print(f"📂 AdvGLUE verisi okunuyor: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"❌ HATA: '{filepath}' bulunamadı! Lütfen dosya yolunu kontrol et.")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # AdvGLUE JSON yapısı: { "sst2": [ {...}, ... ], "qqp": ... }
    if "sst2" not in data:
        raise ValueError("❌ JSON içinde 'sst2' anahtarı bulunamadı!")
    
    sst2_data = data["sst2"]
    print(f"✅ Toplam {len(sst2_data)} adet SST-2 saldırı örneği yüklendi.")
    return sst2_data

# --- 2. ADIM: LAYER 1 (MLM TEMİZLEYİCİ) ---
class MLMPurifier:
    def __init__(self, model_name=PURIFIER_MODEL_NAME, device=DEVICE):
        print(f"🛠️ Layer 1 Hazırlanıyor: MLM Purifier ({model_name})...")
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForMaskedLM.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def purify(self, text):
        if not text or len(text.strip()) == 0:
            return ""
        
        # Tokenize
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
        input_ids = inputs["input_ids"][0]
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits

        predicted_ids = torch.argmax(predictions, dim=-1)[0]
        
        # Onarım Döngüsü
        restored_tokens = []
        for i, token_id in enumerate(input_ids):
            # [CLS] ve [SEP] koru
            if token_id in [101, 102]: 
                restored_tokens.append(token_id.item())
            else:
                # BERT'in tahminini al (Akıllı Onarım)
                restored_tokens.append(predicted_ids[i].item())

        purified_text = self.tokenizer.decode(restored_tokens, skip_special_tokens=True)
        return purified_text

# --- 3. ADIM: ANA TEST DÖNGÜSÜ ---
def main():
    print("="*60)
    print("      DL-AG ARA KONTROL: ADVGLUE BAŞARIM TESTİ")
    print(f"      Hedef: TextBlob (%40.80) vs MLM (Yeni)")
    print("="*60)
    print(f"Cihaz: {DEVICE}")

    # Veriyi Yükle
    try:
        dataset = load_advglue_sst2(ADVGLUE_PATH)
    except Exception as e:
        print(e)
        return

    # Layer 2 Modelini Yükle
    print("\n📥 Layer 2 (Sınıflandırıcı) Yükleniyor...")
    tokenizer_l2 = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    try:
        model_l2 = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model_l2.to(DEVICE)
        model_l2.eval()
        print("✅ Layer 2 Modeli Hazır.")
    except:
        print("❌ HATA: Eğitilmiş model bulunamadı!")
        return

    # Layer 1 Başlat
    purifier = MLMPurifier(device=DEVICE)

    preds = []
    refs = []

    print("\n🚀 Test Başlıyor (Bu işlem işlemci hızına göre biraz sürebilir)...")
    
    # Hızlı sonuç görmek için istersen ilk 100 tanesini test et: dataset[:100]
    # Ama gerçek skor için hepsini yapmalıyız.
    for i, item in enumerate(tqdm(dataset)):
        # AdvGLUE veri yapısına göre anahtarlar: 'sentence', 'label'
        text = item['sentence']
        label = item['label']
        
        # 1. TEMİZLE (MLM)
        clean_text = purifier.purify(text)

        # 2. TAHMİN ET (DistilBERT)
        inputs = tokenizer_l2(clean_text, return_tensors="pt", truncation=True, padding=True, max_length=256).to(DEVICE)
        
        with torch.no_grad():
            outputs = model_l2(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=-1).item()
            
        preds.append(predicted_class_id)
        refs.append(label)

        # İlk 3 örneği görelim, nasıl temizlemiş?
        if i < 3:
            print(f"\n--- Örnek {i+1} ---")
            print(f"Gürültülü: {text}")
            print(f"Temiz    : {clean_text}")

    # SONUÇ
    acc = accuracy_score(refs, preds)
    print("\n" + "="*40)
    print(f"ESKİ SKOR (TextBlob): %40.80")
    print(f"YENİ SKOR (MLM)     : %{acc*100:.2f}")
    print("="*40)
    
    diff = (acc * 100) - 40.80
    if diff > 0:
        print(f"🎉 TEBRİKLER! Skor {diff:.2f} puan arttı.")
    else:
        print(f"⚠️ Skor {abs(diff):.2f} puan düştü. (Bu da bir sonuçtur, analiz edilir)")

if __name__ == "__main__":
    main()