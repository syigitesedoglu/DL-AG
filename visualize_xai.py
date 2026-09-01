# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 11:22:09 2025

@author: yigit
"""

import torch
import shap
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import scipy as sp

# --- AYARLAR ---
MODEL_PATH = "./layer2_model_augmented/final_robust_model" 
BASE_MODEL_NAME = "distilbert-base-uncased"

# Analiz edilecek örnek cümleler (Biri normal, biri saldırı)
EXAMPLES = [
    "The movie was absolutely wonderful and the acting was great.", # Pozitif
    "The m0vie was abs0lutely w0nderful and the acting was gr8."   # Saldırı (Adversarial)
]

def main():
    print("="*50)
    print("   XAI GÖRSELLEŞTİRME (SHAP ANALİZİ)")
    print("   Hedef: Modelin hangi kelimelere odaklandığını görmek")
    print("="*50)

    # 1. Model ve Tokenizer Yükle
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval() # Değerlendirme modu

    # SHAP için tahmin fonksiyonu
    def f(emails):
        inputs = tokenizer(emails.tolist(), return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs)
        scores = outputs.logits.detach().numpy()
        # Softmax ile olasılığa çevir
        return sp.special.softmax(scores, axis=1)

    # 2. Explainer Oluştur
    print("⏳ SHAP Explainer oluşturuluyor (Biraz sürebilir)...")
    explainer = shap.Explainer(f, tokenizer)
    
    # 3. Analiz Yap
    print("🔍 Cümleler analiz ediliyor...")
    shap_values = explainer(EXAMPLES)

    # 4. Sonuçları Yazdır
    print("\n" + "="*50)
    print("   ANALİZ SONUÇLARI")
    print("="*50)

    for i, sentence in enumerate(EXAMPLES):
        print(f"\n📝 Cümle {i+1}: {sentence}")
        print("-" * 30)
        
        # Olasılıklar
        probs = f(np.array([sentence]))[0]
        pred_label = "POZİTİF" if probs[1] > probs[0] else "NEGATİF"
        print(f"Tahmin: {pred_label} (Güven: %{max(probs)*100:.2f})")
        
        # En etkili kelimeler
        values = shap_values[i].values
        # Output shape is (seq_len, 2). We act on the predicted class.
        class_idx = 1 if pred_label == "POZİTİF" else 0
        word_values = values[:, class_idx]
        
        tokens = shap_values[i].data
        
        # En çok etki eden 3 kelimeyi bul
        top_indices = np.argsort(-np.abs(word_values))[:3]
        
        print("💡 Kararı Etkileyen En Kritik Kelimeler:")
        for idx in top_indices:
            token = tokens[idx]
            val = word_values[idx]
            impact = "Pozitif Yönlü" if val > 0 else "Negatif Yönlü"
            print(f"   - '{token.strip()}' : {val:.4f} ({impact})")

    print("\n✅ Analiz Tamamlandı. Bu çıktıları tezde 'Model Davranış Analizi' olarak kullanabiliriz.")

if __name__ == "__main__":
    main()