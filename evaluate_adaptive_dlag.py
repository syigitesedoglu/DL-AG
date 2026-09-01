# -*- coding: utf-8 -*-
"""
Created on Fri Dec 12 11:13:33 2025

@author: yigit
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset
from textblob import TextBlob
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import os
import math

# --- AYARLAR ---
MODEL_PATH = "./layer2_model_augmented/final_robust_model" 
BASE_MODEL_NAME = "distilbert-base-uncased"
# Anormallik tespiti için kullanılacak "bekçi" model
PERPLEXITY_MODEL_NAME = "gpt2" 
# Eşik değeri: Perplexity bu değerden yüksekse Spell Check devreye girer.
# (Bu değeri deneme yanılma ile optimize edebiliriz, şimdilik 100 iyi bir başlangıç)
PERPLEXITY_THRESHOLD = 100.0 

# --- GLOBAL MODELLERİ YÜKLE (Hız için bir kere yüklenir) ---
print(" Bekçi Model (GPT-2) Yükleniyor...")
perp_tokenizer = GPT2Tokenizer.from_pretrained(PERPLEXITY_MODEL_NAME)
perp_model = GPT2LMHeadModel.from_pretrained(PERPLEXITY_MODEL_NAME)
perp_model.eval()
if torch.cuda.is_available():
    perp_model.to("cuda")
print(" Bekçi Hazır.")

def calculate_perplexity(text, device):
    """Cümlenin 'gariplik' (perplexity) skorunu hesaplar."""
    if not text or not isinstance(text, str) or len(text.strip()) == 0: return 0.0
    
    encodings = perp_tokenizer(text, return_tensors='pt')
    input_ids = encodings.input_ids.to(device)

    with torch.no_grad():
        outputs = perp_model(input_ids, labels=input_ids)
    
    loss = outputs.loss
    if loss is None or torch.isnan(loss): return 0.0

    # Loss'un üssünü alarak perplexity'yi buluyoruz
    ppl = math.exp(loss.item())
    return ppl

def basic_spell_check(text):
    """Eski Layer 1: Standart Spell Checker"""
    try:
        blob = TextBlob(text)
        return str(blob.correct())
    except:
        return text

def smart_layer1_defense(text, device):
    """
    YENİ AKILLI LAYER 1 (Adaptive Defense)
    Önce cümlenin garipliğine bakar. Sadece garipse düzeltir.
    """
    if not isinstance(text, str): return ""
    
    # 1. Cümlenin gariplik skorunu ölç
    ppl_score = calculate_perplexity(text, device)
    
    # 2. Karar ver
    if ppl_score > PERPLEXITY_THRESHOLD:
        # Cümle garip (saldırı olabilir), temizle!
        cleaned_text = basic_spell_check(text)
        # print(f"GARİP CÜMLE TESPİT EDİLDİ (Skor: {ppl_score:.2f}) -> Temizlendi.") # Debug için açılabilir
        return cleaned_text
    else:
        # Cümle normal, dokunma.
        return text

def get_input_text(task_name, example):
    """Görevlere göre doğru sütunları çeker."""
    if task_name == "sst2": return example['sentence'], None
    elif task_name == "qqp": return example['question1'], example['question2']
    elif task_name == "mnli": return example['premise'], example['hypothesis']
    elif task_name == "qnli": return example['question'], example['sentence']
    elif task_name == "rte": return example['sentence1'], example['sentence2']
    else: return example.get('sentence', ""), None

def evaluate_task(task_name, model, tokenizer, device):
    config_name = f"adv_{task_name}"
    print(f"\n--- {task_name.upper()} Görevi Değerlendiriliyor (Config: {config_name}) ---")
    print(f" Yöntem: Adaptive DL-AG (Eşik: {PERPLEXITY_THRESHOLD})")
    
    try:
        dataset = load_dataset("adv_glue", config_name, split="validation", trust_remote_code=True)
    except Exception as e:
        print(f" HATA: {task_name} yüklenemedi! Detay: {e}")
        return None

    preds = []
    refs = []
    
    for example in tqdm(dataset, desc=f"{task_name} Testing"):
        label = example['label']
        
        # 1. Doğru sütunları çek
        text_a, text_b = get_input_text(task_name, example)
        
        # 2. AKILLI LAYER 1 Uygula
        # Burada artık smart_layer1_defense kullanıyoruz
        clean_a = smart_layer1_defense(text_a, device)
        clean_b = smart_layer1_defense(text_b, device) if text_b else None
        
        # 3. Tokenize & Tahmin (Ana Model)
        if clean_b:
            inputs = tokenizer(clean_a, clean_b, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        else:
            inputs = tokenizer(clean_a, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_id = torch.argmax(logits, dim=-1).item()
            
        preds.append(predicted_class_id)
        refs.append(label)

    accuracy = accuracy_score(refs, preds)
    print(f">>> {task_name.upper()} Accuracy: {accuracy:.4f}")
    return accuracy

def main():
    print("="*50)
    print("   ADAPTIVE DL-AG FİNAL TESTİ")
    print("   (Dinamik Kapı Bekçili Sistem)")
    print("="*50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    if not os.path.exists(MODEL_PATH):
        print(" HATA: Ana model klasörü bulunamadı!")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(device)
        model.eval()
        print(" Ana Model (DistilBERT) başarıyla yüklendi.")
    except Exception as e:
        print(f" Model yükleme hatası: {e}")
        return

    tasks = ["sst2", "qqp", "mnli", "qnli", "rte"]
    results = {}

    for task in tasks:
        score = evaluate_task(task, model, tokenizer, device)
        if score is not None:
            results[task] = score

    print("\n" + "="*40)
    print("   ADAPTIVE DL-AG SONUÇ RAPORU")
    print("="*40)
    
    total_score = 0
    count = 0
    for task, score in results.items():
        print(f"{task.upper()}: {score:.4f}")
        total_score += score
        count += 1
        
    if count > 0:
        avg = total_score / count
        print("-" * 40)
        print(f"ORTALAMA: {avg:.4f}")
        print("-" * 40)
        print("(Önceki Skor: %42.60 - Bakalım artış var mı?)")

if __name__ == "__main__":
    main()