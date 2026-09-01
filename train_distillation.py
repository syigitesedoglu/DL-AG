# -*- coding: utf-8 -*-
"""
DL-AG FINAL: TEACHER-STUDENT KNOWLEDGE DISTILLATION
---------------------------------------------------
Teacher: RoBERTa (SST-2 Expert) -> Akıl Hocası
Student: DistilBERT (Bizim Model) -> Çırak
Defense: TextBlob (Layer 1) -> Veriyi yumuşatıp öğrenciye veriyoruz
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
# DÜZELTME: AdamW artık torch.optim içinden çağrılıyor
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.optim import AdamW 
from textblob import TextBlob
from tqdm import tqdm
import json
import os

# --- 1. AYARLAR ---
# Hoca (Hazır, Zehir gibi model)
TEACHER_MODEL_NAME = "textattack/roberta-base-SST-2"
# Öğrenci (Senin eğiteceğin model)
STUDENT_MODEL_NAME = "distilbert-base-uncased"
# Veri Seti
DATA_PATH = "dev.json"
# Kayıt Yolu
SAVE_PATH = "./dl_ag_final_student"

# Hiperparametreler (Ryzen için optimize edildi)
BATCH_SIZE = 16 
EPOCHS = 5          
LR = 5e-5
TEMPERATURE = 2.0   # Hocanın bilgisini yumuşatma katsayısı
ALPHA = 0.5         # %50 Hoca, %50 Gerçek Cevap

# Cihaz Seçimi (Otomatik)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("🚀 GPU Aktif (NVIDIA)")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("🚀 GPU Aktif (Mac M-Serisi)")
else:
    DEVICE = torch.device("cpu")
    print("⚠️ CPU Modu Aktif (Biraz sürebilir, sabırlı ol)")

# --- 2. LAYER 1: SAVUNMA MEKANİZMASI ---
def layer1_defense(text):
    """
    Modeli eğitirken veriyi TextBlob'dan geçiriyoruz.
    Böylece model 'temizlenmiş' veriyi görmeye alışıyor.
    """
    try:
        blob = TextBlob(text)
        return str(blob.correct())
    except:
        return text

# --- 3. VERİ SETİ HAZIRLIĞI ---
class AdvGlueDataset(Dataset):
    def __init__(self, filepath, tokenizer):
        self.data = []
        self.tokenizer = tokenizer
        
        print(f"📂 Veri okunuyor: {filepath}")
        if not os.path.exists(filepath):
            raise FileNotFoundError("❌ HATA: dev.json dosyası bulunamadı!")

        with open(filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
            
        # AdvGLUE yapısına göre veriyi çek
        if "sst2" in raw_data:
            items = raw_data["sst2"]
        else:
            items = [] 
            print("⚠️ UYARI: 'sst2' verisi bulunamadı.")

        print(f"✅ Toplam {len(items)} satır veri işleniyor...")

        for item in tqdm(items, desc="Ön İşleme (TextBlob)"):
            text = item['sentence']
            label = item['label']
            
            # KRİTİK ADIM: Temizlenmiş veriyi listeye ekle
            clean_text = layer1_defense(text)
            
            self.data.append({
                'text': clean_text,
                'label': label
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=128,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(item['label'], dtype=torch.long)
        }

# --- 4. DISTILLATION LOSS (Sihirli Formül) ---
def distillation_loss(student_logits, teacher_logits, labels, T, alpha):
    # 1. Hoca ile Öğrenci arasındaki fark (Soft Target Loss)
    distillation = nn.KLDivLoss(reduction="batchmean")(
        F.log_softmax(student_logits / T, dim=1),
        F.softmax(teacher_logits / T, dim=1)
    ) * (T * T)
    
    # 2. Öğrenci ile Gerçek Cevap arasındaki fark (Hard Target Loss)
    student_loss = F.cross_entropy(student_logits, labels)
    
    # İkisinin dengeli toplamı
    return alpha * distillation + (1.0 - alpha) * student_loss

# --- 5. EĞİTİM DÖNGÜSÜ ---
def main():
    print("="*50)
    print("🚀 TEACHER-STUDENT EĞİTİMİ BAŞLIYOR")
    print("="*50)

    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_NAME)
    
    # Hoca Modelini Yükle (Freeze - Eğitilmeyecek)
    print("\n👨‍🏫 Teacher (RoBERTa) Yükleniyor...")
    teacher = AutoModelForSequenceClassification.from_pretrained(TEACHER_MODEL_NAME)
    teacher.to(DEVICE)
    teacher.eval()
    
    # Öğrenci Modelini Yükle (Train - Eğitilecek)
    print("🎓 Student (DistilBERT) Yükleniyor...")
    student = AutoModelForSequenceClassification.from_pretrained(STUDENT_MODEL_NAME, num_labels=2)
    student.to(DEVICE)
    student.train()

    # Veriyi Yükle
    dataset = AdvGlueDataset(DATA_PATH, tokenizer)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    optimizer = AdamW(student.parameters(), lr=LR)

    print(f"\n🔥 Eğitim Başladı! (Toplam {EPOCHS} Epoch)")
    print("Not: Loss değeri düştükçe öğrenci akıllanıyor demektir.")
    
    for epoch in range(EPOCHS):
        total_loss = 0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for batch in loop:
            input_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            
            optimizer.zero_grad()
            
            # Hoca ne diyor?
            with torch.no_grad():
                teacher_outputs = teacher(input_ids, attention_mask=mask)
                teacher_logits = teacher_outputs.logits
                
            # Öğrenci ne diyor?
            student_outputs = student(input_ids, attention_mask=mask)
            student_logits = student_outputs.logits
            
            # Hata hesapla ve öğren
            loss = distillation_loss(student_logits, teacher_logits, labels, TEMPERATURE, ALPHA)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")
            
        print(f"✅ Epoch {epoch+1} Bitti. Ortalama Kayıp: {total_loss / len(loader):.4f}")

    # Kaydet
    print(f"\n💾 Model Kaydediliyor: {SAVE_PATH}")
    student.save_pretrained(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)
    print("🎉 EĞİTİM TAMAMLANDI!.")

if __name__ == "__main__":
    main()