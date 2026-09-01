# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW

print("1. Kütüphaneler yüklendi. Modeller HuggingFace'ten çekiliyor...")
print("(Bu işlem internet hızınıza göre ilk seferde 1-2 dakika sürebilir, lütfen bekleyin...)")

# =============================================================================
# MODELLERİ VE TOKENIZER'I YÜKLEME
# =============================================================================
# Öğretmen Model: Zaten eğitilmiş, devasa BERT-Base
teacher_name = "textattack/bert-base-uncased-SST-2" 
teacher_model = AutoModelForSequenceClassification.from_pretrained(teacher_name, output_hidden_states=True)

# Öğrenci Model: Eğitilecek küçük DistilBERT
student_name = "distilbert-base-uncased"
student_model = AutoModelForSequenceClassification.from_pretrained(student_name, num_labels=2, output_hidden_states=True)

tokenizer = AutoTokenizer.from_pretrained(teacher_name)

# GPU varsa GPU'yu, yoksa CPU'yu seç
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Kullanılan Donanım: {device}")

teacher_model.to(device)
student_model.to(device)

# Optimizatör tanımlaması
optimizer = AdamW(student_model.parameters(), lr=5e-5)

# =============================================================================
# SENTETİK (DUMMY) VERİ OLUŞTURMA
# =============================================================================
print("2. Örnek veriler (Batch) oluşturuluyor...")
texts = ["This movie is incredibly awesome and I loved it!", 
         "What a terrible and boring film, totally waste of time."]
labels = torch.tensor([1, 0]).to(device)  # 1: Pozitif, 0: Negatif

# Metinleri tokenlara çevir
encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
inputs = encoded['input_ids'].to(device)
masks = encoded['attention_mask'].to(device)

# =============================================================================
# STRATEJİ BELİRLEME
# =============================================================================
STRATEGY = "PKD-Skip" # Seçenekler: "Logit-Only", "PKD-Last", "PKD-Skip"
print(f"3. Aktif Eğitim Stratejisi: {STRATEGY}")

if STRATEGY == "Logit-Only":
    layer_mapping = {} 
elif STRATEGY == "PKD-Last":
    layer_mapping = {4: 10, 5: 11, 6: 12}
elif STRATEGY == "PKD-Skip":
    layer_mapping = {1: 2, 3: 6, 5: 10, 6: 12}

# =============================================================================
# EĞİTİM DÖNGÜSÜ (TRAINING LOOP)
# =============================================================================
print("4. İleri besleme (Forward Pass) ve Optimizasyon başlıyor...")

student_model.train() # Öğrenci eğitim modunda
teacher_model.eval()  # Öğretmen kesinlikle test (eval) modunda, öğrenmeyecek!

# Öğretmen model sadece tahmin yapar, gradyan hesaplamaz (RAM patlamaması için)
with torch.no_grad(): 
    teacher_outputs = teacher_model(input_ids=inputs, attention_mask=masks)

# Öğrenci model öğrenir
student_outputs = student_model(input_ids=inputs, attention_mask=masks)

# Logitler ve Gizli Katmanlar (.detach() ile grafikten koparıyoruz)
teacher_logits = teacher_outputs.logits.detach()
student_logits = student_outputs.logits

teacher_hidden = [h.detach() for h in teacher_outputs.hidden_states]
student_hidden = student_outputs.hidden_states

# Hiperparametreler
T = 2.0
alpha = 0.5
lambda_pkd = 0.1

# Kayıp 1: Çapraz Entropi
loss_ce = F.cross_entropy(student_logits, labels)

# Kayıp 2: Kullback-Leibler Damıtması
loss_kl = F.kl_div(
    F.log_softmax(student_logits / T, dim=-1),
    F.softmax(teacher_logits / T, dim=-1),
    reduction='batchmean'
) * (T * T)

# Kayıp 3: Patient KD (Ara katmanları hizalama)
loss_pkd = torch.tensor(0.0, device=device)

if len(layer_mapping) > 0:
    for s_idx, t_idx in layer_mapping.items():
        s_layer = student_hidden[s_idx]
        t_layer = teacher_hidden[t_idx]
        
        # L2 Normalizasyonu
        s_layer_norm = F.normalize(s_layer, p=2, dim=-1)
        t_layer_norm = F.normalize(t_layer, p=2, dim=-1)
        
        # MSE Özellik Hizalama
        loss_pkd += F.mse_loss(s_layer_norm, t_layer_norm)

# TOPLAM OPTİMİZASYON (KAYIP)
total_loss = (1.0 - alpha) * loss_ce + alpha * loss_kl + lambda_pkd * loss_pkd

# Geriye Yayılım (Backpropagation)
optimizer.zero_grad()
total_loss.backward()
optimizer.step()

print("-" * 50)
print(f"✅ TEBRİKLER! Eğitim adımı başarıyla tamamlandı!")
print(f"📊 Toplam Loss (Hata Değeri): {total_loss.item():.4f}")
print("-" * 50)