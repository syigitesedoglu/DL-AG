import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import os

# --- ALP İÇİN ROBUST TRAINER SINIFI (BOOST MODU) ---
class RobustTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        
        # 1. Standart Loss Hesaplama
        outputs = model(**inputs)
        logits = outputs.get("logits")
        labels = inputs.get("labels")
        
        loss_fct = nn.CrossEntropyLoss()
        cls_loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        # --- 2. ALP (Adversarial Logit Pairing) Hesaplanması ---
        
        if logits.shape[0] % 2 != 0:
            return (cls_loss, outputs) if return_outputs else cls_loss
            
        half_batch = logits.shape[0] // 2
        
        clean_logits = logits[:half_batch]
        adv_logits = logits[half_batch:]
        
        mse_loss = nn.MSELoss()(clean_logits, adv_logits)
        
        # DEĞİŞİKLİK 1: Lambda Katsayısı Artırıldı (0.5 -> 2.0)
        # Bu, modeli "tutarlı olmaya" çok daha fazla zorlayacak.
        ALP_lambda = 2.0 
        
        # 3. Toplam Loss
        total_loss = cls_loss + ALP_lambda * mse_loss

        return (total_loss, outputs) if return_outputs else total_loss

# --- AYARLAR ---
# DEĞİŞİKLİK 2: Sıfırdan değil, senin 22 saatte eğittiğin modelden başlıyoruz.
MODEL_NAME = "./layer2_model_full_power/final_robust_model_full"
OUTPUT_DIR = "./layer2_model_boosted"  # Yeni klasöre kaydedelim karışmasın
CSV_FILE = "sst2_augmented.csv" # Eğitim verisi aynı kalmalı (ALP çiftleri burada)

def main():
    print("="*50)
    print(f" BOOST MODU BAŞLATILIYOR (Hedef: %45+ Robustness)")
    print(f" Başlangıç Modeli: {MODEL_NAME}")
    print(" Ayarlar: Alpha=2.0 | Epoch=1 | LR=5e-6")
    print("="*50)
    
    if not os.path.exists(CSV_FILE):
        print(f" HATA: '{CSV_FILE}' bulunamadı!")
        return

    # 1. Veri Yükleme
    print("--- 1. Veri Seti Yükleniyor ---")
    data_files = {"train": CSV_FILE}
    raw_datasets = load_dataset("csv", data_files=data_files)

    # 2. Tokenizer ve Model
    print("--- 2. Tokenizer ve Model Hazırlanıyor ---")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True, padding="max_length", max_length=128)

    print("--- 3. Tokenize İşlemi ---")
    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    
    split_datasets = tokenized_datasets["train"].train_test_split(test_size=0.1)
    train_dataset = split_datasets["train"]
    eval_dataset = split_datasets["test"]

    # 3. Eğitim Ayarları (BOOST İÇİN GÜNCELLENDİ)
    print("--- 4. Eğitim Konfigürasyonu (Boost Ayarları) ---")
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1,          # DEĞİŞİKLİK 3: Sadece 1 Epoch (Hızlı bitmesi için)
        per_device_train_batch_size=16, 
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        learning_rate=5e-6,          # DEĞİŞİKLİK 4: Çok düşük hız (Modeli bozmadan sertleştirmek için)
        weight_decay=0.01,
        logging_steps=100,
        use_cpu=True
    )

    trainer = RobustTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    print("\n" + "="*50)
    print(" BOOST EĞİTİMİ BAŞLIYOR... (Tahmini 3-4 Saat)")
    print("="*50 + "\n")
    
    trainer.train()
    
    final_path = os.path.join(OUTPUT_DIR, "final_robust_model_boosted")
    trainer.save_model(final_path)
    
    print("\n" + "="*50)
    print(f" BOOST İŞLEMİ TAMAMLANDI!")
    print(f" Yeni Model: {final_path}")
    print(" Şimdi evaluate kodunda model yolunu burası yapıp test et!")
    print("="*50)

if __name__ == '__main__':
    main()