import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from tqdm import tqdm
import time
import os

# GPU kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CONFIGURATION ---
HUGGING_FACE_TOKEN = "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" # GÜNCEL TOKEN'ınızı BURAYA YAPIŞTIRIN!
MODEL_NAME = "distilbert-base-uncased"
TASK = "sst2"
OUTPUT_DIR = "./pgd_sst2_robust_model" 

# PGD Parametreleri
PGD_EPSILON = 1.0
PGD_ALPHA = 0.3
PGD_K = 3


# --- PGD SINIFI: Adversarial Training Çekirdeği ---
class PGD_Adversarial_Training:
    # GRAD_NORM HATASI ÇÖZÜMÜ İÇİN emb_name KOŞULU GÜNCELLENDİ
    def __init__(self, model, emb_name='word_embeddings', epsilon=PGD_EPSILON, alpha=PGD_ALPHA, K=PGD_K):
        self.model = model
        self.emb_name = emb_name
        self.epsilon = epsilon
        self.alpha = alpha
        self.K = K
        
        # SIFIR GRADYAN HATASI ÇÖZÜMÜ: Doğru embedding katmanını arama
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'word_embeddings' in name and 'weight' in name:
                self.emb_backup = param.data.clone()
                break # Sadece ilkini yakalamak yeterli

    def attack(self):
        """PGD saldırısını başlatır (Rastgele gürültü ile)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'word_embeddings' in name and 'weight' in name:
                param.data.add_(torch.randn_like(param.data) * self.alpha)

    def restore(self):
        """Modeli orijinal ağırlıklarına geri yükler."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'word_embeddings' in name and 'weight' in name:
                param.data.copy_(self.emb_backup)

    def project(self, param):
        """Pertürbasyonu epsilon sınırı içinde tutar."""
        if param.data.is_cuda:
            norm = torch.norm(param.data - self.emb_backup.cuda(), p=2, dim=-1, keepdim=True)
            if norm.mean() > self.epsilon:
                data = self.emb_backup.cuda() + self.epsilon * (param.data - self.emb_backup.cuda()) / norm
                param.data.copy_(data)
        else:
            norm = torch.norm(param.data - self.emb_backup, p=2, dim=-1, keepdim=True)
            if norm.mean() > self.epsilon:
                data = self.emb_backup + self.epsilon * (param.data - self.emb_backup) / norm
                param.data.copy_(data)

    def step(self):
        """PGD adımını atar (Gradyan yönünde ilerleme)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and 'word_embeddings' in name and 'weight' in name:
                param.data.add_(self.alpha * param.grad.sign()) 
                self.project(param)


# --- PGD TRAINER: Trainer'a Saldırı Mantığını Enjekte Eden Sınıf ---
class PGD_Trainer(Trainer):
    # TÜM HATA ÇÖZÜMLERİ BU METOTTA BİRLEŞTİRİLMİŞTİR
    def training_step(self, model, inputs, optimizer=None, **kwargs):
        
        # 1. Normal eğitim adımı (super().training_step'i çağırın)
        if optimizer is not None:
            loss = super().training_step(model, inputs, optimizer=optimizer, **kwargs)
        else:
             loss = super().training_step(model, inputs, **kwargs)

        # --- ADVERSARIAL TRAINING BAŞLANGIÇ ---
        
        pgd = PGD_Adversarial_Training(model, epsilon=PGD_EPSILON, alpha=PGD_ALPHA, K=PGD_K)

        # 1A. PGD için Gradyan Hesaplama (SIFIR GRADYAN HATASI ÇÖZÜMÜ)
        # Modelin optimizer'dan önce bir gradyan oluşturmasını sağlıyoruz.
        # Bu, grad_norm'un sıfır çıkmasını engeller.
        if optimizer is not None:
            # Modelin normal kaybını hesapla
            with torch.enable_grad():
                initial_loss = model(**inputs).loss
            initial_loss.backward()
            
        
        # 2. PGD saldırısını başlat
        pgd.attack() 
        model.zero_grad() 

        # 3. K adım PGD uygula (Adversarial kayıp optimizasyonu)
        for _ in range(pgd.K):
            with torch.enable_grad():
                adv_loss = model(**inputs).loss
            
            adv_loss.backward()
            
            pgd.step()
            model.zero_grad()

        pgd.restore()
        
        # --- ADVERSARIAL TRAINING BİTİŞ ---

        return loss

def main():
    # 1. Veri Hazırlama
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2) 

    raw_datasets = load_dataset("glue", TASK)

    def tokenize_function(examples):
        return tokenizer(examples["sentence"], truncation=True)

    tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
    tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["validation"]

    # 2. Eğitim Argümanları
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3, 
        per_device_train_batch_size=32,
        per_device_eval_batch_size=64,
        logging_steps=100,
    )

    # 3. Trainer'ı Başlat
    trainer = PGD_Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # 4. Eğitimi Başlat
    print(f"\n--- ADVERSARIAL TRAINING BAŞLIYOR (Görev: {TASK.upper()}) ---")
    trainer.train()
    print("--- ADVERSARIAL TRAINING TAMAMLANDI ---")

    # 5. En iyi modeli kaydet (final_pgd_model)
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_pgd_model"))
    print(f"Sağlamlaştırılmış model: {OUTPUT_DIR}/final_pgd_model konumuna kaydedildi.")


if __name__ == '__main__':
    main()