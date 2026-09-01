import torch
import pandas as pd
import os
import time
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

# ==========================================
#  AYARLAR
# ==========================================
MODEL_PATH = "./layer2_model_boosted/final_robust_model_boosted" 
TEST_DATA_PATH = "dev.json"   # AdvGLUE Dosyası
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def evaluate_model(model, data_loader, device):
    model = model.eval()
    predictions = []
    real_values = []

    print(f"   İşleniyor... (Batch Size: {data_loader.batch_size})")

    with torch.no_grad():
        for i, d in enumerate(data_loader):
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            _, preds = torch.max(outputs.logits, dim=1)

            predictions.extend(preds.cpu().tolist())
            real_values.extend(labels.cpu().tolist())
            
            if i % 5 == 0:
                print(".", end="", flush=True)

    print() 
    
    # METRİKLERİ HESAPLA
    acc = accuracy_score(real_values, predictions)
    
    # Detaylı Rapor (Precision, Recall, F1)
    # target_names=['Negative', 'Positive'] SST-2 için genelde böyledir (0=Neg, 1=Pos)
    report = classification_report(real_values, predictions, target_names=['Negative (0)', 'Positive (1)'])
    
    return acc, report

def load_advglue_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dosya bulunamadı: {file_path}")

    print(f"--- AdvGLUE JSON Dosyası Okunuyor: {file_path} ---")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    if 'sst2' not in data:
        raise ValueError("JSON dosyasında 'sst2' anahtarı bulunamadı!")
    
    adv_data = data['sst2']
    df = pd.DataFrame(adv_data)
    
    print(" SST-2 verileri başarıyla çekildi.")
    
    if 'sentence' not in df.columns or 'label' not in df.columns:
        raise ValueError(f"Sütunlar eksik! Bulunanlar: {df.columns}")

    return df

def main():
    print("="*50)
    print(" DETAYLI ADVGLUE TESTİ (ROBUSTNESS CHECK)")
    print(f" Cihaz: {DEVICE}")
    print(f" Model: {MODEL_PATH}")
    print("="*50)

    # 1. Veri Yükleme
    try:
        df = load_advglue_data(TEST_DATA_PATH)
    except Exception as e:
        print(f"\n HATA: {e}")
        return

    # 2. Model Yükleme
    print(f"--- Model Yükleniyor... ---")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
    except Exception as e:
        print(f" Model yükleme hatası: {e}")
        return

    # 3. Hazırlık
    test_dataset = TextDataset(
        texts=df.sentence.to_numpy(),
        labels=df.label.to_numpy(),
        tokenizer=tokenizer
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0
    )

    print(f" Toplam Test Verisi: {len(df)} adet")
    print("--- Test Başlıyor... ---")
    
    start_time = time.time()
    
    # 4. Skor Hesapla
    accuracy, report = evaluate_model(model, test_data_loader, DEVICE)
    
    end_time = time.time()
    duration = end_time - start_time

    print("\n" + "="*50)
    print(f" SONUÇLAR")
    print("="*50)
    print(f" ROBUST ACCURACY: %{accuracy*100:.2f}")
    print("-" * 30)
    print(" DETAYLI METRİKLER (Recall, Precision, F1):")
    print(report)
    print("-" * 30)
    print(f" Test Süresi: {duration:.1f} saniye")
    print("="*50)

if __name__ == "__main__":
    main()