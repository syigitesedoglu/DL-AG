import json
import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from tqdm import tqdm
import time

# PGD'yi bırakıp Spell Checker'ı ekliyoruz
from spellchecker import SpellChecker 
spell = SpellChecker()

# run_evaluation.py dosyasındaki apply_spell_correction fonksiyonunu bulun

# --- Yeni Savunma Fonksiyonu ---
def apply_spell_correction(text):
    """Metindeki kelimeleri kontrol eder ve yazım hatalarını düzeltir."""
    if not isinstance(text, str) or not text:
        return "" # Metin yoksa veya geçerli değilse boş döndür
        
    words = text.split()
    
    corrected_words = []
    for word in words:
        # Düzeltmeyi dene
        correction = spell.correction(word)
        
        # NoneType kontrolü: Eğer düzeltme yapılamıyorsa (None gelirse), orijinal kelimeyi kullan.
        if correction is None:
            corrected_words.append(word)
        else:
            corrected_words.append(correction)

    return " ".join(corrected_words)
# -------------------------------


# GPU kontrolü
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CONFIGURATION ---
HUGGING_FACE_TOKEN = "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" # TOKEN'ınızı buraya yapıştırın!
DEV_FILE_PATH = 'dev.json' 
PREDICTIONS_FILE_PATH = 'predictions.json'
batch_size = 32

# HUGGING FACE MODELLERİ (Baseline'daki aynı modeller)
TASK_MODELS = {
    'sst2': 'distilbert-base-uncased-finetuned-sst-2-english',
    'qqp': 'textattack/bert-base-uncased-qqp',
    'mnli': 'textattack/bert-base-uncased-mnli', 
    'qnli': 'textattack/bert-base-uncased-qnli',
    'rte': 'textattack/bert-base-uncased-rte',
}

# GLUE görevlerinin veri formatları
TASK_TO_KEYS = {
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"), 
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None), 
}
# -----------------------------------------------------------------


def run_task_evaluation(task_name, model, tokenizer, data, device, final_predictions):
    """Belirli bir AdvGLUE görevi için tahminleri çalıştırır."""
    
    task_data = data[task_name]
    df = pd.DataFrame(task_data)
    
    key_name = task_name if task_name != 'mnli-mm' else 'mnli'
    sentence1_key, sentence2_key = TASK_TO_KEYS[key_name]
    
    task_predictions = [] 
    
    for i in tqdm(range(0, len(df), batch_size), desc=f"{task_name.upper()} Tahminleri (SPELL CHECK)"):
        batch = df.iloc[i:i + batch_size]
        
        # --- SAVUNMA KATMANI: Girdiye Spell Correction Uygula ---
        sentences1_raw = list(batch[sentence1_key])
        sentences1 = [apply_spell_correction(s) for s in sentences1_raw] # Savunma uygulandı
        
        if sentence2_key:
            sentences2_raw = list(batch[sentence2_key])
            sentences2 = [apply_spell_correction(s) for s in sentences2_raw] # Savunma uygulandı
            
            encoded_input = tokenizer(
                sentences1, sentences2, padding=True, truncation=True, return_tensors='pt'
            ).to(device)
        else:
            # Tek cümle girişi (SST-2)
            encoded_input = tokenizer(
                sentences1, padding=True, truncation=True, return_tensors='pt'
            ).to(device)

        # Tahmin yap
        with torch.no_grad():
            outputs = model(**encoded_input)
        
        logits = outputs.logits
        predictions_np = torch.argmax(logits, dim=-1).cpu().numpy()
        
        task_predictions.extend(predictions_np.tolist())

    final_predictions[task_name] = task_predictions
    print(f"\n{task_name.upper()} görevi için {len(task_predictions)} tahmin yapıldı.")


def main():
    # GPU/CPU ayarı
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Kullanılan cihaz: {device}")

    # Veriyi yükle
    try:
        with open(DEV_FILE_PATH, 'r', encoding='utf-8') as f:
            advglue_data = json.load(f)
        print(f"{DEV_FILE_PATH} başarıyla yüklendi. Toplam görev: {len(advglue_data.keys())}")
    except FileNotFoundError:
        print(f"HATA: {DEV_FILE_PATH} bulunamadı.")
        return

    final_predictions = {}
    
    tasks_to_run = [t for t in TASK_MODELS.keys() if t in advglue_data.keys()] 

    for task_name in tasks_to_run:
        if task_name == 'mnli-mm':
            continue 

        model_name = TASK_MODELS[task_name]
        print(f"\n--- BAŞLANIYOR: {task_name.upper()} ({model_name}) ---")
        
        start_time = time.time()
        
        # MODEL YÜKLEME (PGD'deki gibi değil, sadece tahmin yapıyoruz)
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=HUGGING_FACE_TOKEN)
            model = AutoModelForSequenceClassification.from_pretrained(model_name, token=HUGGING_FACE_TOKEN)
            model.to(device)
            model.eval()
        except Exception as e:
            print(f"HATA: {task_name} modeli yüklenemedi. Atlanıyor. Detay: {e}")
            continue

        # Tahminleri Çalıştır
        run_task_evaluation(task_name, model, tokenizer, advglue_data, device, final_predictions)
        
        # MNLI-mm için de tahminleri kaydet
        if task_name == 'mnli':
            print("MNLI-mm (Mismatched) için aynı model ile tahminler kaydediliyor...")
            run_task_evaluation('mnli-mm', model, tokenizer, advglue_data, device, final_predictions)

        end_time = time.time()
        print(f"--- BİTTİ: {task_name.upper()}. Süre: {end_time - start_time:.2f} saniye. ---")


    # Nihai Tahmin Dosyasını Kaydet
    with open(PREDICTIONS_FILE_PATH, 'w') as f:
        json.dump(final_predictions, f, indent=4)
    
    print(f"\nTÜM GÖREVLER TAMAMLANDI. Tahmin dosyası '{PREDICTIONS_FILE_PATH}' olarak kaydedildi.")


if __name__ == '__main__':
    main()