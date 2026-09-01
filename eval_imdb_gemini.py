# -*- coding: utf-8 -*-
"""
DL-AG FINAL CHECK: GENERALIZATION TEST (Gemini / Dec 2025)

Model: Teacher-Student checkpoint trained on AdvGLUE (dl_ag_final_student)
Goal: IMDB accuracy should stay above ~80% so AdvGLUE ~75% is not pure overfitting.

Pairs with: train_kd_gemini_advglue.py
"""

import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_PATH = "./dl_ag_final_student"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_SAMPLES = 500


def main():
    print("=" * 60)
    print("DL-AG IMDB GENELLEME (saglamlama)")
    print("Amac: AdvGLUE yuksekken IMDB'nin %80 ustunde kalmasi")
    print("=" * 60)

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Model yuklenemedi ({MODEL_PATH}): {e}")
        return

    print("IMDB test (500 ornek)...")
    dataset = load_dataset("imdb", split="test").shuffle(seed=42).select(range(N_SAMPLES))

    preds, refs = [], []
    for example in tqdm(dataset):
        # Original Gemini script: Layer1 closed for speed; raw IMDB text
        text = example["text"]
        label = int(example["label"])
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=256
        ).to(DEVICE)
        with torch.no_grad():
            pred = int(torch.argmax(model(**inputs).logits, dim=-1).item())
        preds.append(pred)
        refs.append(label)

    acc = accuracy_score(refs, preds) * 100
    print("=" * 40)
    print(f"IMDB SKORU: %{acc:.2f}")
    print("=" * 40)
    if acc > 80:
        print("OK: dil yetenegi korunmus (son_rapor: ~81.6 bandi)")
    elif acc > 70:
        print("Orta: kabul edilebilir dusus")
    else:
        print("Uyari: catastrophic forgetting riski (son_rapor ara faz ~50)")


if __name__ == "__main__":
    main()
