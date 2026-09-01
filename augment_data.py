# -*- coding: utf-8 -*-
"""
Created on Mon Dec  8 21:49:44 2025

@author: yigit
"""

import pandas as pd
from datasets import load_dataset
from textattack.augmentation import EmbeddingAugmenter
from tqdm import tqdm
import time

DATASET_NAME = "glue"
TASK = "sst2"
AUGMENTATIONS_PER_EXAMPLE = 2 # Her cümle için kaç tane saldırgan örnek üretilsin?
PCT_WORDS_TO_SWAP = 0.1 # Cümlenin %10'unu değiştirelim (Anlamsal koruma için)

def main():
    print(f"---1. {TASK.upper()} Veri Seti Yükleniyor ---")
    dataset = load_dataset(DATASET_NAME, TASK, split="train")
    print(f"Orijinal Veri Sayısı: {len(dataset)}")

    # TextAttack Embedding Augmenter (Anlamsal Değişiklik)
    # Bu, kelimeleri word embedding uzayındaki komşularıyla değiştirir.
    augmenter = EmbeddingAugmenter(pct_words_to_swap = PCT_WORDS_TO_SWAP, transformations_per_example=AUGMENTATIONS_PER_EXAMPLE)

    augmented_sentences = []
    augmented_labels = []

    print("--- 2. Veri Büyütme (Adversarial Augmentation) Başlıyor ---")
    start_time = time.time()

    for example in tqdm(dataset):
        original_text = example['sentence']
        label = example['label']
        
        # 1. Orijinal veriyi ekle
        augmented_sentences.append(original_text)
        augmented_labels.append(label)
        
        # 2. Augmentasyon yap (Saldırgan örnek üret)
        try:
            # Augmenter bir liste döndürür
            new_examples = augmenter.augment(original_text)
            for aug_text in new_examples:
                augmented_sentences.append(aug_text)
                augmented_labels.append(label) # Etiket değişmez (Robustness hedefi)
        except Exception as e:
            pass # Hata olursa geç

    end_time = time.time()
    print(f"Büyütme Tamamlandı. Süre: {end_time - start_time:.2f} saniye.")
    print(f"Yeni Veri Seti Boyutu: {len(augmented_sentences)}")

    # Veriyi Kaydet (CSV olarak)
    df = pd.DataFrame({'sentence': augmented_sentences, 'label': augmented_labels})
    df.to_csv('sst2_augmented.csv', index=False)
    print("--- 3. Veri 'sst2_augmented.csv' olarak kaydedildi. ---")

if __name__ == '__main__':
    main()