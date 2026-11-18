# -*- coding: utf-8 -*-
"""Build adversarial-augmented SST-2 CSV for ARD training."""

from __future__ import annotations

import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

import pandas as pd
from datasets import load_dataset
from textattack.augmentation import EmbeddingAugmenter
from tqdm import tqdm

from repo_paths import AUGMENTED_CSV, DATA_DIR, ensure_import_path

ensure_import_path()

DATASET_NAME = "glue"
TASK = "sst2"
AUGMENTATIONS_PER_EXAMPLE = 2  # how many adversarial variants per sentence
PCT_WORDS_TO_SWAP = 0.1  # swap ~10% of words (keep semantics roughly intact)


def main():
    print(f"--- 1. Loading {TASK.upper()} dataset ---")
    dataset = load_dataset(DATASET_NAME, TASK, split="train")
    print(f"Original example count: {len(dataset)}")

    # TextAttack embedding augmenter: replace words with embedding-space neighbors
    augmenter = EmbeddingAugmenter(
        pct_words_to_swap=PCT_WORDS_TO_SWAP,
        transformations_per_example=AUGMENTATIONS_PER_EXAMPLE,
    )

    augmented_sentences = []
    augmented_labels = []

    print("--- 2. Starting adversarial augmentation ---")
    start_time = time.time()

    for example in tqdm(dataset):
        original_text = example["sentence"]
        label = example["label"]

        # Keep the original example
        augmented_sentences.append(original_text)
        augmented_labels.append(label)

        # Generate adversarial variants
        try:
            new_examples = augmenter.augment(original_text)
            for aug_text in new_examples:
                augmented_sentences.append(aug_text)
                augmented_labels.append(label)  # label unchanged (robustness objective)
        except Exception:
            pass  # skip failed augmentations

    end_time = time.time()
    print(f"Augmentation finished in {end_time - start_time:.2f} seconds.")
    print(f"New dataset size: {len(augmented_sentences)}")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"sentence": augmented_sentences, "label": augmented_labels})
    df.to_csv(AUGMENTED_CSV, index=False)
    print(f"--- 3. Saved dataset to '{AUGMENTED_CSV.name}' ({AUGMENTED_CSV}). ---")


if __name__ == "__main__":
    main()
