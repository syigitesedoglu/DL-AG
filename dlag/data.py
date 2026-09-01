"""
Training data builders for paper-protocol ARD:
- Prefer sst2_augmented.csv triples (clean + 2 adv) when present
- Mix additional clean IMDB to fight catastrophic forgetting
- AdvGLUE JSON loader for evaluation
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass
class PairExample:
    clean: str
    adv: str
    label: int


def load_advglue_task(path: str | Path, task: str = "sst2") -> List[Dict[str, Any]]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    key = task if task in data else task.replace("-", "")
    if key not in data and task == "sst-2":
        key = "sst2"
    items = data.get(key, [])
    out = []
    for it in items:
        if "sentence" in it:
            text = it["sentence"]
        elif "premise" in it and "hypothesis" in it:
            text = it["premise"] + " [SEP] " + it["hypothesis"]
        elif "question" in it and "sentence" in it:
            text = it["question"] + " [SEP] " + it["sentence"]
        elif "question1" in it and "question2" in it:
            text = it["question1"] + " [SEP] " + it["question2"]
        else:
            continue
        out.append({"text": text, "label": int(it["label"]), "raw": it})
    return out


def pairs_from_augmented_csv(csv_path: str | Path, group_size: int = 3) -> List[PairExample]:
    """
    augment_data.py wrote: original, aug1, aug2, original, ...
    group_size=3 -> one clean + up to two adv per group.
    """
    df = pd.read_csv(csv_path)
    rows = df.to_dict("records")
    pairs: List[PairExample] = []
    for i in range(0, len(rows) - 1, group_size):
        group = rows[i : i + group_size]
        if not group:
            continue
        clean = str(group[0]["sentence"])
        y = int(group[0]["label"])
        for g in group[1:]:
            adv = str(g["sentence"])
            if adv.strip() and adv != clean:
                pairs.append(PairExample(clean=clean, adv=adv, label=y))
        # also include identity pair so clean batches exist
        pairs.append(PairExample(clean=clean, adv=clean, label=y))
    return pairs


def mix_imdb_clean(pairs: List[PairExample], n: int, seed: int = 42) -> List[PairExample]:
    if n <= 0:
        return pairs
    try:
        from datasets import load_dataset
    except ImportError:
        return pairs
    ds = load_dataset("imdb", split="train").shuffle(seed=seed).select(range(min(n, 25000)))
    extra = [
        PairExample(clean=ex["text"], adv=ex["text"], label=int(ex["label"]))
        for ex in ds
    ]
    out = list(pairs) + extra
    rng = random.Random(seed)
    rng.shuffle(out)
    return out


class ARDPairDataset(Dataset):
    def __init__(
        self,
        pairs: Sequence[PairExample],
        tokenizer,
        max_length: int = 128,
        purify_adv=None,
        teacher_tokenizer=None,
    ):
        self.pairs = list(pairs)
        self.tokenizer = tokenizer  # student tokenizer
        self.teacher_tokenizer = teacher_tokenizer  # if None, reuse student (legacy)
        self.max_length = max_length
        self.purify_adv = purify_adv

    def __len__(self) -> int:
        return len(self.pairs)

    def _enc(self, tok, text: str) -> Dict[str, torch.Tensor]:
        return tok(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ex = self.pairs[idx]
        adv_text = self.purify_adv(ex.adv) if self.purify_adv else ex.adv
        t_tok = self.teacher_tokenizer or self.tokenizer

        # Student sees ADV; teacher soft targets from CLEAN (and omega from ADV)
        s_adv = self._enc(self.tokenizer, adv_text)
        t_clean = self._enc(t_tok, ex.clean)
        t_adv = self._enc(t_tok, adv_text)

        return {
            "adv_input_ids": s_adv["input_ids"].squeeze(0),
            "adv_attention_mask": s_adv["attention_mask"].squeeze(0),
            "teacher_clean_input_ids": t_clean["input_ids"].squeeze(0),
            "teacher_clean_attention_mask": t_clean["attention_mask"].squeeze(0),
            "teacher_adv_input_ids": t_adv["input_ids"].squeeze(0),
            "teacher_adv_attention_mask": t_adv["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex.label, dtype=torch.long),
        }
