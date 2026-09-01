#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stronger DL-AG SST-2 campaign (score-lift protocol).

Diff vs quick 15k run:
  - ALL augmented pairs (no max-train-pairs cap)
  - 5 epochs
  - Student starts from SST-2 finetuned DistilBERT
  - Train-time Layer1 on adversarial texts
  - More IMDB clean mix
  - Separate save root so weak-run checkpoints are not overwritten

Does NOT fabricate scores — only strengthens training/eval protocol.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    py = sys.executable
    cmd = [
        py,
        "run_5seed_full.py",
        "--seeds",
        "42",
        "43",
        "44",
        "--epochs",
        "5",
        "--batch-size",
        "8",
        "--imdb-clean",
        "5000",
        "--max-train-pairs",
        "0",
        "--teacher",
        "textattack/roberta-base-SST-2",
        "--student",
        "distilbert-base-uncased-finetuned-sst-2-english",
        "--purify-train-adv",
        "--lr",
        "2e-5",
        "--save-root",
        str(ROOT / "models" / "dl_ag_student_strong"),
        "--results-dir",
        str(ROOT / "multiseed_results_strong"),
    ]
    print("STRONG DL-AG:", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    print("\nSTRONG DL-AG COMPLETE")
    print("Results: multiseed_results_strong/")
    print("Models: models/dl_ag_student_strong/seed_*")


if __name__ == "__main__":
    main()
