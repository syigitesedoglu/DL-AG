#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Stability multi-seed run (lower seed variance).

Protocol vs paper/full reviewer run:
  - lr = 5e-6
  - epochs = 3
  - freeze DistilBERT backbone (classifier + projector only)
  - separate Drive folder (does NOT overwrite dlag_reviewer_3seed)

Usage (Colab):
  python run_colab_stability.py --seeds 42 43 44 45 46 --fresh
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEFAULT_DRIVE = "/content/drive/MyDrive/dlag_stability_5seed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drive-root", type=str, default=DEFAULT_DRIVE)
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--max-train-pairs", type=int, default=0)
    args = ap.parse_args()

    drive = Path(args.drive_root)
    save_root = drive / "models" / "dl_ag_student_distilbert"
    results = drive / "multiseed_results_stability"
    save_root.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "run_5seed_full.py",
        "--seeds",
        *[str(s) for s in args.seeds],
        "--epochs",
        str(args.epochs),
        "--batch-size",
        "8",
        "--imdb-clean",
        "5000",
        "--lr",
        str(args.lr),
        "--student",
        "distilbert-base-uncased-finetuned-sst-2-english",
        "--teacher",
        "textattack/roberta-base-SST-2",
        "--save-root",
        str(save_root),
        "--results-dir",
        str(results),
        "--freeze-backbone",
    ]
    if args.max_train_pairs and args.max_train_pairs > 0:
        cmd += ["--max-train-pairs", str(args.max_train_pairs)]
    if args.fresh:
        cmd += ["--fresh"]

    print("STABILITY PROTOCOL", flush=True)
    print("  lr=", args.lr, "epochs=", args.epochs, "freeze-backbone=ON", flush=True)
    print("  drive=", drive, flush=True)
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)
    print("\nDONE. Stability results under:", results)


if __name__ == "__main__":
    main()
