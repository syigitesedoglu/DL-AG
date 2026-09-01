#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Full reviewer response suite (default: 5 seeds).

Order:
  1) DL-AG ARD + eval          (run_5seed_full.py)
  2) BERT-Base + DistilBERT    (baselines_3seed.py)
  3) PKD Logit-Only vs Skip    (pkd_ablation_3seed.py)
  4) Table-9 L1/L2/Hybrid      (table9_ablation_3seed.py)
  5) t-tests / CIs             (stats_tests.py)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from repo_paths import (
    MODELS_DLAG_STUDENT,
    REPO_ROOT,
    RESULTS_FULL,
    SCRIPT_BASELINES,
    SCRIPT_PKD,
    SCRIPT_RUN_5SEED_FULL,
    SCRIPT_STATS,
    SCRIPT_TABLE9,
    ensure_import_path,
)

ensure_import_path()

SEEDS = [42, 43, 44, 45, 46]
ROOT = REPO_ROOT
RESULTS = RESULTS_FULL
DLAG_ROOT = MODELS_DLAG_STUDENT


def run(cmd):
    print("\n>>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def dlag_done(seeds, model_root: Path, results_dir: Path):
    for s in seeds:
        model = model_root / f"seed_{s}"
        evals = list(results_dir.glob(f"seed_{s}_sst2_*.json"))
        meta = model / "train_meta.json"
        if not model.exists() or not evals or not meta.exists():
            return False
    return True


def wait_dlag(seeds, model_root: Path, results_dir: Path, poll_s=120):
    print("Waiting for DL-AG train+eval to finish...")
    while not dlag_done(seeds, model_root, results_dir):
        print(f"  not ready yet; sleep {poll_s}s", flush=True)
        time.sleep(poll_s)
    print("DL-AG complete.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--skip-dlag", action="store_true")
    ap.add_argument("--wait-dlag", action="store_true")
    ap.add_argument(
        "--max-train-pairs",
        type=int,
        default=0,
        help="0=all pairs (more stable across seeds; advisor wants ~±0.5)",
    )
    ap.add_argument("--dlag-epochs", type=int, default=5)
    ap.add_argument(
        "--drive-root",
        type=str,
        default="",
        help="e.g. /content/drive/MyDrive/dlag_reviewer_3seed — models+results live on Drive",
    )
    ap.add_argument("--fresh", action="store_true", help="Retrain DL-AG from scratch")
    args = ap.parse_args()

    py = sys.executable
    seeds = args.seeds

    if args.drive_root:
        drive = Path(args.drive_root)
        model_root = drive / "models" / "dl_ag_student_distilbert"
        results_dir = drive / "multiseed_results_full"
    else:
        model_root = DLAG_ROOT
        results_dir = RESULTS

    model_root.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"DL-AG model_root={model_root}", flush=True)
    print(f"DL-AG results_dir={results_dir}", flush=True)

    if args.wait_dlag:
        wait_dlag(seeds, model_root, results_dir)
    elif not args.skip_dlag:
        cmd = [
            py,
            str(SCRIPT_RUN_5SEED_FULL),
            "--seeds",
            *[str(s) for s in seeds],
            "--epochs",
            str(args.dlag_epochs),
            "--batch-size",
            "8",
            "--imdb-clean",
            "5000",
            "--student",
            "distilbert-base-uncased-finetuned-sst-2-english",
            "--teacher",
            "textattack/roberta-base-SST-2",
            "--save-root",
            str(model_root),
            "--results-dir",
            str(results_dir),
            "--results-drive",
            str(results_dir),
        ]
        if args.max_train_pairs and args.max_train_pairs > 0:
            cmd += ["--max-train-pairs", str(args.max_train_pairs)]
        if args.fresh:
            cmd += ["--fresh"]
        run(cmd)
    else:
        print("Skipping DL-AG train (assumed done).")

    run([
        py,
        str(SCRIPT_BASELINES),
        "--seeds",
        *[str(s) for s in seeds],
        "--out",
        str(results_dir / "baselines_summary.json"),
    ])
    run([
        py,
        str(SCRIPT_PKD),
        "--seeds",
        *[str(s) for s in seeds],
        "--out",
        str(results_dir / "pkd_ablation_summary.json"),
    ])
    run([
        py,
        str(SCRIPT_TABLE9),
        "--seeds",
        *[str(s) for s in seeds],
        "--dlag-root",
        str(model_root),
        "--out",
        str(results_dir / "table9_ablation_summary.json"),
    ])
    run([
        py,
        str(SCRIPT_STATS),
        "--results-dir",
        str(results_dir),
        "--baselines",
        str(results_dir / "baselines_summary.json"),
        "--pkd",
        str(results_dir / "pkd_ablation_summary.json"),
        "--out",
        str(results_dir / "significance_tests.json"),
    ])
    print("\nREVIEWER SUITE COMPLETE")
    print(f"See {results_dir}")


if __name__ == "__main__":
    main()
