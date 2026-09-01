# -*- coding: utf-8 -*-
"""
Legacy 5-seed runner (balanced mixed training path).

Per seed:
  1) scripts/training/train_mixed_balanced.py
  2) scripts/evaluation/final_validation.py
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np

from repo_paths import (
    MODELS_BALANCED,
    REPO_ROOT,
    RESULTS_LEGACY,
    SCRIPT_FINAL_VALIDATION,
    SCRIPT_TRAIN_BALANCED,
    ensure_import_path,
)

ensure_import_path()

SEEDS = [42, 43, 44, 45, 46]
ROOT = REPO_ROOT
RESULTS_DIR = RESULTS_LEGACY


def run_cmd(cmd):
    print("\n>>>", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def summarize(results):
    adv = np.array([r["advglue_acc"] for r in results], dtype=float)
    imdb = np.array([r["imdb_acc"] for r in results], dtype=float)
    print("\n" + "=" * 60)
    print("5-SEED SUMMARY (mean ± std)")
    print("=" * 60)
    print(f"AdvGLUE: {adv.mean():.2f} ± {adv.std(ddof=1):.2f}")
    print(f"IMDB:    {imdb.mean():.2f} ± {imdb.std(ddof=1):.2f}")
    print("Per-seed:")
    for r in results:
        print(f"  seed {r['seed']}: AdvGLUE={r['advglue_acc']:.2f}, IMDB={r['imdb_acc']:.2f}")
    summary = {
        "seeds": SEEDS,
        "advglue_mean": float(adv.mean()),
        "advglue_std": float(adv.std(ddof=1)),
        "imdb_mean": float(imdb.mean()),
        "imdb_std": float(imdb.std(ddof=1)),
        "per_seed": results,
    }
    out = RESULTS_DIR / "summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    parser.add_argument("--skip-train", action="store_true", help="Only eval existing seed checkpoints")
    parser.add_argument("--skip-eval", action="store_true", help="Only train")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    collected = []

    for seed in args.seeds:
        model_dir = MODELS_BALANCED / f"seed_{seed}"
        result_path = RESULTS_DIR / f"seed_{seed}.json"

        if not args.skip_train:
            run_cmd([py, str(SCRIPT_TRAIN_BALANCED), "--seed", str(seed)])

        if not args.skip_eval:
            run_cmd([
                py,
                str(SCRIPT_FINAL_VALIDATION),
                "--seed",
                str(seed),
                "--model-path",
                str(model_dir),
                "--out",
                str(result_path),
            ])

        if result_path.exists():
            with open(result_path, "r", encoding="utf-8") as f:
                collected.append(json.load(f))

    if len(collected) >= 2:
        summarize(collected)
    else:
        print("Not enough result files to summarize yet.")


if __name__ == "__main__":
    main()
