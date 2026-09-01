#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Multi-seed orchestration for paper-protocol DL-AG (default: 3 seeds).

Per seed:
  1) train_ard.py  -> models/dl_ag_student_distilbert/seed_{SEED}
  2) evaluator.py  -> multiseed_results_full/seed_{SEED}_*.json

Epoch checkpoints + auto-resume are handled by train_ard.py.
Prefer --save-root / --results-dir on Google Drive for Colab disconnect safety.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SEEDS = [42, 43, 44, 45, 46]  # reviewer multi-seed (5)
RESULTS = ROOT / "multiseed_results_full"
MODEL_ROOT = ROOT / "models" / "dl_ag_student_distilbert"


def run(cmd):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(ROOT), check=True)


def epochs_done(model_dir: Path) -> int:
    meta = model_dir / "train_meta.json"
    if not meta.exists():
        return 0
    try:
        with open(meta, encoding="utf-8") as f:
            return int(json.load(f).get("epochs_done", 0))
    except Exception:
        return 0


def sync_to_drive(src: Path, dest: Path):
    """Copy src -> dest; no-op if paths are the same file/dir."""
    if not src.exists():
        return
    if os.path.normpath(str(src)) == os.path.normpath(str(dest)):
        return
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        try:
            if src.resolve() == dest.resolve():
                return
        except Exception:
            pass
    if src.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    else:
        shutil.copy2(src, dest)
    print(f"Synced -> {dest}", flush=True)


def summarize(paths, results_dir: Path):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            rows.append(json.load(f))
    vals = []
    for r in rows:
        s = r["summary"]
        if "advglue_l1l2" in s:
            vals.append(s["advglue_l1l2"]["RA"])
        elif "mean_attack_RA" in s:
            vals.append(s["mean_attack_RA"])
        elif "textfooler" in s:
            vals.append(s["textfooler"]["RA"])
    arr = np.array(vals, dtype=float)
    print("\n=== MULTI-SEED SUMMARY ===")
    if len(arr) == 0:
        print("No scores found.")
        return
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    print(f"primary RA: {arr.mean():.2f} ± {std:.2f}")
    for r, v in zip(rows, vals):
        print(f"  seed {r['seed']}: {v:.2f}")
    out = results_dir / "summary.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "mean": float(arr.mean()),
                "std": std,
                "values": vals,
                "seeds": [r["seed"] for r in rows],
            },
            f,
            indent=2,
        )
    print(f"Wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--full-attacks", action="store_true")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--imdb-clean", type=int, default=2000)
    ap.add_argument("--teacher", type=str, default="textattack/roberta-base-SST-2")
    ap.add_argument(
        "--student",
        type=str,
        default="distilbert-base-uncased",
        help="Use a finetuned SST-2 DistilBERT for stronger start.",
    )
    ap.add_argument("--max-train-pairs", type=int, default=0)
    ap.add_argument("--max-eval", type=int, default=0)
    ap.add_argument("--purify-train-adv", action="store_true")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--save-root", type=str, default=str(MODEL_ROOT))
    ap.add_argument(
        "--drive-mirror",
        type=str,
        default="",
        help="Extra copy of each epoch ckpt (usually unnecessary if --save-root is on Drive)",
    )
    ap.add_argument(
        "--fresh",
        action="store_true",
        help="Force retrain from scratch (pass --fresh to train_ard)",
    )
    ap.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Pass --freeze-backbone to train_ard (stability protocol)",
    )
    ap.add_argument(
        "--results-dir",
        type=str,
        default=str(RESULTS),
        help="Where to write seed_*.json eval outputs",
    )
    ap.add_argument(
        "--results-drive",
        type=str,
        default="",
        help="Also copy each eval json here (e.g. Drive results folder)",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    py = sys.executable
    mode = "full" if args.full_attacks else "advglue-only"
    result_paths = []
    save_root = Path(args.save_root)
    save_root.mkdir(parents=True, exist_ok=True)

    print(f"save-root={save_root}", flush=True)
    print(f"results-dir={results_dir}", flush=True)
    if args.drive_mirror:
        print(f"drive-mirror={args.drive_mirror}", flush=True)

    for seed in args.seeds:
        model_dir = save_root / f"seed_{seed}"
        done = epochs_done(model_dir)
        out = results_dir / f"seed_{seed}_sst2_{mode}.json"

        if not args.skip_train:
            if done >= args.epochs and not args.fresh:
                print(
                    f"Seed {seed}: train already complete ({done}/{args.epochs}); skip train.",
                    flush=True,
                )
            else:
                cmd = [
                    py,
                    "train_ard.py",
                    "--seed",
                    str(seed),
                    "--epochs",
                    str(args.epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--imdb-clean",
                    str(args.imdb_clean),
                    "--teacher",
                    args.teacher,
                    "--student",
                    args.student,
                    "--lr",
                    str(args.lr),
                    "--save-root",
                    str(save_root),
                ]
                if args.max_train_pairs:
                    cmd += ["--max-train-pairs", str(args.max_train_pairs)]
                if args.purify_train_adv:
                    cmd += ["--purify-train-adv"]
                if args.drive_mirror:
                    cmd += ["--drive-mirror", args.drive_mirror]
                if args.fresh:
                    cmd += ["--fresh"]
                if args.freeze_backbone:
                    cmd += ["--freeze-backbone"]
                run(cmd)

        if out.exists() and not args.fresh and epochs_done(model_dir) >= args.epochs:
            print(f"Seed {seed}: eval exists -> {out}; skip eval.", flush=True)
        else:
            cmd = [
                py,
                "evaluator.py",
                "--seed",
                str(seed),
                "--model-path",
                str(model_dir),
                "--mode",
                mode,
                "--task",
                "sst2",
                "--out",
                str(out),
            ]
            if args.max_eval:
                cmd += ["--max-examples", str(args.max_eval)]
            run(cmd)

        if args.results_drive:
            sync_to_drive(out, Path(args.results_drive) / out.name)

        result_paths.append(out)

    summarize(result_paths, results_dir)
    if args.results_drive:
        sync_to_drive(results_dir / "summary.json", Path(args.results_drive) / "summary.json")


if __name__ == "__main__":
    main()
