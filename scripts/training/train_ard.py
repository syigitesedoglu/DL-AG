#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DL-AG Layer-2 ARD training (paper protocol).

Teacher sees CLEAN text; student sees ADV text.
Loss: (1-a) CE + a T^2 omega KL + hidden alignment.

Default paths assume adversarial-glue repo root.
RoBERTa-Large is heavy — use --teacher for a large SST-2 checkpoint on GPU/Colab.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from dlag.ard import FeatureProjector, ard_total_loss, default_layer_map
from dlag.data import ARDPairDataset, mix_imdb_clean, pairs_from_augmented_csv
from dlag.layer1 import Layer1Purifier
from dlag.utils import get_device, make_generator, set_seed
from repo_paths import AUGMENTED_CSV, MODELS_DLAG_STUDENT, ensure_import_path

ensure_import_path()


def parse_args():
    p = argparse.ArgumentParser(description="DL-AG ARD training (paper protocol)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--teacher",
        type=str,
        default="textattack/roberta-base-SST-2",
        help="Paper uses RoBERTa-Large SST-2 teacher. Override with a large HF SST-2 checkpoint on GPU.",
    )
    p.add_argument("--student", type=str, default="distilbert-base-uncased")
    p.add_argument("--augmented-csv", type=str, default=str(AUGMENTED_CSV))
    p.add_argument("--imdb-clean", type=int, default=2000, help="Extra clean IMDB rows (forgetting control)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--T", type=float, default=2.0)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--max-train-pairs", type=int, default=0, help="0 = use all pairs")
    p.add_argument("--purify-train-adv", action="store_true", help="Run Layer1 on adv texts during training")
    p.add_argument("--align-weight", type=float, default=0.1)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--save-root", type=str, default=str(MODELS_DLAG_STUDENT))
    p.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing checkpoint and train from scratch",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="(legacy) Resume is default; use --fresh to disable",
    )
    p.add_argument(
        "--drive-mirror",
        type=str,
        default="",
        help="Optional path (e.g. /content/drive/MyDrive/dlag_ckpts) to copy checkpoint after each epoch",
    )
    p.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze DistilBERT encoder; train classifier(+pre_classifier) and projector only (stability run)",
    )
    return p.parse_args()


def _save_ckpt(out_dir, student, student_tok, projector, optim, meta, drive_mirror=""):
    os.makedirs(out_dir, exist_ok=True)
    student.save_pretrained(out_dir)
    student_tok.save_pretrained(out_dir)
    torch.save(projector.state_dict(), os.path.join(out_dir, "feature_projector.pt"))
    torch.save(optim.state_dict(), os.path.join(out_dir, "optimizer.pt"))
    with open(os.path.join(out_dir, "train_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Checkpoint saved -> {out_dir} (epochs_done={meta.get('epochs_done')})")
    if drive_mirror:
        import shutil

        dest = os.path.join(drive_mirror, Path(out_dir).name)
        os.makedirs(drive_mirror, exist_ok=True)
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        shutil.copytree(out_dir, dest)
        print(f"Mirrored -> {dest}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print("=" * 60)
    print("DL-AG ARD TRAINING (paper protocol)")
    print(f"seed={args.seed} device={device}")
    print(f"teacher={args.teacher}")
    print(f"student={args.student}")
    print("=" * 60)

    if not Path(args.augmented_csv).exists():
        raise FileNotFoundError(
            f"Missing {args.augmented_csv}. Run: python scripts/data/augment_data.py"
        )

    pairs = pairs_from_augmented_csv(args.augmented_csv)
    pairs = mix_imdb_clean(pairs, n=args.imdb_clean, seed=args.seed)
    if args.max_train_pairs and args.max_train_pairs > 0:
        pairs = pairs[: args.max_train_pairs]
    print(f"Training pairs: {len(pairs)}")

    out_dir = args.out or os.path.join(args.save_root, f"seed_{args.seed}")
    start_epoch = 0
    meta_path = os.path.join(out_dir, "train_meta.json")

    student_tok = AutoTokenizer.from_pretrained(args.student)
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        args.teacher, output_hidden_states=True
    )

    want_resume = not args.fresh
    resume_ok = (
        want_resume
        and os.path.isdir(out_dir)
        and os.path.exists(os.path.join(out_dir, "config.json"))
        and os.path.exists(meta_path)
    )
    if resume_ok:
        with open(meta_path, encoding="utf-8") as f:
            old_meta = json.load(f)
        start_epoch = int(old_meta.get("epochs_done", 0))
        print(f"RESUME from {out_dir} (epochs_done={start_epoch})")
        student = AutoModelForSequenceClassification.from_pretrained(
            out_dir, output_hidden_states=True
        )
        student_tok = AutoTokenizer.from_pretrained(out_dir)
    else:
        if args.fresh:
            print("FRESH start (--fresh): ignoring any existing checkpoint.")
        elif want_resume:
            print("No valid checkpoint found; training from scratch.")
        student = AutoModelForSequenceClassification.from_pretrained(
            args.student, num_labels=2, output_hidden_states=True
        )

    teacher.to(device)
    student.to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    if args.freeze_backbone:
        n_frozen = 0
        for name, p in student.named_parameters():
            # DistilForSequenceClassification: distilbert.* frozen; pre_classifier + classifier train
            if name.startswith("classifier") or name.startswith("pre_classifier"):
                p.requires_grad_(True)
            else:
                p.requires_grad_(False)
                n_frozen += p.numel()
        print(f"Freeze backbone ON: frozen_params≈{n_frozen:,}; train classifier/pre_classifier + projector")

    t_hidden = teacher.config.hidden_size
    s_hidden = student.config.hidden_size
    t_layers = teacher.config.num_hidden_layers
    layer_map = default_layer_map(t_layers, student.config.num_hidden_layers)
    projector = FeatureProjector(layer_map, s_hidden, t_hidden).to(device)
    if resume_ok and os.path.exists(os.path.join(out_dir, "feature_projector.pt")):
        projector.load_state_dict(
            torch.load(os.path.join(out_dir, "feature_projector.pt"), map_location=device)
        )

    purify_fn = None
    if args.purify_train_adv:
        print("Building Layer1 purifier for training-time adv cleanup...")
        purifier = Layer1Purifier(delta=2, lam=0.7)
        purify_fn = purifier.purify

    ds = ARDPairDataset(
        pairs,
        student_tok,
        max_length=args.max_length,
        purify_adv=purify_fn,
        teacher_tokenizer=teacher_tok,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        generator=make_generator(args.seed),
    )

    flip_teacher = False
    with torch.no_grad():
        n_ok = n_flip = 0
        for i in range(min(32, len(ds))):
            ex = ds[i]
            logits = teacher(
                ex["teacher_clean_input_ids"].unsqueeze(0).to(device),
                attention_mask=ex["teacher_clean_attention_mask"].unsqueeze(0).to(device),
            ).logits[0]
            y = int(ex["labels"].item())
            pred = int(logits.argmax().item())
            n_ok += int(pred == y)
            n_flip += int((1 - pred) == y)
        flip_teacher = n_flip > n_ok
        print(f"Teacher label probe: flip_teacher_logits={flip_teacher} (ok={n_ok}, flip={n_flip})")

    trainable = [p for p in student.parameters() if p.requires_grad] + list(projector.parameters())
    print(f"Optimizer tensors: {len(trainable)} (lr={args.lr})")
    optim = AdamW(trainable, lr=args.lr)
    if resume_ok and os.path.exists(os.path.join(out_dir, "optimizer.pt")):
        try:
            optim.load_state_dict(
                torch.load(os.path.join(out_dir, "optimizer.pt"), map_location=device)
            )
        except Exception as e:
            print(f"optimizer state load skipped: {e}")

    if start_epoch >= args.epochs:
        print(f"Already completed {start_epoch}/{args.epochs} epochs. Nothing to do.")
        return

    student.train()
    for epoch in range(start_epoch, args.epochs):
        running = 0.0
        loop = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in loop:
            adv_ids = batch["adv_input_ids"].to(device)
            adv_mask = batch["adv_attention_mask"].to(device)
            t_clean_ids = batch["teacher_clean_input_ids"].to(device)
            t_clean_mask = batch["teacher_clean_attention_mask"].to(device)
            t_adv_ids = batch["teacher_adv_input_ids"].to(device)
            t_adv_mask = batch["teacher_adv_attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optim.zero_grad()
            with torch.no_grad():
                t_clean = teacher(input_ids=t_clean_ids, attention_mask=t_clean_mask)
                t_adv = teacher(input_ids=t_adv_ids, attention_mask=t_adv_mask)
                t_clean_logits = t_clean.logits
                t_adv_logits = t_adv.logits
                if flip_teacher:
                    t_clean_logits = t_clean_logits.flip(dims=[-1])
                    t_adv_logits = t_adv_logits.flip(dims=[-1])

            s_adv = student(input_ids=adv_ids, attention_mask=adv_mask)

            loss, stats = ard_total_loss(
                student_logits_adv=s_adv.logits,
                teacher_logits_clean=t_clean_logits,
                teacher_logits_adv=t_adv_logits,
                labels=labels,
                student_hidden=s_adv.hidden_states,
                teacher_hidden=t_clean.hidden_states,
                projector=projector,
                T=args.T,
                alpha=args.alpha,
                beta=args.beta,
                align_weight=args.align_weight,
            )
            loss.backward()
            optim.step()
            running += stats["total"]
            loop.set_postfix(loss=f"{stats['total']:.4f}", omega=f"{stats['omega_mean']:.3f}")

        mean_loss = running / max(len(loader), 1)
        print(f"Epoch {epoch+1} mean loss={mean_loss:.4f}")

        meta = {
            "seed": args.seed,
            "teacher": args.teacher,
            "student": args.student,
            "epochs": args.epochs,
            "epochs_done": epoch + 1,
            "last_mean_loss": mean_loss,
            "T": args.T,
            "alpha": args.alpha,
            "beta": args.beta,
            "lr": args.lr,
            "imdb_clean": args.imdb_clean,
            "n_pairs": len(pairs),
            "layer_map": layer_map,
            "flip_teacher": flip_teacher,
            "freeze_backbone": bool(args.freeze_backbone),
            "protocol": (
                "stability-ARD freeze-backbone + low-lr"
                if args.freeze_backbone
                else "paper-ARD + epoch checkpoints + optional resume"
            ),
        }
        _save_ckpt(out_dir, student, student_tok, projector, optim, meta, args.drive_mirror)

    print(f"Training complete -> {out_dir}")


if __name__ == "__main__":
    main()
