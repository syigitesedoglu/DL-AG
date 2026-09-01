# Dual-Layer Adversarial Guard (DL-AG)

Private research code for adversarial robustness in NLP (DistilBERT student + Layer-1 text repair).

## Setup

```bash
pip install -r requirements_dlag.txt
```

## Core package

- `dlag/` — Layer 1 repair, attacks, ARD helpers, data utilities

## Training

- `train_ard.py` — Layer 2 robust distillation (ARD)
- `train_mixed_balanced.py` — balanced mixed training
- `train_layer2.py`, `train_distillation.py` — alternate training paths

## Evaluation

- `evaluator.py`, `final_validation.py`, `advglue_evaluate.py`
- `evaluate_dlag_final.py`, `evaluate_adaptive_dlag.py`

## Multi-seed (5 seeds: 42–46)

Paper / reviewer protocol:

```bash
# Full DL-AG pipeline per seed (train + eval)
python run_5seed_full.py

# Lighter 5-seed runner (balanced model path)
python run_5seed.py

# Entire reviewer suite (DL-AG + baselines + ablations + stats)
python run_reviewer_suite.py

# Individual pieces
python baselines_3seed.py
python pkd_ablation_3seed.py
python table9_ablation_3seed.py
python stats_tests.py
```

Colab: `DLAG_reviewer_5seed_colab.ipynb`, `run_colab_3seed_drive.py`

Checkpoints are written under `models/` (gitignored). Copy to Drive on Colab with `--save-root` / `--drive-root` flags in the runner scripts.

## Upstream

Based on the public [AdvGLUE](https://github.com/AI-secure/adversarial-glue) benchmark (`upstream` remote).
