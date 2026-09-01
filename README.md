# Dual-Layer Adversarial Guard (DL-AG)

Private research code for adversarial robustness in NLP.

## Setup

```bash
pip install -r requirements_dlag.txt
```

Generate augmented training pairs (once):

```bash
python augment_data.py
```

## Layout

| Path | Role |
|------|------|
| `dlag/` | Layer 1 purifier, attacks, ARD training helpers |
| `train_ard.py` | Layer 2 robust distillation |
| `evaluator.py` | Paper-protocol evaluation (attacks + AdvGLUE) |
| `run_5seed_full.py` | 5-seed train + eval (seeds 42–46) |
| `run_reviewer_suite.py` | Full reviewer pipeline (DL-AG + baselines + ablations + stats) |

## Multi-seed / reviewer protocol

```bash
python run_5seed_full.py
python run_reviewer_suite.py
```

Individual steps: `baselines_3seed.py`, `pkd_ablation_3seed.py`, `table9_ablation_3seed.py`, `stats_tests.py`

Legacy balanced-training path: `run_5seed.py` (uses `train_mixed_balanced.py` + `final_validation.py`)

Colab: `DLAG_reviewer_5seed_colab.ipynb`

Checkpoints under `models/` and results under `multiseed_results_full/` are gitignored.

## Upstream

AdvGLUE benchmark: https://github.com/AI-secure/adversarial-glue (`upstream` remote)
