# Dual-Layer Adversarial Guard (DL-AG)

Private repository for paper experiments, the 5-seed protocol, and the reviewer suite.

**GitHub:** https://github.com/syigitesedoglu/DL-AG (private)

---

## Quick start

Run all commands from the **repository root**.

```bash
pip install -r requirements_dlag.txt

# 1) Build augmented SST-2 pairs (once)
python scripts/data/augment_data.py

# 2) Paper protocol — 5-seed train + eval
python scripts/multiseed/run_5seed_full.py

# 3) Full reviewer pipeline (DL-AG + baselines + ablations + stats)
python scripts/multiseed/run_reviewer_suite.py
```

Colab: `notebooks/DLAG_reviewer_5seed_colab.ipynb`

---

## Repository layout

```
DL-AG/
├── dlag/                          # Core Python package
├── scripts/
│   ├── data/                      # Data preparation
│   ├── training/                  # Model training
│   ├── evaluation/                # Testing & metrics
│   ├── multiseed/                 # Orchestration (5 seeds)
│   └── ablations/                 # Table / ablation experiments
├── data/                          # Static data (dev.json, generated CSV)
├── notebooks/                     # Colab notebooks
├── models/                        # Checkpoints (gitignored)
├── multiseed_results_full/        # JSON results (gitignored)
├── repo_paths.py                  # Shared path constants
└── requirements_dlag.txt
```

---

## `dlag/` — core package

| File | Role |
|------|------|
| `layer1.py` | Layer 1: character/noise repair via Levenshtein search + LM score |
| `ard.py` | Layer 2: ARD loss, feature projector, KL + alignment |
| `attacks.py` | Adversarial example generation (TextFooler, DeepWordBug, Pruthi, …) |
| `data.py` | AdvGLUE loading, ARD pair dataset, IMDB mixing |
| `utils.py` | Seeding, device helpers, utilities |

---

## `scripts/data/`

| Script | Purpose | Output |
|--------|---------|--------|
| `augment_data.py` | TextAttack embedding augmentation on SST-2 train | `data/sst2_augmented.csv` |

---

## `scripts/training/`

| Script | Purpose | Checkpoints |
|--------|---------|-------------|
| `train_ard.py` | **Main paper path.** Teacher on clean text, student on adv text; ARD + optional Layer 1 | `models/dl_ag_student_distilbert/seed_{N}/` |
| `train_mixed_balanced.py` | **Legacy path.** Mixed AdvGLUE + IMDB knowledge distillation | `dl_ag_balanced_model/seed_{N}/` |

The paper and 5-seed suite use **`train_ard.py`**.

---

## `scripts/evaluation/`

| Script | Purpose |
|--------|---------|
| `evaluator.py` | Paper evaluator: attacks → Layer 1 → student; writes RA / ASR / latency JSON |
| `final_validation.py` | Legacy validation for the balanced model (AdvGLUE + IMDB) |

---

## `scripts/multiseed/`

| Script | Purpose |
|--------|---------|
| `run_5seed_full.py` | 5 seeds (42–46): `train_ard` → `evaluator`, mean ± std summary |
| `run_5seed.py` | Legacy: `train_mixed_balanced` → `final_validation` |
| `run_reviewer_suite.py` | Full pipeline: 5-seed DL-AG → baselines → PKD → Table 9 → t-tests |

**Reviewer suite order:**
1. DL-AG (`run_5seed_full`)
2. BERT-Base / DistilBERT baselines (`baselines_3seed`)
3. PKD ablation (`pkd_ablation_3seed`)
4. L1 / L2 / Hybrid ablation (`table9_ablation_3seed`)
5. Welch t-test + 95% CI (`stats_tests`)

On Colab with Google Drive: `--drive-root /content/drive/MyDrive/...`

---

## `scripts/ablations/`

| Script | Paper reference |
|--------|-----------------|
| `baselines_3seed.py` | Table 11 — BERT-Base & DistilBERT fine-tuning |
| `pkd_ablation_3seed.py` | Table 10 — Logit-Only vs PKD-Skip |
| `table9_ablation_3seed.py` | Table 9 — L1-only / L2-only / Hybrid |
| `stats_tests.py` | Reviewer statistics (Welch t-test, 95% CI) |

---

## `data/`

| File | Description |
|------|-------------|
| `dev.json` | AdvGLUE dev split (SST-2, etc.) — tracked in git |
| `sst2_augmented.csv` | Output of `augment_data.py` — **gitignored**, generated locally |

See also `data/README.md`.

---

## Gitignored artifacts

- `models/`, `dl_ag_*_model/` — model checkpoints
- `multiseed_results/`, `multiseed_results_full/` — experiment JSON outputs
- `data/*.csv` — large augmented dataset
- `*.pt`, `*.zip`, internal notes

---

## Upstream

AdvGLUE benchmark: https://github.com/AI-secure/adversarial-glue  
Git remote: `upstream` (reference only — do not push there)

---

## Notes

- Run scripts from the repo root; paths are resolved via `repo_paths.py`.
- The repo can be made public after paper acceptance (aligned with our data-availability statement).
- Collaborator access: GitHub → Settings → Collaborators.
