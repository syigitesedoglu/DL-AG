# Dual-Layer Adversarial Guard (DL-AG)

Open-source code for the paper **A Dual-Layer Adversarial Guard** (PeerJ Computer Science submission).

**Repository:** https://github.com/syigitesedoglu/DL-AG  
**License:** MIT  
**Corresponding author:** Selçuk Yiğit ESEDOĞLU (`s.esedoglu@ogr.iuc.edu.tr`)

DL-AG is a two-stage defense for text classifiers:
- **Layer 1** — character/noise repair (Levenshtein search + language-model score)
- **Layer 2** — robust distillation (RoBERTa-Large teacher → DistilBERT student) with ARD / Lipschitz-style regularization and feature alignment

---

## Data
- Code: this repository (MIT)
- Benchmarks: original AdvGLUE / GLUE sources
- Checkpoints: reproduce with the training scripts (large files are gitignored)

---

## Quick start

Run all commands from the **repository root** (GPU recommended).

```bash
pip install -r requirements_dlag.txt

# 1) Build augmented SST-2 pairs (once)
python scripts/data/augment_data.py

# 2) Paper protocol — 5-seed train + eval
python scripts/multiseed/run_5seed_full.py

# 3) Full reviewer pipeline (DL-AG + baselines + ablations + stats)
python scripts/multiseed/run_reviewer_suite.py
```

Colab notebook: `notebooks/DLAG_reviewer_5seed_colab.ipynb`

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
├── data/                          # Static data (dev.json; CSVs generated locally)
├── notebooks/                     # Colab notebooks
├── models/                        # Checkpoints (gitignored; created by training)
├── multiseed_results_full/        # JSON results (gitignored)
├── repo_paths.py                  # Shared path constants
├── requirements_dlag.txt
├── LICENSE
└── CITATION.cff
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

## Scripts

### `scripts/data/`

| Script | Purpose | Output |
|--------|---------|--------|
| `augment_data.py` | TextAttack embedding augmentation on SST-2 train | `data/sst2_augmented.csv` |

### `scripts/training/`

| Script | Purpose | Checkpoints |
|--------|---------|-------------|
| `train_ard.py` | **Main paper path.** Teacher on clean text, student on adv text; ARD + optional Layer 1 | `models/dl_ag_student_distilbert/seed_{N}/` |
| `train_mixed_balanced.py` | Legacy mixed AdvGLUE + IMDB distillation | `dl_ag_balanced_model/seed_{N}/` |

The paper and 5-seed suite use **`train_ard.py`**.

### `scripts/evaluation/`

| Script | Purpose |
|--------|---------|
| `evaluator.py` | Paper evaluator: attacks → Layer 1 → student; RA / ASR / latency JSON |
| `final_validation.py` | Legacy validation for the balanced model |

### `scripts/multiseed/`

| Script | Purpose |
|--------|---------|
| `run_5seed_full.py` | 5 seeds (42–46): `train_ard` → `evaluator`, mean ± std |
| `run_5seed.py` | Legacy balanced-model path |
| `run_reviewer_suite.py` | Full pipeline: DL-AG → baselines → PKD → Table 9 → t-tests |

**Reviewer suite order:**
1. DL-AG (`run_5seed_full`)
2. BERT-Base / DistilBERT baselines (`baselines_3seed`)
3. PKD ablation (`pkd_ablation_3seed`)
4. L1 / L2 / Hybrid ablation (`table9_ablation_3seed`)
5. Welch t-test + 95% CI (`stats_tests`)

### `scripts/ablations/`

| Script | Role |
|--------|------|
| `baselines_3seed.py` | BERT-Base & DistilBERT fine-tuning |
| `pkd_ablation_3seed.py` | Logit-Only vs PKD-Skip |
| `table9_ablation_3seed.py` | L1-only / L2-only / Hybrid |
| `stats_tests.py` | Welch t-test, 95% CI |

---

## `data/`

| File | Description |
|------|-------------|
| `dev.json` | AdvGLUE-related dev split used in experiments — tracked in git |
| `sst2_augmented.csv` | Output of `augment_data.py` — **gitignored**, generate locally |

See also `data/README.md`.

---

## Gitignored artifacts

- `models/`, `dl_ag_*_model/` — checkpoints (regenerate with training scripts)
- `multiseed_results/`, `multiseed_results_full/` — experiment JSON
- `data/*.csv` — large augmented datasets
- `*.pt`, `*.bin`, `*.safetensors`

---

## Upstream

- AdvGLUE: https://github.com/AI-secure/adversarial-glue  
- Hugging Face model hubs used by `transformers` (DistilBERT, RoBERTa-Large, BERT-Base)

---

## Citation

If you use this code, please cite the paper (citation details will be updated upon publication). Soft citation metadata is in `CITATION.cff`.

---

## Notes

- Run scripts from the repo root; paths resolve via `repo_paths.py`.
- Questions: corresponding author email above.
