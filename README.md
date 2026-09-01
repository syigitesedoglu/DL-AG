# Dual-Layer Adversarial Guard (DL-AG)

Private repo — makale deneyleri, 5-seed protokolü ve reviewer suite.

**GitHub:** https://github.com/syigitesedoglu/DL-AG (private)

---

## Hızlı başlangıç

```bash
pip install -r requirements_dlag.txt

# 1) Augment edilmiş SST-2 çiftlerini üret (bir kez)
python scripts/data/augment_data.py

# 2) Paper protokolü — 5 seed train + eval
python scripts/multiseed/run_5seed_full.py

# 3) Tüm reviewer paketi (DL-AG + baseline + ablation + istatistik)
python scripts/multiseed/run_reviewer_suite.py
```

Colab: `notebooks/DLAG_reviewer_5seed_colab.ipynb`

---

## Klasör yapısı

```
DL-AG/
├── dlag/                          # Çekirdek Python paketi
├── scripts/
│   ├── data/                      # Veri hazırlama
│   ├── training/                  # Model eğitimi
│   ├── evaluation/                # Test & metrik
│   ├── multiseed/                 # Orkestrasyon (5 seed)
│   └── ablations/                 # Tablo/ablation deneyleri
├── data/                          # Statik veri (dev.json, üretilen CSV)
├── notebooks/                     # Colab notebookları
├── models/                        # Checkpoint (gitignore)
├── multiseed_results_full/        # JSON sonuçlar (gitignore)
├── repo_paths.py                  # Tüm yollar burada
└── requirements_dlag.txt
```

---

## `dlag/` — çekirdek paket

| Dosya | Ne yapar |
|--------|-----------|
| `layer1.py` | Katman 1: Levenshtein + LM skoru ile karakter/gürültü onarımı |
| `ard.py` | Katman 2: ARD kaybı, feature projector, KL + hizalama |
| `attacks.py` | TextFooler, DeepWordBug, Pruthi vb. saldırı üretimi |
| `data.py` | AdvGLUE yükleme, ARD pair dataset, IMDB karışımı |
| `utils.py` | Seed, device, yardımcılar |

---

## `scripts/data/`

| Script | Ne yapar | Çıktı |
|--------|-----------|--------|
| `augment_data.py` | SST-2 train üzerinde TextAttack embedding augment | `data/sst2_augmented.csv` |

---

## `scripts/training/`

| Script | Ne yapar | Checkpoint |
|--------|-----------|------------|
| `train_ard.py` | **Ana paper yolu.** Teacher=clean, student=adv, ARD + Layer1 opsiyonel | `models/dl_ag_student_distilbert/seed_{N}/` |
| `train_mixed_balanced.py` | **Eski/legacy yol.** AdvGLUE + IMDB karışık KD | `dl_ag_balanced_model/seed_{N}/` |

Paper ve 5-seed suite **`train_ard.py`** kullanır.

---

## `scripts/evaluation/`

| Script | Ne yapar |
|--------|-----------|
| `evaluator.py` | Paper evaluator: saldırılar + Layer1 + student, RA/ASR/latency JSON |
| `final_validation.py` | Legacy balanced model için AdvGLUE + IMDB doğrulama |

---

## `scripts/multiseed/`

| Script | Ne yapar |
|--------|-----------|
| `run_5seed_full.py` | 5 seed (42–46): `train_ard` → `evaluator`, mean±std özeti |
| `run_5seed.py` | Legacy: `train_mixed_balanced` → `final_validation` |
| `run_reviewer_suite.py` | Sırayla: 5-seed DL-AG → baselines → PKD → Tablo9 → t-test |

**Reviewer suite sırası:**
1. DL-AG (run_5seed_full)
2. BERT / DistilBERT baseline (baselines_3seed)
3. PKD ablation (pkd_ablation_3seed)
4. L1 / L2 / Hybrid ablation (table9_ablation_3seed)
5. Welch t-test + CI (stats_tests)

Colab'da Drive için: `--drive-root /content/drive/MyDrive/...`

---

## `scripts/ablations/`

| Script | Makaledeki karşılık |
|--------|---------------------|
| `baselines_3seed.py` | Tablo 11 — BERT-Base & DistilBERT fine-tune |
| `pkd_ablation_3seed.py` | Tablo 10 — Logit-Only vs PKD-Skip |
| `table9_ablation_3seed.py` | Tablo 9 — L1-only / L2-only / Hybrid |
| `stats_tests.py` | Reviewer istatistikleri (Welch, 95% CI) |

---

## `data/`

| Dosya | Açıklama |
|--------|-----------|
| `dev.json` | AdvGLUE dev split (SST-2 vb.) — repo içinde |
| `sst2_augmented.csv` | `augment_data.py` çıktısı — **gitignore**, yerelde üretilir |

---

## Gitignore (repo dışı kalır)

- `models/`, `dl_ag_*_model/` — checkpoint
- `multiseed_results/`, `multiseed_results_full/` — JSON sonuçlar
- `data/*.csv` — büyük augment dosyası
- `*.pt`, `*.zip`, internal notlar

---

## Upstream

AdvGLUE benchmark: https://github.com/AI-secure/adversarial-glue  
Remote adı: `upstream` (push etme, sadece referans)

---

## Notlar

- Tüm scriptler repo kökünden çalıştırılmalı (`cwd` = repo root); yollar `repo_paths.py` üzerinden.
- Makale kabulünden sonra repo public yapılabilir (data availability ile uyumlu).
- Hoca erişimi: GitHub → Settings → Collaborators.
