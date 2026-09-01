"""
Central path layout for the DL-AG repository.

All scripts should import from here so files can live in subfolders
without breaking defaults or subprocess calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

# --- Data & artifacts (gitignored where noted) ---
DATA_DIR = REPO_ROOT / "data"
AUGMENTED_CSV = DATA_DIR / "sst2_augmented.csv"
ADVGLUE_DEV_JSON = DATA_DIR / "dev.json"

MODELS_DIR = REPO_ROOT / "models"
MODELS_DLAG_STUDENT = MODELS_DIR / "dl_ag_student_distilbert"
MODELS_BALANCED = REPO_ROOT / "dl_ag_balanced_model"

RESULTS_LEGACY = REPO_ROOT / "multiseed_results"
RESULTS_FULL = REPO_ROOT / "multiseed_results_full"

# --- Script entrypoints ---
SCRIPTS = REPO_ROOT / "scripts"
SCRIPT_AUGMENT = SCRIPTS / "data" / "augment_data.py"
SCRIPT_TRAIN_ARD = SCRIPTS / "training" / "train_ard.py"
SCRIPT_TRAIN_BALANCED = SCRIPTS / "training" / "train_mixed_balanced.py"
SCRIPT_EVALUATOR = SCRIPTS / "evaluation" / "evaluator.py"
SCRIPT_FINAL_VALIDATION = SCRIPTS / "evaluation" / "final_validation.py"

SCRIPT_RUN_5SEED = SCRIPTS / "multiseed" / "run_5seed.py"
SCRIPT_RUN_5SEED_FULL = SCRIPTS / "multiseed" / "run_5seed_full.py"
SCRIPT_RUN_REVIEWER = SCRIPTS / "multiseed" / "run_reviewer_suite.py"

SCRIPT_BASELINES = SCRIPTS / "ablations" / "baselines_3seed.py"
SCRIPT_PKD = SCRIPTS / "ablations" / "pkd_ablation_3seed.py"
SCRIPT_TABLE9 = SCRIPTS / "ablations" / "table9_ablation_3seed.py"
SCRIPT_STATS = SCRIPTS / "ablations" / "stats_tests.py"

NOTEBOOK_5SEED = REPO_ROOT / "notebooks" / "DLAG_reviewer_5seed_colab.ipynb"


def ensure_import_path() -> Path:
    """Allow `import dlag` from any script under scripts/."""
    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return REPO_ROOT
