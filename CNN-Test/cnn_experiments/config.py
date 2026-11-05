"""Experiment-wide constants and repository-relative paths."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]  # CNN-Test/
REPOSITORY_ROOT = PROJECT_ROOT.parent  # Existing five source files live here.
RESULTS_DIR = PROJECT_ROOT / "results"
PREPARED_DIR = RESULTS_DIR / "prepared"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
PLOTS_DIR = PROJECT_ROOT / "plots"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings"

BASELINE_FILE = REPOSITORY_ROOT / "sst2_train_500.txt"
EDA_FILE = REPOSITORY_ROOT / "lol.txt"
SEQUENTIAL_FILE = REPOSITORY_ROOT / "aeda_lol.txt"
GROUP_AEDA_TRAIN_FILE = REPOSITORY_ROOT / "aeda_sst2_train.txt"

SEED = 42
VALIDATION_FOLDS = 10
GLOVE_DIM = 300
MAX_SEQUENCE_LENGTH = 50
FILTERS = 128
KERNEL_SIZE = 5
HIDDEN_UNITS = 20
NUM_CLASSES = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 3
PUNCTUATION = (".", ",", "!", "?", ";", ":")

CONDITIONS = ("baseline", "eda", "aeda", "eda_aeda")
DISPLAY_NAMES = {
    "baseline": "Baseline",
    "eda": "EDA",
    "aeda": "AEDA",
    "eda_aeda": "EDA → AEDA",
}
PREPARED_FILENAMES = {
    "baseline": "baseline_train.jsonl",
    "eda": "eda_train.jsonl",
    "aeda": "aeda_train.jsonl",
    "eda_aeda": "eda_aeda_train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
}


def default_aeda_repo() -> Path:
    """Locate the separately cloned official AEDA repository in this layout."""
    return REPOSITORY_ROOT.parent / "aeda_nlp_reference"


def default_glove_path() -> Path:
    return EMBEDDING_DIR / "glove.6B.300d.txt"

