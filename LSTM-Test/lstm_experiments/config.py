"""Match the archived CNN protocol except for the sequence encoder."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CNN_ROOT = PROJECT_ROOT.parent / "CNN-Test"
RESULTS_DIR = PROJECT_ROOT / "results"
PREPARED_DIR = RESULTS_DIR / "prepared"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
PLOTS_DIR = PROJECT_ROOT / "plots"
EMBEDDING_DIR = PROJECT_ROOT / "data" / "embeddings"
SEED = 42
GLOVE_DIM = 300
MAX_SEQUENCE_LENGTH = 50
LSTM_HIDDEN_SIZES = (64, 32)  # Units per direction, matching the EDA reference.
LSTM_DROPOUT = 0.5
HIDDEN_UNITS = 20
NUM_CLASSES = 2
LEARNING_RATE = 0.001
BATCH_SIZE = 64
MAX_EPOCHS = 30
PATIENCE = 3
PUNCTUATION = (".", ",", "!", "?", ";", ":")
CONDITIONS = ("baseline", "eda", "aeda", "eda_aeda")
DISPLAY_NAMES = {"baseline": "Baseline", "eda": "EDA", "aeda": "AEDA", "eda_aeda": "EDA → AEDA"}
PREPARED_FILENAMES = {name: f"{name}_train.jsonl" for name in CONDITIONS}
PREPARED_FILENAMES.update(validation="validation.jsonl", test="test.jsonl")


def default_glove_path() -> Path:
    local = EMBEDDING_DIR / "glove.6B.300d.txt"
    shared = CNN_ROOT / "data" / "embeddings" / local.name
    return shared if not local.is_file() and shared.is_file() else local
