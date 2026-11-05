"""Vocabulary construction and memory-efficient loading of GloVe 300d vectors."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

try:
    from .config import GLOVE_DIM
    from .dataset import PAD_TOKEN, UNK_TOKEN
    from .text_utils import tokenize
except ImportError:
    from config import GLOVE_DIM
    from dataset import PAD_TOKEN, UNK_TOKEN
    from text_utils import tokenize


@dataclass
class EmbeddingResult:
    matrix: torch.Tensor
    covered_tokens: set[str]
    vocabulary_size: int


def build_vocabulary(records: Iterable[Mapping[str, object]]) -> dict[str, int]:
    """Build one deterministic vocabulary from the union of fitting text only."""
    counter: Counter[str] = Counter()
    for record in records:
        counter.update(tokenize(str(record["text"])))
    vocabulary = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for token in sorted(counter):
        vocabulary[token] = len(vocabulary)
    return vocabulary


def load_glove_embeddings(glove_path: Path, vocabulary: Mapping[str, int]) -> EmbeddingResult:
    """Read only vectors whose token occurs in the shared fitting vocabulary.

    Unknown words and padding use all-zero vectors; random substitute embeddings
    are intentionally never used by the real experiments.
    """
    if not glove_path.is_file():
        raise FileNotFoundError(
            f"Missing GloVe 300d file: {glove_path}. Run download_glove.py or pass --glove-path."
        )

    matrix = np.zeros((len(vocabulary), GLOVE_DIM), dtype=np.float32)
    covered_tokens: set[str] = set()
    with glove_path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            token, separator, values = line.rstrip("\n").partition(" ")
            if not separator or token not in vocabulary:
                continue
            vector = np.fromstring(values, sep=" ", dtype=np.float32)
            if vector.size != GLOVE_DIM:
                continue
            matrix[vocabulary[token]] = vector
            covered_tokens.add(token)

    return EmbeddingResult(
        matrix=torch.from_numpy(matrix),
        covered_tokens=covered_tokens,
        vocabulary_size=len(vocabulary),
    )


def save_vocabulary(vocabulary: Mapping[str, int], destination: Path) -> None:
    destination.write_text(json.dumps(vocabulary, ensure_ascii=False, indent=2), encoding="utf-8")

