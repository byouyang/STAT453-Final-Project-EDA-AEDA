"""PyTorch dataset and fixed-length encoding for the Experiment A LSTM."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch.utils.data import Dataset

try:
    from .text_utils import tokenize
except ImportError:  # Allows direct execution during local debugging.
    from text_utils import tokenize


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"


def encode_text(text: str, vocabulary: Mapping[str, int], max_length: int) -> list[int]:
    tokens = tokenize(text)[:max_length]
    encoded = [vocabulary.get(token, vocabulary[UNK_TOKEN]) for token in tokens]
    return encoded + [vocabulary[PAD_TOKEN]] * (max_length - len(encoded))


class TextClassificationDataset(Dataset):
    """Pre-encodes a prepared split so all conditions share encoding behavior."""

    def __init__(
        self,
        records: Sequence[Mapping[str, object]],
        vocabulary: Mapping[str, int],
        max_length: int,
    ) -> None:
        self.inputs = torch.tensor(
            [encode_text(str(record["text"]), vocabulary, max_length) for record in records],
            dtype=torch.long,
        )
        self.labels = torch.tensor([int(record["label"]) for record in records], dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.inputs[index], self.labels[index]

