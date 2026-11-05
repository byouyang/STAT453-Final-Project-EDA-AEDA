"""The common CNN architecture used for all four Experiment A conditions."""

from __future__ import annotations

import torch
from torch import nn

try:
    from .config import FILTERS, GLOVE_DIM, HIDDEN_UNITS, KERNEL_SIZE, NUM_CLASSES
except ImportError:
    from config import FILTERS, GLOVE_DIM, HIDDEN_UNITS, KERNEL_SIZE, NUM_CLASSES


class TextCNN(nn.Module):
    """GloVe -> Conv1d(128, k=5) -> global max -> Dense(20) -> 2 logits.

    The GloVe table is frozen to match the original EDA implementation, which
    supplied static pre-trained vectors directly to the convolutional layer.
    """

    def __init__(self, embedding_matrix: torch.Tensor) -> None:
        super().__init__()
        if embedding_matrix.ndim != 2 or embedding_matrix.shape[1] != GLOVE_DIM:
            raise ValueError(f"Expected a [vocab, {GLOVE_DIM}] embedding matrix.")
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.float(), freeze=True, padding_idx=0
        )
        self.conv = nn.Conv1d(GLOVE_DIM, FILTERS, KERNEL_SIZE)
        self.relu = nn.ReLU()
        self.hidden = nn.Linear(FILTERS, HIDDEN_UNITS)
        self.output = nn.Linear(HIDDEN_UNITS, NUM_CLASSES)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(token_ids)  # [batch, sequence, 300]
        convolved = self.relu(self.conv(embedded.transpose(1, 2)))
        pooled = torch.amax(convolved, dim=2)
        hidden = self.relu(self.hidden(pooled))
        return self.output(hidden)  # Raw logits: no Softmax before CrossEntropyLoss.

