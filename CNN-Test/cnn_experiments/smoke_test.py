"""Fast forward/backward and punctuation-preservation test; no experiment result is produced."""

from __future__ import annotations

import copy

import torch
from torch import nn

try:
    from .config import GLOVE_DIM, MAX_SEQUENCE_LENGTH, SEED
    from .dataset import TextClassificationDataset
    from .model import TextCNN
    from .text_utils import contains_expected_punctuation, tokenize
    from .train import set_global_seed
except ImportError:
    from config import GLOVE_DIM, MAX_SEQUENCE_LENGTH, SEED
    from dataset import TextClassificationDataset
    from model import TextCNN
    from text_utils import contains_expected_punctuation, tokenize
    from train import set_global_seed


def main() -> None:
    example = "this movie ! is really , good"
    tokens = tokenize(example)
    assert contains_expected_punctuation(tokens), f"Punctuation disappeared: {tokens}"

    # A tiny fixed matrix is only for mechanical smoke testing. Real experiments
    # refuse to run without pre-trained GloVe vectors.
    vocabulary = {"<pad>": 0, "<unk>": 1, "this": 2, "movie": 3, "!": 4, "is": 5, "really": 6, ",": 7, "good": 8, "bad": 9}
    records = [
        {"text": example, "label": 1},
        {"text": "this movie is bad", "label": 0},
    ]
    dataset = TextClassificationDataset(records, vocabulary, MAX_SEQUENCE_LENGTH)
    fixed_matrix = torch.arange(len(vocabulary) * GLOVE_DIM, dtype=torch.float32).reshape(
        len(vocabulary), GLOVE_DIM
    ) / 10_000.0
    fixed_matrix[0].zero_()
    set_global_seed(SEED)
    model = TextCNN(copy.deepcopy(fixed_matrix))
    inputs, labels = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    logits = model(inputs)
    assert logits.shape == (2, 2), logits.shape
    loss = nn.CrossEntropyLoss()(logits, labels)
    loss.backward()
    assert model.conv.weight.grad is not None and torch.isfinite(model.conv.weight.grad).all()
    print("Smoke test passed: logits, backward pass, and !/, token preservation verified.")


if __name__ == "__main__":
    main()

