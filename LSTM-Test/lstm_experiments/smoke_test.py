"""Offline model checks using synthetic vectors; never writes experiment scores."""

import copy
from pathlib import Path
import tempfile

import torch
from torch import nn

from .config import GLOVE_DIM, MAX_SEQUENCE_LENGTH, SEED
from .dataset import TextClassificationDataset
from .model import TextLSTM
from .text_utils import contains_expected_punctuation, tokenize
from .train import set_global_seed, train_one_condition


def main():
    set_global_seed(SEED)
    assert contains_expected_punctuation(tokenize("this movie ! is really , good"))
    vocabulary = {"<pad>": 0, "<unk>": 1, "good": 2, "bad": 3, "movie": 4, "!": 5}
    matrix = torch.randn(len(vocabulary), GLOVE_DIM) * 0.1
    matrix[:2].zero_()
    model = TextLSTM(matrix)
    model.eval()
    # Mixed lengths, unknown tokens, and empty input all use the same batch.
    inputs = torch.tensor([[2, 4, 5, 0], [3, 0, 0, 0], [0, 0, 0, 0], [1, 4, 0, 0]])
    logits = model(inputs)
    assert logits.shape == (4, 2) and torch.isfinite(logits).all()
    padded = torch.nn.functional.pad(inputs, (0, 7))
    torch.testing.assert_close(logits, model(padded))
    for index in range(len(inputs)):
        torch.testing.assert_close(logits[index], model(inputs[index:index + 1])[0])
    loss = nn.CrossEntropyLoss()(logits, torch.tensor([1, 0, 0, 1]))
    loss.backward()
    assert model.embedding.weight.grad is None
    for layer in (model.lstm1, model.lstm2):
        for parameter in layer.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert layer.weight_ih_l0.grad.abs().sum() > 0
        assert layer.weight_ih_l0_reverse.grad.abs().sum() > 0
    # Check the two final directional states against separate unpadded examples.
    for index, row in enumerate(inputs):
        length = int(row.ne(0).sum())
        if length:
            first, _ = model.lstm1(model.embedding(row[:length].unsqueeze(0)))
            _, (states, _) = model.lstm2(first)
            sentence = torch.cat((states[0], states[1]), dim=1)
            expected = model.output(model.relu(model.hidden(sentence)))
            torch.testing.assert_close(logits[index], expected[0], atol=1e-6, rtol=1e-5)
    # Dropout acts in training and is disabled during evaluation.
    for dropout in (model.dropout1, model.dropout2):
        dropout.train()
        dropped = dropout(torch.ones(4096))
        assert (dropped == 0).any() and (dropped == 2).any()
        dropout.eval()
        torch.testing.assert_close(dropout(torch.ones(16)), torch.ones(16))

    # Exercise actual optimizer, best-checkpoint save/restore, and evaluation.
    records = [{"text": "good movie !", "label": 1}, {"text": "bad movie", "label": 0}] * 2
    dataset = TextClassificationDataset(records, vocabulary, MAX_SEQUENCE_LENGTH)
    with tempfile.TemporaryDirectory() as temporary:
        checkpoint_path = Path(temporary) / "smoke.pt"
        result = train_one_condition(
            model, dataset, dataset, dataset, checkpoint_path,
            learning_rate=0.001, batch_size=4, max_epochs=2, patience=1,
            seed=SEED, device=torch.device("cpu"), checkpoint_metadata={"synthetic_test": True},
        )
        assert 0 <= result["test_accuracy"] <= 1
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        restored = TextLSTM(copy.deepcopy(matrix))
        restored.load_state_dict(checkpoint["model_state"])
        restored.eval()
        model.eval()
        torch.testing.assert_close(model(inputs), restored(inputs))
    print("Smoke test passed: stacked bidirectional states, dropout, padding, batch ordering, empty/unknown inputs, frozen embeddings, gradients, and checkpoint round trip.")


if __name__ == "__main__":
    main()
