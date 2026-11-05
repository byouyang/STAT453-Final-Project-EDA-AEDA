"""Evaluation helpers that operate on raw LSTM logits."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    correct = 0
    for inputs, labels in loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        logits = model(inputs)
        loss = criterion(logits, labels)
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_examples += batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
    if total_examples == 0:
        raise ValueError("Cannot evaluate an empty split.")
    return {
        "loss": total_loss / total_examples,
        "accuracy": correct / total_examples,
        "examples": float(total_examples),
    }

