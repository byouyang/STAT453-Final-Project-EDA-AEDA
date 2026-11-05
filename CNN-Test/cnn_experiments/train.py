"""Deterministic training with validation-loss early stopping and checkpoints."""

from __future__ import annotations

import random
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

try:
    from .evaluate import evaluate
except ImportError:
    from evaluate import evaluate


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def train_one_condition(
    model: nn.Module,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    test_dataset: Dataset,
    checkpoint_path: Path,
    *,
    learning_rate: float,
    batch_size: int,
    max_epochs: int,
    patience: int,
    seed: int,
    device: torch.device,
    checkpoint_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Fit a fresh model and restore its minimum-validation-loss checkpoint."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=learning_rate)
    train_loader = make_loader(train_dataset, batch_size, shuffle=True, seed=seed)
    validation_loader = make_loader(validation_dataset, batch_size, shuffle=False, seed=seed)
    test_loader = make_loader(test_dataset, batch_size, shuffle=False, seed=seed)

    history: list[dict[str, float]] = []
    best_validation_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_examples = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item() * labels.size(0)
            train_examples += labels.size(0)

        validation = evaluate(model, validation_loader, criterion, device)
        train_loss = train_loss_sum / train_examples
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "validation_loss": validation["loss"],
            }
        )

        if validation["loss"] < best_validation_loss - 1e-12:
            best_validation_loss = validation["loss"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_validation_loss": best_validation_loss,
                    "metadata": dict(checkpoint_metadata),
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    elapsed_seconds = time.perf_counter() - started
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    test = evaluate(model, test_loader, criterion, device)
    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_validation_loss": best_validation_loss,
        "epochs_completed": len(history),
        "training_time_seconds": elapsed_seconds,
        "test_loss": test["loss"],
        "test_accuracy": test["accuracy"],
    }

