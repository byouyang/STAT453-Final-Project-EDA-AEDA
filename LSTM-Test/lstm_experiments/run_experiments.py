"""Run the four fixed-config, leakage-checked LSTM Experiment A conditions."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

try:
    from .config import (
        BATCH_SIZE,
        CHECKPOINT_DIR,
        CONDITIONS,
        DISPLAY_NAMES,
        GLOVE_DIM,
        LEARNING_RATE,
        MAX_EPOCHS,
        MAX_SEQUENCE_LENGTH,
        PATIENCE,
        PLOTS_DIR,
        PROJECT_ROOT,
        PUNCTUATION,
        RESULTS_DIR,
        SEED,
        CNN_ROOT,
        default_glove_path,
    )
    from .dataset import TextClassificationDataset
    from .embeddings import build_vocabulary, load_glove_embeddings, save_vocabulary
    from .model import TextLSTM
    from .prepare_data import prepare_all, read_prepared_records, write_csv
    from .text_utils import contains_expected_punctuation, tokenize
    from .train import set_global_seed, train_one_condition
except ImportError:
    from config import (
        BATCH_SIZE,
        CHECKPOINT_DIR,
        CONDITIONS,
        DISPLAY_NAMES,
        GLOVE_DIM,
        LEARNING_RATE,
        MAX_EPOCHS,
        MAX_SEQUENCE_LENGTH,
        PATIENCE,
        PLOTS_DIR,
        PROJECT_ROOT,
        PUNCTUATION,
        RESULTS_DIR,
        SEED,
        CNN_ROOT,
        default_glove_path,
    )
    from dataset import TextClassificationDataset
    from embeddings import build_vocabulary, load_glove_embeddings, save_vocabulary
    from model import TextLSTM
    from prepare_data import prepare_all, read_prepared_records, write_csv
    from text_utils import contains_expected_punctuation, tokenize
    from train import set_global_seed, train_one_condition


def choose_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is not available.")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def text_quality_row(
    dataset: str,
    split: str,
    records: Sequence[Mapping[str, object]],
    vocabulary: Mapping[str, int],
    covered_tokens: set[str],
) -> dict[str, Any]:
    token_count = 0
    unembedded_count = 0
    unseen_vocabulary_count = 0
    truncated_samples = 0
    truncated_tokens = 0
    maximum_length = 0
    for record in records:
        tokens = tokenize(str(record["text"]))
        token_count += len(tokens)
        maximum_length = max(maximum_length, len(tokens))
        unseen_vocabulary_count += sum(token not in vocabulary for token in tokens)
        unembedded_count += sum(
            token not in vocabulary or token not in covered_tokens for token in tokens
        )
        if len(tokens) > MAX_SEQUENCE_LENGTH:
            truncated_samples += 1
            truncated_tokens += len(tokens) - MAX_SEQUENCE_LENGTH
    return {
        "dataset": dataset,
        "split": split,
        "samples": len(records),
        "tokens_before_padding": token_count,
        "unseen_vocabulary_tokens": unseen_vocabulary_count,
        "unembedded_tokens": unembedded_count,
        "unembedded_token_rate": unembedded_count / token_count if token_count else 0.0,
        "truncated_samples": truncated_samples,
        "truncated_tokens": truncated_tokens,
        "maximum_tokens_before_truncation": maximum_length,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
    }


def write_text_quality_artifacts(
    train_records: Mapping[str, Sequence[Mapping[str, object]]],
    validation_records: Sequence[Mapping[str, object]],
    test_records: Sequence[Mapping[str, object]],
    vocabulary: Mapping[str, int],
    covered_tokens: set[str],
) -> None:
    rows = [
        text_quality_row(DISPLAY_NAMES[condition], "train", records, vocabulary, covered_tokens)
        for condition, records in train_records.items()
    ]
    rows.extend(
        [
            text_quality_row("Common", "validation", validation_records, vocabulary, covered_tokens),
            text_quality_row("Common", "test", test_records, vocabulary, covered_tokens),
        ]
    )
    write_csv(
        RESULTS_DIR / "text_quality.csv",
        [
            "dataset",
            "split",
            "samples",
            "tokens_before_padding",
            "unseen_vocabulary_tokens",
            "unembedded_tokens",
            "unembedded_token_rate",
            "truncated_samples",
            "truncated_tokens",
            "maximum_tokens_before_truncation",
            "max_sequence_length",
        ],
        rows,
    )

    punctuation_rows: list[dict[str, Any]] = []
    for punctuation in PUNCTUATION:
        punctuation_rows.append(
            {
                "punctuation_token": punctuation,
                "count_in_aeda_train": sum(
                    tokenize(str(record["text"])).count(punctuation)
                    for record in train_records["aeda"]
                ),
                "count_in_sequential_train": sum(
                    tokenize(str(record["text"])).count(punctuation)
                    for record in train_records["eda_aeda"]
                ),
                "in_shared_vocabulary": punctuation in vocabulary,
                "has_glove_vector": punctuation in covered_tokens,
            }
        )
    write_csv(
        RESULTS_DIR / "punctuation_check.csv",
        [
            "punctuation_token",
            "count_in_aeda_train",
            "count_in_sequential_train",
            "in_shared_vocabulary",
            "has_glove_vector",
        ],
        punctuation_rows,
    )
    probe = tokenize("this movie ! is really , good")
    punctuation_status = {
        "tokenizer_probe": "this movie ! is really , good",
        "tokens": probe,
        "both_exclamation_and_comma_preserved": contains_expected_punctuation(probe),
        "punctuation_vector_rows": punctuation_rows,
    }
    (RESULTS_DIR / "punctuation_check.json").write_text(
        json.dumps(punctuation_status, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def plot_results(metrics: Sequence[Mapping[str, Any]], histories: Mapping[str, Sequence[Mapping[str, float]]]) -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    labels = [metric["method"] for metric in metrics]
    accuracies = [float(metric["test_accuracy_pct"]) for metric in metrics]
    colors = ["#3569a8", "#c99a22", "#c76b2d", "#6b8451"]

    fig, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(labels, accuracies, color=colors, edgecolor="#243248", linewidth=0.8)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Test Accuracy (%)")
    axis.set_title("LSTM Test Accuracy Comparison")
    axis.grid(axis="y", color="#d8dde5", linewidth=0.8)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, accuracies):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 1.2, f"{value:.2f}%", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "test_accuracy_comparison.png", dpi=180)
    plt.close(fig)

    improvement_metrics = metrics[1:]
    improvement_labels = [metric["method"] for metric in improvement_metrics]
    improvements = [float(metric["improvement_vs_baseline_pp"]) for metric in improvement_metrics]
    fig, axis = plt.subplots(figsize=(8, 5))
    bars = axis.bar(improvement_labels, improvements, color=colors[1:], edgecolor="#243248", linewidth=0.8)
    axis.axhline(0, color="#243248", linewidth=1.0)
    extent = max(1.0, max(abs(value) for value in improvements) + 0.75)
    axis.set_ylim(-extent, extent)
    axis.set_ylabel("Improvement over Baseline (percentage points)")
    axis.set_title("LSTM Accuracy Improvement over Baseline")
    axis.grid(axis="y", color="#d8dde5", linewidth=0.8)
    axis.set_axisbelow(True)
    for bar, value in zip(bars, improvements):
        offset = 0.08 * extent if value >= 0 else -0.08 * extent
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.2f} pp",
            ha="center",
            va="bottom" if value >= 0 else "top",
        )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "improvement_over_baseline.png", dpi=180)
    plt.close(fig)

    # A shared y-scale makes validation-loss magnitude comparable across methods.
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharey=True)
    for axis, condition in zip(axes.flat, CONDITIONS):
        history = histories[condition]
        epochs = [row["epoch"] for row in history]
        axis.plot(epochs, [row["train_loss"] for row in history], label="Training", color="#3569a8", marker="o", markersize=3)
        axis.plot(epochs, [row["validation_loss"] for row in history], label="Validation", color="#c76b2d", marker="o", markersize=3)
        axis.set_title(DISPLAY_NAMES[condition])
        axis.set_xlabel("Epoch")
        axis.set_ylabel("Cross-entropy loss")
        axis.grid(color="#d8dde5", linewidth=0.8)
        axis.legend(frameon=False)
    fig.suptitle("Training and Validation Loss by Condition", y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(PLOTS_DIR / "training_validation_loss.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_experiment_report(
    metrics: Sequence[Mapping[str, Any]],
    pairwise: Sequence[Mapping[str, Any]],
    device: torch.device,
) -> None:
    lines = [
        "# STAT 453 Experiment A — LSTM Results",
        "",
        "This report was generated from the actual fixed-seed run. All accuracy deltas below are percentage points (pp).",
        "",
        "## LSTM Test Accuracy",
        "",
        "| Method | Train samples | Best epoch | Best validation loss | Test accuracy | Improvement vs. Baseline |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for metric in metrics:
        improvement = "—" if metric["method"] == "Baseline" else f"{metric['improvement_vs_baseline_pp']:+.2f} pp"
        lines.append(
            f"| {metric['method']} | {metric['train_samples']} | {metric['best_epoch']} | "
            f"{metric['best_validation_loss']:.4f} | {metric['test_accuracy_pct']:.2f}% | {improvement} |"
        )
    lines.extend(
        [
            "",
            "## Pairwise Augmentation Differences",
            "",
            "| Comparison | Difference |",
            "|---|---:|",
        ]
    )
    lines.extend(f"| {row['comparison']} | {row['difference_pp']:+.2f} pp |" for row in pairwise)
    lines.extend(
        [
            "",
            "## Dataset and Leakage Verification",
            "",
            "- Common validation data were drawn once from the 500 original sources with stratified group splitting (seed 42).",
            "- The EDA and EDA → AEDA candidate files each had 3,000 rows; source-linked validation siblings were excluded before fitting.",
            "- AEDA used 3,000 sampled, official-method punctuation augmentations and was subset to the same post-filter class totals as EDA and EDA → AEDA.",
            "- `overlap_report.csv` reports both canonical and punctuation-insensitive checks; all final training-versus-validation/test cells must be zero.",
            "- `punctuation_check.csv` and `text_quality.csv` document token preservation, GloVe coverage, and truncation at 50 tokens.",
            "",
            "## Interpretation",
            "",
            "This single-seed run has no error bars or significance tests. AEDA uses a larger original source pool than the other conditions, so its difference cannot be attributed solely to punctuation augmentation.",
            "",
            "The training, validation, and test rows are copied byte-for-byte from the archived CNN experiment; see data_preparation_manifest.json for hashes. The CNN comparison is descriptive, not a claim of statistical significance.",
            "",
            "## Reproducibility",
            "",
            f"- Device: `{device}`",
            f"- Seed: `{SEED}`; Adam learning rate `{LEARNING_RATE}`; batch size `{BATCH_SIZE}`; up to `{MAX_EPOCHS}` epochs; patience `{PATIENCE}`.",
            f"- Model: frozen GloVe {GLOVE_DIM}d → BiLSTM(64 per direction, full sequence) → Dropout(0.5) → BiLSTM(32 per direction, concatenated final states) → Dropout(0.5) → Dense(20, ReLU) → 2 raw logits.",
            "- Layer architecture matches the EDA reference; packed padding handling and the archived CNN data/training protocol are retained.",
            "- See `metrics.csv`, `comparison.csv`, `pairwise_differences.csv`, `../checkpoints/`, and `../plots/` for the underlying outputs.",
        ]
    )
    (RESULTS_DIR / "EXPERIMENT_A_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_cnn_comparison(metrics: Sequence[Mapping[str, Any]]) -> None:
    """Compare actual LSTM scores with the archived CNN scores by condition."""
    reference = CNN_ROOT / "results" / "metrics.csv"
    if not reference.is_file():
        print("CNN metrics unavailable; skipping architecture comparison.")
        return
    with reference.open(encoding="utf-8", newline="") as handle:
        cnn = {row["condition"]: row for row in csv.DictReader(handle)}
    rows = []
    for metric in metrics:
        cnn_accuracy = float(cnn[metric["condition"]]["test_accuracy_pct"])
        rows.append({
            "method": metric["method"],
            "cnn_test_accuracy_pct": cnn_accuracy,
            "lstm_test_accuracy_pct": metric["test_accuracy_pct"],
            "lstm_minus_cnn_pp": metric["test_accuracy_pct"] - cnn_accuracy,
        })
    write_csv(RESULTS_DIR / "cnn_lstm_comparison.csv", list(rows[0]), rows)


def run(args: argparse.Namespace) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    prepare_all()

    train_records = {condition: read_prepared_records(condition) for condition in CONDITIONS}
    validation_records = read_prepared_records("validation")
    test_records = read_prepared_records("test")
    vocabulary = build_vocabulary(
        record for condition in CONDITIONS for record in train_records[condition]
    )
    embedding_result = load_glove_embeddings(args.glove_path, vocabulary)
    save_vocabulary(vocabulary, RESULTS_DIR / "vocabulary.json")
    embedding_metadata = {
        "glove_path": Path(os.path.relpath(args.glove_path.resolve(), PROJECT_ROOT)).as_posix(),
        "dimension": GLOVE_DIM,
        "vocabulary_size": embedding_result.vocabulary_size,
        "pretrained_vector_count": len(embedding_result.covered_tokens),
        "pretrained_vector_coverage": len(embedding_result.covered_tokens) / max(1, len(vocabulary) - 2),
        "vocabulary_built_from": "union of the four actual fitting datasets only",
    }
    (RESULTS_DIR / "embedding_metadata.json").write_text(
        json.dumps(embedding_metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_text_quality_artifacts(
        train_records,
        validation_records,
        test_records,
        vocabulary,
        embedding_result.covered_tokens,
    )

    device = choose_device(args.device)
    print(f"Training on {device}; loading four conditions from the shared CNN archive.", flush=True)
    set_global_seed(SEED)
    initial_model = TextLSTM(embedding_result.matrix.clone())
    initial_state = copy.deepcopy(initial_model.state_dict())
    torch.save(
        {"model_state": initial_state, "seed": SEED, "purpose": "common initial state"},
        CHECKPOINT_DIR / "initial_state.pt",
    )

    validation_dataset = TextClassificationDataset(validation_records, vocabulary, MAX_SEQUENCE_LENGTH)
    test_dataset = TextClassificationDataset(test_records, vocabulary, MAX_SEQUENCE_LENGTH)
    metrics: list[dict[str, Any]] = []
    histories: dict[str, Sequence[Mapping[str, float]]] = {}
    for condition in CONDITIONS:
        print(f"Starting {DISPLAY_NAMES[condition]}...", flush=True)
        model = TextLSTM(embedding_result.matrix.clone())
        model.load_state_dict(initial_state)
        train_dataset = TextClassificationDataset(train_records[condition], vocabulary, MAX_SEQUENCE_LENGTH)
        checkpoint_path = CHECKPOINT_DIR / f"{condition}_best.pt"
        result = train_one_condition(
            model,
            train_dataset,
            validation_dataset,
            test_dataset,
            checkpoint_path,
            learning_rate=LEARNING_RATE,
            batch_size=BATCH_SIZE,
            max_epochs=MAX_EPOCHS,
            patience=PATIENCE,
            seed=SEED,
            device=device,
            checkpoint_metadata={
                "condition": condition,
                "display_name": DISPLAY_NAMES[condition],
                "seed": SEED,
                "train_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "test_samples": len(test_dataset),
            },
        )
        histories[condition] = result["history"]
        with (RESULTS_DIR / f"history_{condition}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "validation_loss"])
            writer.writeheader()
            writer.writerows(result["history"])
        metrics.append(
            {
                "condition": condition,
                "method": DISPLAY_NAMES[condition],
                "train_samples": len(train_dataset),
                "validation_samples": len(validation_dataset),
                "test_samples": len(test_dataset),
                "best_epoch": result["best_epoch"],
                "epochs_completed": result["epochs_completed"],
                "best_validation_loss": result["best_validation_loss"],
                "test_loss": result["test_loss"],
                "test_accuracy": result["test_accuracy"],
                "test_accuracy_pct": 100.0 * result["test_accuracy"],
                "training_time_seconds": result["training_time_seconds"],
                "checkpoint": checkpoint_path.relative_to(PROJECT_ROOT).as_posix(),
            }
        )
        print(
            f"{DISPLAY_NAMES[condition]}: {100.0 * result['test_accuracy']:.2f}% "
            f"(best epoch {result['best_epoch']}, {result['training_time_seconds']:.1f}s)"
        )

    baseline_accuracy = metrics[0]["test_accuracy_pct"]
    for metric in metrics:
        metric["improvement_vs_baseline_pp"] = metric["test_accuracy_pct"] - baseline_accuracy
    write_csv(
        RESULTS_DIR / "metrics.csv",
        [
            "condition",
            "method",
            "train_samples",
            "validation_samples",
            "test_samples",
            "best_epoch",
            "epochs_completed",
            "best_validation_loss",
            "test_loss",
            "test_accuracy",
            "test_accuracy_pct",
            "improvement_vs_baseline_pp",
            "training_time_seconds",
            "checkpoint",
        ],
        metrics,
    )
    write_csv(
        RESULTS_DIR / "comparison.csv",
        ["method", "test_accuracy_pct", "improvement_vs_baseline_pp"],
        [
            {
                "method": metric["method"],
                "test_accuracy_pct": metric["test_accuracy_pct"],
                "improvement_vs_baseline_pp": "" if metric["condition"] == "baseline" else metric["improvement_vs_baseline_pp"],
            }
            for metric in metrics
        ],
    )
    by_condition = {metric["condition"]: metric for metric in metrics}
    pairwise = [
        {
            "comparison": "EDA → AEDA − EDA",
            "difference_pp": by_condition["eda_aeda"]["test_accuracy_pct"] - by_condition["eda"]["test_accuracy_pct"],
        },
        {
            "comparison": "EDA → AEDA − AEDA",
            "difference_pp": by_condition["eda_aeda"]["test_accuracy_pct"] - by_condition["aeda"]["test_accuracy_pct"],
        },
        {
            "comparison": "AEDA − EDA",
            "difference_pp": by_condition["aeda"]["test_accuracy_pct"] - by_condition["eda"]["test_accuracy_pct"],
        },
    ]
    write_csv(RESULTS_DIR / "pairwise_differences.csv", ["comparison", "difference_pp"], pairwise)
    plot_results(metrics, histories)
    write_cnn_comparison(metrics)
    run_configuration = {
        "seed": SEED,
        "device": str(device),
        "torch_version": str(torch.__version__),
        "identical_data_to_cnn": True,
        "architecture": "frozen GloVe 300d -> BiLSTM(64 per direction, full sequence) -> Dropout(0.5) -> BiLSTM(32 per direction, final states) -> Dropout(0.5) -> Dense(20, ReLU) -> 2 logits",
        "lstm_hidden_sizes_per_direction": [64, 32],
        "bidirectional": True,
        "dropout_after_each_lstm": 0.5,
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "loss": "CrossEntropyLoss",
        "batch_size": BATCH_SIZE,
        "maximum_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "max_sequence_length": MAX_SEQUENCE_LENGTH,
        "same_initial_state_checkpoint": (CHECKPOINT_DIR / "initial_state.pt").relative_to(PROJECT_ROOT).as_posix(),
    }
    (RESULTS_DIR / "run_configuration.json").write_text(
        json.dumps(run_configuration, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_experiment_report(metrics, pairwise, device)
    print(f"Results written to {RESULTS_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glove-path", type=Path, default=default_glove_path())
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()

