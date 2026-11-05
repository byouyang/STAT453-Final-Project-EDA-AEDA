"""Prepare leakage-safe, source-linked datasets for all Experiment A conditions.

The 500-source split is applied before EDA/EDA->AEDA rows are selected, so no
augmented sibling of a validation example can reach any training loader.  AEDA
uses the official implementation from the separately cloned AEDA repository.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

try:
    from .config import (
        BASELINE_FILE,
        EDA_FILE,
        GROUP_AEDA_TRAIN_FILE,
        PREPARED_DIR,
        PREPARED_FILENAMES,
        PROJECT_ROOT,
        RESULTS_DIR,
        SEED,
        SEQUENTIAL_FILE,
        VALIDATION_FOLDS,
        default_aeda_repo,
    )
    from .text_utils import overlap_keys, punctuation_insensitive_key
except ImportError:
    from config import (
        BASELINE_FILE,
        EDA_FILE,
        GROUP_AEDA_TRAIN_FILE,
        PREPARED_DIR,
        PREPARED_FILENAMES,
        PROJECT_ROOT,
        RESULTS_DIR,
        SEED,
        SEQUENTIAL_FILE,
        VALIDATION_FOLDS,
        default_aeda_repo,
    )
    from text_utils import overlap_keys, punctuation_insensitive_key


@dataclass(frozen=True)
class Record:
    uid: str
    label: int
    text: str
    source_id: str
    source_index: int
    original_text: str
    augmentation_index: int | None = None


def read_tsv(path: Path, prefix: str) -> list[Record]:
    """Load label<TAB>text rows without discarding punctuation or interior space."""
    if not path.is_file():
        raise FileNotFoundError(f"Required data file does not exist: {path}")
    records: list[Record] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.rstrip("\r\n")
            try:
                raw_label, text = line.split("\t", 1)
            except ValueError as exc:
                raise ValueError(f"{path}:{line_number} is not label<TAB>text") from exc
            if raw_label not in {"0", "1"}:
                raise ValueError(f"{path}:{line_number} has non-binary label {raw_label!r}")
            uid = f"{prefix}:{line_number}"
            records.append(
                Record(
                    uid=uid,
                    label=int(raw_label),
                    text=text,
                    source_id=uid,
                    source_index=line_number,
                    original_text=text,
                )
            )
    if not records:
        raise ValueError(f"{path} is empty")
    return records


def record_keysets(records: Iterable[Record]) -> dict[str, set[str]]:
    values = {"canonical": set(), "punctuation_insensitive": set()}
    for record in records:
        for name, key in overlap_keys(record.text).items():
            if key:
                values[name].add(key)
    return values


def matched_normalizations(record: Record, forbidden_keys: dict[str, set[str]]) -> list[str]:
    return [
        name
        for name, key in overlap_keys(record.text).items()
        if key and key in forbidden_keys[name]
    ]


def filter_forbidden(
    records: Sequence[Record], forbidden_keys: dict[str, set[str]]
) -> tuple[list[Record], Counter[str]]:
    kept: list[Record] = []
    reasons: Counter[str] = Counter()
    for record in records:
        matches = matched_normalizations(record, forbidden_keys)
        if matches:
            reasons.update(matches)
        else:
            kept.append(record)
    return kept, reasons


def overlap_count(left: Sequence[Record], right: Sequence[Record], normalization: str) -> int:
    left_keys = record_keysets(left)[normalization]
    right_keys = record_keysets(right)[normalization]
    return len(left_keys & right_keys)


def split_common_sources(records: Sequence[Record]) -> tuple[list[Record], list[Record]]:
    """90/10 stratified split that keeps normalized duplicate sources together.

    `StratifiedGroupKFold` can produce an imbalanced individual fold when every
    source is a singleton.  Splitting the source groups themselves with
    `StratifiedShuffleSplit` gives the requested 90/10 class balance while
    still preventing a repeated normalized sentence from straddling the split.
    """
    grouped: dict[str, list[Record]] = {}
    for record in records:
        grouped.setdefault(punctuation_insensitive_key(record.text) or record.uid, []).append(record)
    group_keys = list(grouped)
    group_labels: list[int] = []
    for group_key in group_keys:
        labels = {record.label for record in grouped[group_key]}
        if len(labels) != 1:
            raise ValueError(f"Conflicting labels within normalized source group {group_key!r}")
        group_labels.append(labels.pop())
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=1 / VALIDATION_FOLDS, random_state=SEED)
    train_group_indices, validation_group_indices = next(
        splitter.split(np.zeros(len(group_keys)), np.asarray(group_labels))
    )
    train_group_keys = {group_keys[index] for index in train_group_indices.tolist()}
    validation_group_keys = {group_keys[index] for index in validation_group_indices.tolist()}
    if train_group_keys & validation_group_keys:
        raise AssertionError("Invalid common source split")
    train_set = {
        record.uid for group_key in train_group_keys for record in grouped[group_key]
    }
    validation_set = {
        record.uid for group_key in validation_group_keys for record in grouped[group_key]
    }
    if train_set & validation_set or len(train_set) + len(validation_set) != len(records):
        raise AssertionError("Invalid expanded common source split")
    return (
        [record for record in records if record.uid in train_set],
        [record for record in records if record.uid in validation_set],
    )


def link_augmented_blocks(
    path: Path,
    source_records: Sequence[Record],
    prefix: str,
    *,
    require_original_copy: bool,
) -> list[Record]:
    """Link six contiguous augmentation rows to each original 500-source row."""
    augmented_rows = read_tsv(path, f"{prefix}_raw")
    if len(augmented_rows) % len(source_records) != 0:
        raise ValueError(
            f"{path} has {len(augmented_rows)} rows, not an integral number of source blocks."
        )
    block_size = len(augmented_rows) // len(source_records)
    if block_size != 6:
        raise ValueError(f"Expected 6 rows/source in {path}; found {block_size}.")

    linked: list[Record] = []
    for source_position, source in enumerate(source_records):
        block = augmented_rows[source_position * block_size : (source_position + 1) * block_size]
        if any(row.label != source.label for row in block):
            raise ValueError(f"Label mismatch in {prefix} block for source {source.source_index}.")
        if require_original_copy:
            original_key = overlap_keys(source.text)["canonical"]
            if not any(overlap_keys(row.text)["canonical"] == original_key for row in block):
                raise ValueError(
                    f"EDA block {source.source_index} has no unchanged copy of its original source."
                )
        for augmentation_index, row in enumerate(block):
            linked.append(
                Record(
                    uid=f"{prefix}:{source.source_index}:{augmentation_index}",
                    label=row.label,
                    text=row.text,
                    source_id=source.source_id,
                    source_index=source.source_index,
                    original_text=source.text,
                    augmentation_index=augmentation_index,
                )
            )
    return linked


def sample_per_label(
    records: Sequence[Record], target_counts: Counter[int], rng: np.random.Generator
) -> list[Record]:
    """Deterministically subsample while matching supplied binary class totals."""
    selected_positions: set[int] = set()
    for label in (0, 1):
        positions = [index for index, record in enumerate(records) if record.label == label]
        target = int(target_counts[label])
        if len(positions) < target:
            raise ValueError(
                f"Need {target} class-{label} records but only {len(positions)} are eligible."
            )
        chosen = rng.choice(np.asarray(positions), size=target, replace=False)
        selected_positions.update(int(position) for position in chosen)
    return [record for index, record in enumerate(records) if index in selected_positions]


def load_official_aeda_function(aeda_repo: Path):
    module_path = aeda_repo / "code" / "aeda.py"
    if not module_path.is_file():
        raise FileNotFoundError(f"Official AEDA implementation not found: {module_path}")
    specification = importlib.util.spec_from_file_location("official_aeda", module_path)
    if specification is None or specification.loader is None:
        raise ImportError(f"Could not import {module_path}")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    module.random.seed(SEED)
    return module.insert_punctuation_marks


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_source_path(path: Path) -> str:
    """Record provenance relative to CNN-Test instead of a contributor's machine."""
    return Path(os.path.relpath(path, PROJECT_ROOT)).as_posix()


def write_jsonl(path: Path, records: Sequence[Record]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_prepared_records(name: str) -> list[dict[str, Any]]:
    path = PREPARED_DIR / PREPARED_FILENAMES[name]
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(dataset: str, split: str, records: Sequence[Record]) -> dict[str, Any]:
    counts = Counter(record.label for record in records)
    return {
        "dataset": dataset,
        "split": split,
        "samples": len(records),
        "label_0": counts[0],
        "label_1": counts[1],
    }


def write_overlap_report(comparisons: Sequence[tuple[str, Sequence[Record], Sequence[Record], str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, left, right, action in comparisons:
        for normalization in ("canonical", "punctuation_insensitive"):
            rows.append(
                {
                    "comparison": name,
                    "normalization": normalization,
                    "unique_text_overlap": overlap_count(left, right, normalization),
                    "action": action,
                }
            )
    write_csv(
        RESULTS_DIR / "overlap_report.csv",
        ["comparison", "normalization", "unique_text_overlap", "action"],
        rows,
    )
    return rows


def require_no_overlap(name: str, left: Sequence[Record], right: Sequence[Record]) -> None:
    for normalization in ("canonical", "punctuation_insensitive"):
        count = overlap_count(left, right, normalization)
        if count:
            raise AssertionError(f"{name}: {count} {normalization} overlap(s) remain")


def prepare_all(aeda_repo: Path) -> dict[str, Any]:
    """Build every dataset artifact and return an inspectable preparation manifest."""
    aeda_repo = aeda_repo.resolve()
    official_train_path = aeda_repo / "data" / "train" / "sst2" / "train.txt"
    official_test_path = aeda_repo / "data" / "test" / "sst2" / "test.txt"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    baseline_sources = read_tsv(BASELINE_FILE, "baseline_source")
    test_records = read_tsv(official_test_path, "test")
    official_aeda_sources = read_tsv(official_train_path, "official_aeda_source")
    group_aeda_sources = read_tsv(GROUP_AEDA_TRAIN_FILE, "group_aeda_source")
    if len(official_aeda_sources) != 7791 or len(test_records) != 1821:
        raise ValueError(
            f"Expected official AEDA SST-2 sizes 7791/1821, found "
            f"{len(official_aeda_sources)}/{len(test_records)}."
        )

    group_matches_official = [
        (left.label, left.text) == (right.label, right.text)
        for left, right in zip(group_aeda_sources, official_aeda_sources)
    ]
    group_aeda_identical = (
        len(group_aeda_sources) == len(official_aeda_sources) and all(group_matches_official)
    )

    test_keys = record_keysets(test_records)
    retained_baseline_sources, baseline_test_removals = filter_forbidden(baseline_sources, test_keys)
    common_train_sources, validation_records = split_common_sources(retained_baseline_sources)
    validation_keys = record_keysets(validation_records)
    train_source_ids = {record.source_id for record in common_train_sources}

    eda_all = link_augmented_blocks(
        EDA_FILE, baseline_sources, "eda", require_original_copy=True
    )
    sequential_all = link_augmented_blocks(
        SEQUENTIAL_FILE, baseline_sources, "eda_aeda", require_original_copy=False
    )
    eda_source_train = [record for record in eda_all if record.source_id in train_source_ids]
    sequential_source_train = [record for record in sequential_all if record.source_id in train_source_ids]

    combined_forbidden = {
        name: test_keys[name] | validation_keys[name]
        for name in test_keys
    }
    eda_allowed, eda_text_removals = filter_forbidden(eda_source_train, combined_forbidden)
    sequential_allowed, sequential_text_removals = filter_forbidden(
        sequential_source_train, combined_forbidden
    )
    shared_fit_targets = Counter(
        {
            label: min(
                Counter(record.label for record in eda_allowed)[label],
                Counter(record.label for record in sequential_allowed)[label],
            )
            for label in (0, 1)
        }
    )
    if not all(shared_fit_targets.values()):
        raise ValueError("Leakage filtering left an empty class in EDA or sequential training data.")
    eda_fit = sample_per_label(eda_allowed, shared_fit_targets, np.random.default_rng(SEED + 101))
    sequential_fit = sample_per_label(
        sequential_allowed, shared_fit_targets, np.random.default_rng(SEED + 202)
    )

    aeda_source_forbidden = {
        name: test_keys[name] | validation_keys[name]
        for name in test_keys
    }
    aeda_eligible_sources, aeda_source_removals = filter_forbidden(
        official_aeda_sources, aeda_source_forbidden
    )
    eda_candidate_targets = Counter(record.label for record in eda_all)
    sampling_rng = np.random.default_rng(SEED)
    aeda_candidate_sources = sample_per_label(
        aeda_eligible_sources, eda_candidate_targets, sampling_rng
    )
    official_aeda = load_official_aeda_function(aeda_repo)
    aeda_candidates = [
        Record(
            uid=f"aeda:{source.source_index}",
            label=source.label,
            text=official_aeda(source.text),
            source_id=source.source_id,
            source_index=source.source_index,
            original_text=source.text,
            augmentation_index=0,
        )
        for source in aeda_candidate_sources
    ]
    # A final guard catches the unlikely case where an augmentation itself becomes
    # a validation/test string despite a non-overlapping raw source.
    aeda_allowed, aeda_text_removals = filter_forbidden(aeda_candidates, combined_forbidden)
    aeda_fit = sample_per_label(aeda_allowed, shared_fit_targets, sampling_rng)

    # Required leakage-free postconditions on exact and punctuation-insensitive text.
    require_no_overlap("baseline train vs validation", common_train_sources, validation_records)
    require_no_overlap("validation vs test", validation_records, test_records)
    for name, records in (("EDA", eda_fit), ("AEDA", aeda_fit), ("EDA→AEDA", sequential_fit)):
        require_no_overlap(f"{name} train vs validation", records, validation_records)
        require_no_overlap(f"{name} train vs test", records, test_records)

    prepared = {
        "baseline": common_train_sources,
        "eda": eda_fit,
        "aeda": aeda_fit,
        "eda_aeda": sequential_fit,
        "validation": validation_records,
        "test": test_records,
    }
    for name, records in prepared.items():
        write_jsonl(PREPARED_DIR / PREPARED_FILENAMES[name], records)

    aeda_fit_uids = {record.uid for record in aeda_fit}
    write_csv(
        RESULTS_DIR / "aeda_candidate_manifest.csv",
        [
            "official_source_index",
            "source_id",
            "label",
            "original_sentence",
            "augmented_sentence",
            "selected_for_model_fit",
        ],
        [
            {
                "official_source_index": record.source_index,
                "source_id": record.source_id,
                "label": record.label,
                "original_sentence": record.original_text,
                "augmented_sentence": record.text,
                "selected_for_model_fit": record.uid in aeda_fit_uids,
            }
            for record in aeda_candidates
        ],
    )
    write_csv(
        RESULTS_DIR / "source_split_manifest.csv",
        ["source_index", "source_id", "label", "split", "original_sentence"],
        [
            {
                "source_index": record.source_index,
                "source_id": record.source_id,
                "label": record.label,
                "split": "train",
                "original_sentence": record.text,
            }
            for record in common_train_sources
        ]
        + [
            {
                "source_index": record.source_index,
                "source_id": record.source_id,
                "label": record.label,
                "split": "validation",
                "original_sentence": record.text,
            }
            for record in validation_records
        ],
    )

    summary_rows = [
        summarize("Baseline", "candidate_original_sources", baseline_sources),
        summarize("Baseline", "train", common_train_sources),
        summarize("Common", "validation", validation_records),
        summarize("EDA", "candidate_before_split", eda_all),
        summarize("EDA", "eligible_after_split_and_overlap_checks", eda_allowed),
        summarize("EDA", "train", eda_fit),
        summarize("AEDA", "candidate", aeda_candidates),
        summarize("AEDA", "eligible_after_overlap_checks", aeda_allowed),
        summarize("AEDA", "train", aeda_fit),
        summarize("EDA → AEDA", "candidate_before_split", sequential_all),
        summarize("EDA → AEDA", "eligible_after_split_and_overlap_checks", sequential_allowed),
        summarize("EDA → AEDA", "train", sequential_fit),
        summarize("Common", "test", test_records),
    ]
    write_csv(
        RESULTS_DIR / "dataset_summary.csv",
        ["dataset", "split", "samples", "label_0", "label_1"],
        summary_rows,
    )
    overlap_rows = write_overlap_report(
        [
            ("original_500_sources_vs_test_before_filter", baseline_sources, test_records, "excluded from baseline source pool before split"),
            ("official_aeda_sources_vs_test_before_filter", official_aeda_sources, test_records, "excluded before AEDA sampling"),
            ("official_aeda_sources_vs_validation_before_filter", official_aeda_sources, validation_records, "excluded before AEDA sampling"),
            ("baseline_train_vs_validation", common_train_sources, validation_records, "must be zero"),
            ("validation_vs_test", validation_records, test_records, "must be zero"),
            ("eda_train_vs_validation", eda_fit, validation_records, "must be zero"),
            ("eda_train_vs_test", eda_fit, test_records, "must be zero"),
            ("aeda_train_vs_validation", aeda_fit, validation_records, "must be zero"),
            ("aeda_train_vs_test", aeda_fit, test_records, "must be zero"),
            ("sequential_train_vs_validation", sequential_fit, validation_records, "must be zero"),
            ("sequential_train_vs_test", sequential_fit, test_records, "must be zero"),
        ]
    )

    manifest: dict[str, Any] = {
        "seed": SEED,
        "validation_strategy": "grouped normalized sources + StratifiedShuffleSplit(test_size=0.1, random_state=42)",
        "aeda_sampling_strategy": "class-targeted random sampling with numpy Generator seed 42",
        "aeda_generation_strategy": "official aeda_nlp/code/aeda.py insert_punctuation_marks with random seed 42",
        "group_aeda_train_matches_official": group_aeda_identical,
        "counts": {name: len(records) for name, records in prepared.items()},
        "shared_augmentation_fit_class_targets": dict(shared_fit_targets),
        "aeda_candidate_class_targets": dict(eda_candidate_targets),
        "removals": {
            "baseline_sources_due_to_test": dict(baseline_test_removals),
            "eda_augmented_rows_due_to_validation_or_test": dict(eda_text_removals),
            "aeda_sources_due_to_validation_or_test": dict(aeda_source_removals),
            "aeda_augmented_rows_due_to_validation_or_test": dict(aeda_text_removals),
            "sequential_augmented_rows_due_to_validation_or_test": dict(sequential_text_removals),
        },
        "unique_rows_removed": {
            "baseline_sources_due_to_test": len(baseline_sources) - len(retained_baseline_sources),
            "eda_augmented_rows_due_to_validation_or_test": len(eda_source_train) - len(eda_allowed),
            "aeda_sources_due_to_validation_or_test": len(official_aeda_sources) - len(aeda_eligible_sources),
            "aeda_augmented_rows_due_to_validation_or_test": len(aeda_candidates) - len(aeda_allowed),
            "sequential_augmented_rows_due_to_validation_or_test": len(sequential_source_train) - len(sequential_allowed),
        },
        "source_files": {
            "baseline": {"path": portable_source_path(BASELINE_FILE), "sha256": hash_file(BASELINE_FILE)},
            "eda": {"path": portable_source_path(EDA_FILE), "sha256": hash_file(EDA_FILE)},
            "sequential": {"path": portable_source_path(SEQUENTIAL_FILE), "sha256": hash_file(SEQUENTIAL_FILE)},
            "official_aeda_train": {"path": portable_source_path(official_train_path), "sha256": hash_file(official_train_path)},
            "official_aeda_test": {"path": portable_source_path(official_test_path), "sha256": hash_file(official_test_path)},
        },
        "overlap_checks": overlap_rows,
    }
    (RESULTS_DIR / "data_preparation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aeda-repo",
        type=Path,
        default=default_aeda_repo(),
        help="Path to a clone of https://github.com/akkarimi/aeda_nlp",
    )
    args = parser.parse_args()
    manifest = prepare_all(args.aeda_repo)
    counts = manifest["counts"]
    print(
        "Prepared datasets: "
        + ", ".join(f"{name}={count}" for name, count in counts.items())
    )
    print(f"Leakage report: {RESULTS_DIR / 'overlap_report.csv'}")


if __name__ == "__main__":
    main()

