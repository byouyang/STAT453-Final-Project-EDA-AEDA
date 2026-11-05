"""Reuse and recheck the exact archived CNN splits for a matched comparison."""

import csv
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

from .config import CNN_ROOT, CONDITIONS, PREPARED_DIR, PREPARED_FILENAMES, RESULTS_DIR
from .text_utils import overlap_keys


def write_csv(path, fieldnames, rows):
    with Path(path).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_prepared_records(name):
    path = PREPARED_DIR / PREPARED_FILENAMES[name]
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def prepare_all():
    source = CNN_ROOT / "results" / "prepared"
    expected_counts = dict(baseline=450, eda=2700, aeda=2700, eda_aeda=2700, validation=50, test=1821)
    splits, contents, files = {}, {}, {}
    for name, filename in PREPARED_FILENAMES.items():
        path = source / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing archived CNN split: {path}. Restore CNN-Test/results/prepared first.")
        raw = path.read_bytes()
        records = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
        if len(records) != expected_counts[name]:
            raise ValueError(f"Unexpected {name} size: {len(records)}; expected {expected_counts[name]}.")
        if any(row.get("label") not in (0, 1) or not isinstance(row.get("text"), str) for row in records):
            raise ValueError(f"Invalid text or labels in {path}")
        splits[name], contents[name] = records, raw
        files[name] = {"source": f"../CNN-Test/results/prepared/{filename}", "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(records)}

    target_counts = Counter(row["label"] for row in splits["eda"])
    if any(Counter(row["label"] for row in splits[name]) != target_counts for name in ("aeda", "eda_aeda")):
        raise ValueError("Augmented class counts do not match.")
    comparisons = [(name, heldout) for name in CONDITIONS for heldout in ("validation", "test")]
    comparisons.append(("validation", "test"))
    overlaps = []
    for left, right in comparisons:
        for normalization in ("canonical", "punctuation_insensitive"):
            left_keys = {overlap_keys(row["text"])[normalization] for row in splits[left]}
            right_keys = {overlap_keys(row["text"])[normalization] for row in splits[right]}
            count = len(left_keys & right_keys)
            if count:
                raise ValueError(f"{left} vs {right}: {count} {normalization} overlaps.")
            overlaps.append(dict(comparison=f"{left}_vs_{right}", normalization=normalization, unique_text_overlap=count))

    # Check source siblings as well as text equality for source-linked conditions.
    validation_sources = {row["source_id"] for row in splits["validation"]}
    for name in ("baseline", "eda", "eda_aeda"):
        if validation_sources & {row["source_id"] for row in splits[name]}:
            raise ValueError(f"Validation source siblings leaked into {name}.")

    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    for name, raw in contents.items():
        (PREPARED_DIR / PREPARED_FILENAMES[name]).write_bytes(raw)
    manifest = {
        "strategy": "Byte-for-byte reuse of archived CNN fitting and evaluation splits",
        "files": files,
        "limitation": "AEDA uses a larger original source pool than the other conditions; this confound is retained to match CNN.",
    }
    (RESULTS_DIR / "data_preparation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    original_manifest = CNN_ROOT / "results" / "data_preparation_manifest.json"
    if original_manifest.is_file():
        shutil.copyfile(original_manifest, RESULTS_DIR / "cnn_data_preparation_manifest.json")
    write_csv(RESULTS_DIR / "overlap_report.csv", ["comparison", "normalization", "unique_text_overlap"], overlaps)
    rows = [dict(dataset=name, samples=len(records), negative=sum(row["label"] == 0 for row in records), positive=sum(row["label"] == 1 for row in records)) for name, records in splits.items()]
    write_csv(RESULTS_DIR / "dataset_summary.csv", ["dataset", "samples", "negative", "positive"], rows)
    print("Prepared exact CNN splits: 450 baseline, 2,700 per augmentation, 50 validation, 1,821 test.", flush=True)
    return manifest


if __name__ == "__main__":
    prepare_all()
