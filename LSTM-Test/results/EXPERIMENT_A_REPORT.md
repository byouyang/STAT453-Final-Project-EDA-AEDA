# STAT 453 Experiment A — LSTM Results

This report was generated from the actual fixed-seed run. All accuracy deltas below are percentage points (pp).

## LSTM Test Accuracy

| Method | Train samples | Best epoch | Best validation loss | Test accuracy | Improvement vs. Baseline |
|---|---:|---:|---:|---:|---:|
| Baseline | 450 | 11 | 0.5669 | 74.96% | — |
| EDA | 2700 | 2 | 0.5452 | 75.45% | +0.49 pp |
| AEDA | 2700 | 9 | 0.4137 | 77.87% | +2.91 pp |
| EDA → AEDA | 2700 | 1 | 0.6105 | 72.43% | -2.53 pp |

## Pairwise Augmentation Differences

| Comparison | Difference |
|---|---:|
| EDA → AEDA − EDA | -3.02 pp |
| EDA → AEDA − AEDA | -5.44 pp |
| AEDA − EDA | +2.42 pp |

## Dataset and Leakage Verification

- Common validation data were drawn once from the 500 original sources with stratified group splitting (seed 42).
- The EDA and EDA → AEDA candidate files each had 3,000 rows; source-linked validation siblings were excluded before fitting.
- AEDA used 3,000 sampled, official-method punctuation augmentations and was subset to the same post-filter class totals as EDA and EDA → AEDA.
- `overlap_report.csv` reports both canonical and punctuation-insensitive checks; all final training-versus-validation/test cells must be zero.
- `punctuation_check.csv` and `text_quality.csv` document token preservation, GloVe coverage, and truncation at 50 tokens.

## Interpretation

This single-seed run has no error bars or significance tests. AEDA uses a larger original source pool than the other conditions, so its difference cannot be attributed solely to punctuation augmentation.

The training, validation, and test rows are copied byte-for-byte from the archived CNN experiment; see data_preparation_manifest.json for hashes. The CNN comparison is descriptive, not a claim of statistical significance.

## Reproducibility

- Device: `cpu`
- Seed: `42`; Adam learning rate `0.001`; batch size `64`; up to `30` epochs; patience `3`.
- Model: frozen GloVe 300d → BiLSTM(64 per direction, full sequence) → Dropout(0.5) → BiLSTM(32 per direction, concatenated final states) → Dropout(0.5) → Dense(20, ReLU) → 2 raw logits.
- Layer architecture matches the EDA reference; packed padding handling and the archived CNN data/training protocol are retained.
- See `metrics.csv`, `comparison.csv`, `pairwise_differences.csv`, `../checkpoints/`, and `../plots/` for the underlying outputs.
