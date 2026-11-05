# STAT 453 Experiment A — CNN Results

This report was generated from the actual fixed-seed run. All accuracy deltas below are percentage points (pp).

## CNN Test Accuracy

| Method | Train samples | Best epoch | Best validation loss | Test accuracy | Improvement vs. Baseline |
|---|---:|---:|---:|---:|---:|
| Baseline | 450 | 9 | 0.5557 | 74.57% | — |
| EDA | 2700 | 1 | 0.5924 | 72.32% | -2.25 pp |
| AEDA | 2700 | 2 | 0.4506 | 78.47% | +3.90 pp |
| EDA → AEDA | 2700 | 1 | 0.5717 | 73.92% | -0.66 pp |

## Pairwise Augmentation Differences

| Comparison | Difference |
|---|---:|
| EDA → AEDA − EDA | +1.59 pp |
| EDA → AEDA − AEDA | -4.56 pp |
| AEDA − EDA | +6.15 pp |

## Dataset and Leakage Verification

- Common validation data were drawn once from the 500 original sources with stratified group splitting (seed 42).
- The EDA and EDA → AEDA candidate files each had 3,000 rows; source-linked validation siblings were excluded before fitting.
- AEDA used 3,000 sampled, official-method punctuation augmentations and was subset to the same post-filter class totals as EDA and EDA → AEDA.
- `overlap_report.csv` reports both canonical and punctuation-insensitive checks; all final training-versus-validation/test cells must be zero.
- `punctuation_check.csv` and `text_quality.csv` document token preservation, GloVe coverage, and truncation at 50 tokens.

## Original EDA Paper Reference

| Model | N=500 | N=2,000 | N=5,000 | Full |
|---|---:|---:|---:|---:|
| CNN without EDA | 78.6% | 85.6% | 87.7% | 88.3% |
| CNN with EDA | 80.7% | 86.4% | 88.3% | 88.8% |

The paper values are averages across five datasets and five random seeds, not SST-2-only results. This run is one SST-2 experiment with a shared 50-source validation set and a 1,821-example fixed test set, so the comparison is directional only.

## Reproducibility

- Device: `cpu`
- Seed: `42`; Adam learning rate `0.001`; batch size `64`; up to `30` epochs; patience `3`.
- Model: frozen GloVe 300d → Conv1D(128, kernel 5) → global max pool → Dense(20, ReLU) → 2 raw logits.
- See `metrics.csv`, `comparison.csv`, `pairwise_differences.csv`, `../checkpoints/`, and `../plots/` for the underlying outputs.
