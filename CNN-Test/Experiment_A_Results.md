# STAT 453 Experiment A — 完成摘要

已在 `STAT453-Final-Project-EDA-AEDA` 仓库中完成、验证并重复运行固定配置的四组 PyTorch CNN 实验。

| 方法 | 实际训练样本 | Best Epoch | Test Accuracy | 相对 Baseline |
|---|---:|---:|---:|---:|
| Baseline | 450 | 9 | 74.57% | — |
| EDA | 2,700 | 1 | 72.32% | -2.25 pp |
| AEDA | 2,700 | 2 | 78.47% | +3.90 pp |
| EDA → AEDA | 2,700 | 1 | 73.92% | -0.66 pp |

Pairwise differences: Sequential − EDA = +1.59 pp; Sequential − AEDA = -4.56 pp; AEDA − EDA = +6.15 pp.

## 数据验证

- 500 个原始来源经 seed=42 的分层、来源分组划分为 450 train / 50 validation（train: 220 negative、230 positive；validation: 24 negative、26 positive）。
- 共同、未经增强的 AEDA test set 有 1,821 条（912 negative、909 positive）。
- EDA、AEDA、EDA → AEDA 候选集均为 3,000；最终各为 2,700 条，并具有相同的 1,320 negative / 1,380 positive 拟合分布。
- 官方 AEDA 训练来源在抽样前排除了 33 条与 validation/test 的规范化文本重叠的来源；所有最终 train-vs-validation/test 的 canonical 与 punctuation-insensitive overlap 都为 0。
- AEDA 标点在 tokenizer 中保留；`. , ! ? ; :` 均有 GloVe 向量。详见仓库的 `results/punctuation_check.csv` 和 `results/text_quality.csv`。

## 主要产物位置

在仓库根目录的 `CNN-Test/` 下：

- `cnn_experiments/`：可复现的准备、嵌入、CNN、训练、评估、绘图和 smoke-test 代码。
- `results/EXPERIMENT_A_REPORT.md`：完整实验报告。
- `results/metrics.csv`、`comparison.csv`、`pairwise_differences.csv`：数值结果。
- `results/overlap_report.csv`、`dataset_summary.csv`、`aeda_candidate_manifest.csv`：数据与泄漏审计。
- `plots/`：三张实际结果图；`checkpoints/`：四组最佳 checkpoint 及共同初始状态。

原始 EDA 论文 Table 2 中的 CNN N=500 平均结果为 78.6%（无 EDA）和 80.7%（EDA），即 +2.1 pp；这些是跨五个数据集、五个随机种子的平均值，不能作为本次单次 SST-2 运行的直接复现对比。

