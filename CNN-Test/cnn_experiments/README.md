# CNN Experiment A implementation

Run these modules from the `CNN-Test/` directory. The repository's original five data files are resolved from its parent directory, and the official AEDA checkout defaults to a sibling of the repository named `aeda_nlp_reference`.

```powershell
python -m pip install -r cnn_experiments/requirements.txt
python -m cnn_experiments.download_glove
python -m cnn_experiments.smoke_test
python -m cnn_experiments.prepare_data
python -m cnn_experiments.run_experiments --skip-prepare
```

The last command trains four CNNs and rewrites results. Use a separate clone to reproduce the archived run. See the [project README](../README.md) for the source checkout, complete method, results, and limitations. Outputs are stored in `results/`, `plots/`, and `checkpoints/` at the `CNN-Test/` level.
