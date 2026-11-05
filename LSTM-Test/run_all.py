"""One-command setup, checks, GloVe download, and all four LSTM experiments."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def execute(command, env):
    print("Running: " + " ".join(map(str, command)), flush=True)
    subprocess.run([str(part) for part in command], cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cpu",
                        help="CPU matches the archived CNN run (default).")
    parser.add_argument("--glove-path", type=Path, help="Use an existing GloVe 6B 300d text file.")
    parser.add_argument("--use-current-python", action="store_true",
                        help="Use the active environment; dependencies must already be installed.")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate data and model without downloading GloVe or training experiments.")
    args = parser.parse_args()
    if sys.version_info < (3, 10):
        parser.error("Python 3.10 or newer is required; Python 3.12 matches the CNN environment.")
    # Resolve caller-relative paths before subprocesses change working directory.
    if args.glove_path is not None:
        args.glove_path = args.glove_path.resolve()
        if not args.glove_path.is_file():
            parser.error(f"GloVe file does not exist: {args.glove_path}")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    env.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
    temporary = ROOT / ".cache" / "tmp"
    temporary.mkdir(parents=True, exist_ok=True)
    env["TEMP"] = env["TMP"] = str(temporary)
    env["PIP_CACHE_DIR"] = str(ROOT / ".cache" / "pip")
    if args.use_current_python:
        python = Path(sys.executable)
    else:
        environment = ROOT / ".venv"
        python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not python.is_file():
            print(f"Creating isolated Python environment: {environment}", flush=True)
            execute([sys.executable, "-m", "venv", environment], env)
        # Repair an environment if a previous first-time setup was interrupted.
        if subprocess.run([str(python), "-m", "pip", "--version"], cwd=ROOT, env=env,
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode:
            execute([python, "-m", "ensurepip", "--upgrade"], env)
        execute([python, "-m", "pip", "install", "-r", ROOT / "lstm_experiments" / "requirements.txt"], env)

    execute([python, "-c", "import torch, numpy, matplotlib; print('Dependencies ready; PyTorch', torch.__version__)"], env)
    execute([python, "-m", "lstm_experiments.smoke_test"], env)
    execute([python, "-m", "lstm_experiments.prepare_data"], env)
    if args.check_only:
        print("Checks passed. No experiment training or GloVe download performed.", flush=True)
        return

    glove_path = args.glove_path
    if glove_path is None:
        local = ROOT / "data" / "embeddings" / "glove.6B.300d.txt"
        shared = ROOT.parent / "CNN-Test" / "data" / "embeddings" / local.name
        glove_path = local if local.is_file() else shared if shared.is_file() else local
        if not glove_path.is_file():
            print("First run downloads the Stanford GloVe archive (~862 MB) and extracts the 300d vectors (~1 GB).", flush=True)
            execute([python, "-m", "lstm_experiments.download_glove"], env)

    execute([python, "-m", "lstm_experiments.run_experiments", "--device", args.device,
             "--glove-path", glove_path], env)
    print(f"All four LSTM experiments complete. Report: {ROOT / 'results' / 'EXPERIMENT_A_REPORT.md'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (subprocess.CalledProcessError, OSError) as error:
        print(f"LSTM runner stopped: {error}", file=sys.stderr)
        sys.exit(1)
