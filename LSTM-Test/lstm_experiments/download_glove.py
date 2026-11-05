"""Download and extract the public Stanford GloVe 6B 300-dimensional vectors."""

from __future__ import annotations

import argparse
import shutil
import urllib.request
import zipfile
from pathlib import Path

try:
    from .config import EMBEDDING_DIR, default_glove_path
except ImportError:
    from config import EMBEDDING_DIR, default_glove_path


# Direct Stanford distribution endpoint.  It avoids the intermittently slow
# redirect through nlp.stanford.edu while serving the same public archive.
GLOVE_URL = "https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip"
ARCHIVE_NAME = "glove.6B.zip"
VECTOR_NAME = "glove.6B.300d.txt"


def download_with_progress(url: str, destination: Path) -> None:
    downloaded = 0

    def report(block_count: int, block_size: int, total_size: int) -> None:
        nonlocal downloaded
        current = block_count * block_size
        if current - downloaded >= 32 * 1024 * 1024 or (total_size > 0 and current >= total_size):
            downloaded = current
            if total_size > 0:
                print(f"Downloaded {current / 2**20:.0f}/{total_size / 2**20:.0f} MiB")
            else:
                print(f"Downloaded {current / 2**20:.0f} MiB")

    urllib.request.urlretrieve(url, destination, reporthook=report)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=EMBEDDING_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    vector_path = args.output_dir / VECTOR_NAME
    archive_path = args.output_dir / ARCHIVE_NAME

    if vector_path.is_file():
        print(f"GloVe vectors already present: {vector_path}")
        return

    # Partial downloads/extractions must never masquerade as complete vectors.
    print(f"Downloading {GLOVE_URL}")
    partial_archive = archive_path.with_suffix(".zip.part")
    download_with_progress(GLOVE_URL, partial_archive)
    partial_archive.replace(archive_path)
    partial_vectors = vector_path.with_suffix(".txt.part")
    with zipfile.ZipFile(archive_path) as archive:
        if VECTOR_NAME not in archive.namelist():
            raise RuntimeError(f"{VECTOR_NAME} was not found in {archive_path}")
        with archive.open(VECTOR_NAME) as source, partial_vectors.open("wb") as target:
            shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
    partial_vectors.replace(vector_path)
    archive_path.unlink(missing_ok=True)
    print(f"Extracted {vector_path}")


if __name__ == "__main__":
    main()

