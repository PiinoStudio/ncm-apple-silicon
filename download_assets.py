#!/usr/bin/env python3
"""
Downloads the required NCM checkpoints, datasets, and baseline weights from Hugging Face.
"""
import os
import subprocess
from huggingface_hub import snapshot_download

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    print("=" * 60)
    print("Next-Concept Modeling (NCM) - Asset Downloader")
    print("=" * 60)

    # Clone external/ncm if missing
    external_ncm = os.path.join(ROOT_DIR, "external", "ncm")
    if not os.path.exists(external_ncm):
        print("\n[1/4] Cloning external/ncm repository...")
        os.makedirs(os.path.join(ROOT_DIR, "external"), exist_ok=True)
        subprocess.run(["git", "clone", "https://github.com/HazyResearch/ncm.git", external_ncm], check=True)
    else:
        print("\n[1/4] external/ncm already exists.")

    # Checkpoints
    print("\n[2/4] Downloading NCM checkpoints (xsum & sudoku/hard)...")
    artifacts_dir = os.path.join(ROOT_DIR, "artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    snapshot_download(
        repo_id="hazyresearch/ncm-checkpoints",
        allow_patterns=["xsum/*", "sudoku/hard/*"],
        local_dir=artifacts_dir
    )

    # Symlink sudoku-hard -> sudoku/hard
    sudoku_hard_link = os.path.join(artifacts_dir, "sudoku-hard")
    if not os.path.exists(sudoku_hard_link):
        try:
            os.symlink("sudoku/hard", sudoku_hard_link)
        except OSError:
            pass

    # Datasets
    print("\n[3/4] Downloading NCM datasets...")
    data_dir = os.path.join(ROOT_DIR, "data")
    os.makedirs(data_dir, exist_ok=True)
    snapshot_download(
        repo_id="hazyresearch/ncm-datasets",
        repo_type="dataset",
        allow_patterns=["xsum/*", "sudoku/hard/*"],
        local_dir=data_dir
    )

    data_sudoku_link = os.path.join(data_dir, "sudoku-hard")
    if not os.path.exists(data_sudoku_link):
        try:
            os.symlink("sudoku/hard", data_sudoku_link)
        except OSError:
            pass

    print("\n[4/4] Assets downloaded successfully!")
    print("\nYou can now run the benchmark suite:")
    print("  python benchmark_suite.py --mode all")

if __name__ == "__main__":
    main()
