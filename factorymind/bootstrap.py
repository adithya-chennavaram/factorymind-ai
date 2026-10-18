"""Opt-in first-run setup for a hosted demo.

The dataset and trained model stay out of Git. A deployment can opt in with
FACTORYMIND_AUTO_SETUP=1; ordinary local runs keep the explicit setup commands.
"""
from pathlib import Path
import os
import subprocess
import sys


def ensure_assets(root: Path) -> None:
    required = (
        root / "data/raw/profile.txt",
        root / "data/processed/features.csv",
        root / "artifacts/valve_model.joblib",
        root / "artifacts/predictions.csv",
        root / "artifacts/green_challenge_results.json",
        root / "artifacts/green_challenge_scored.csv",
    )
    if all(path.is_file() for path in required):
        return
    if os.getenv("FACTORYMIND_AUTO_SETUP") != "1":
        return
    subprocess.run([sys.executable, "scripts/download_data.py"], cwd=root, check=True)
    subprocess.run([sys.executable, "-m", "factorymind.train"], cwd=root, check=True)
    subprocess.run([sys.executable, "-m", "scripts.evaluate_green_challenge"], cwd=root, check=True)
