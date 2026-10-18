"""Opt-in first-run setup for a hosted demo.

The dataset and trained model stay out of Git. A deployment can opt in with
FACTORYMIND_AUTO_SETUP=1; ordinary local runs keep the explicit setup commands.
"""
from pathlib import Path
import os
import subprocess
import sys
import threading
import time


_THREAD_LOCK = threading.Lock()
_LOCK_NAME = ".factorymind_bootstrap.lock"


class BootstrapError(RuntimeError):
    """Public-data/model initialization failed."""


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
    lock_path = root / _LOCK_NAME
    with _THREAD_LOCK:
        owner = False
        for _ in range(1200):
            try:
                lock_path.mkdir()
                owner = True
                break
            except FileExistsError:
                if all(path.is_file() for path in required):
                    return
                try:
                    if time.time() - lock_path.stat().st_mtime > 900:
                        lock_path.rmdir()
                        continue
                except OSError:
                    pass
                time.sleep(0.25)
        if not owner:
            raise BootstrapError("Another initialization is still running; retry after the hosted app finishes starting.")
        try:
            if all(path.is_file() for path in required):
                return
            subprocess.run([sys.executable, "scripts/download_data.py"], cwd=root, check=True)
            subprocess.run([sys.executable, "-m", "factorymind.train"], cwd=root, check=True)
            subprocess.run([sys.executable, "-m", "scripts.evaluate_green_challenge"], cwd=root, check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            raise BootstrapError("Public dataset/model initialization failed; check deployment logs and retry.") from exc
        finally:
            try:
                lock_path.rmdir()
            except OSError:
                pass
