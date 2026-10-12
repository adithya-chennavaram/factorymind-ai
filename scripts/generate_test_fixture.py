"""ONLY for local smoke tests; fake data must never appear in hackathon results."""
from pathlib import Path
import numpy as np


def generate(directory: Path, n: int = 48) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    labels = np.tile([73, 80, 90, 100], (n // 4) + 1)[:n]
    np.savetxt(directory / "profile.txt", np.c_[np.full(n, 100), labels, np.zeros(n), np.full(n, 130), np.zeros(n)], fmt="%d", delimiter="\t")
    rng = np.random.default_rng(9)
    for sensor, length in {"PS1": 50, "PS2": 50, "FS1": 30, "TS1": 20, "VS1": 20}.items():
        data = np.stack([rng.normal(loc=float(v) / 10, scale=0.35, size=length) for v in labels])
        np.savetxt(directory / f"{sensor}.txt", data, delimiter="\t")

if __name__ == "__main__":
    generate(Path("data/toy_only"))
    print("Generated synthetic test fixture. Do NOT use for final evidence or submission.")
