"""Streaming per-cycle feature extraction from the UCI Hydraulic dataset.

Input: one 60-second experimental cycle per line in each sensor file.
Only sensor measurements are used as model features. `profile.txt` is a label
file, NEVER a predictive feature. Each profile row corresponds to the same
line number in the sensor files.
"""
from pathlib import Path
import numpy as np
import pandas as pd

SENSORS = ("PS1", "PS2", "FS1", "TS1", "VS1")
TARGET_LABELS = {100: "Optimal switching", 90: "Small lag", 80: "Severe lag", 73: "Near failure"}
STATS = ("mean", "std", "min", "max", "p95", "change")
SENSOR_INFO = {"PS1": ("Pressure", "bar", 100), "PS2": ("Pressure", "bar", 100),
               "FS1": ("Volume flow", "l/min", 10), "TS1": ("Temperature", "°C", 1),
               "VS1": ("Vibration", "mm/s", 1)}


def summarize_cycle(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError(f"Invalid samples for sensor {prefix}")
    midpoint = max(1, len(values) // 2)
    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_p95": float(np.percentile(values, 95)),
        f"{prefix}_change": float(np.mean(values[midpoint:]) - np.mean(values[:midpoint])) if len(values) > 1 else 0.0,
    }


def extract_features(raw_dir: str | Path, *, strict_uci: bool = False) -> pd.DataFrame:
    raw_dir = Path(raw_dir)
    labels_file = raw_dir / "profile.txt"
    if not labels_file.exists():
        raise FileNotFoundError(f"Missing {labels_file}; run python scripts/download_data.py")
    profile = np.loadtxt(labels_file, dtype=int)
    if profile.ndim != 2 or profile.shape[1] != 5:
        raise ValueError("profile.txt must contain five columns")
    if strict_uci and len(profile) != 2205:
        raise ValueError("Official UCI data must contain 2,205 cycles")
    results = [{"cycle_id": i + 1, "valve_condition": int(row[1]), "stable_flag": int(row[4])}
               for i, row in enumerate(profile)]
    for sensor in SENSORS:
        sensor_path = raw_dir / f"{sensor}.txt"
        if not sensor_path.exists():
            raise FileNotFoundError(f"Missing {sensor_path}")
        count = 0
        with sensor_path.open("r", encoding="utf-8") as handle:
            for count, line in enumerate(handle, start=1):
                if count > len(results):
                    raise ValueError(f"{sensor} has more rows than profile.txt")
                values = np.asarray(line.split(), dtype=float)
                if strict_uci and len(values) != SENSOR_INFO[sensor][2] * 60:
                    raise ValueError(f"Invalid sample count for {sensor} cycle {count}")
                results[count - 1].update(summarize_cycle(values, sensor))
        if count != len(results):
            raise ValueError(f"{sensor} has {count} rows vs {len(results)} profiles")
    df = pd.DataFrame(results)
    if not set(df["valve_condition"].unique()).issubset(TARGET_LABELS):
        raise ValueError("Unexpected valve condition labels")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    # Exclude ALL annotation fields and ids: they are not sensor measurements.
    return [f"{s}_{stat}" for s in SENSORS for stat in STATS if f"{s}_{stat}" in df.columns]


def read_cycle_samples(raw_dir: str | Path, sensor: str, cycle_id: int) -> np.ndarray:
    if sensor not in SENSORS:
        raise ValueError("Unsupported sensor")
    if cycle_id < 1:
        raise ValueError("cycle_id must be positive")
    with (Path(raw_dir) / f"{sensor}.txt").open(encoding="utf-8") as file:
        for idx, line in enumerate(file, 1):
            if idx == cycle_id:
                values = np.asarray(line.split(), dtype=float)
                if not len(values) or not np.isfinite(values).all():
                    raise ValueError("Invalid raw sensor samples")
                return values
    raise IndexError("Cycle not found")
