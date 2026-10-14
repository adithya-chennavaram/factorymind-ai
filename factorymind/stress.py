"""Reproducible *simulated sensor-gain* robustness challenges.

IMPORTANT: These tests are hypothetical sensitivity scenarios, not measured sensor
noise, physically validated drift, or proof of a valve-condition diagnosis.
Each gain is applied to ALL six stats of a sensor, consistent with scaling that
sensor's raw values before feature extraction (positive gain).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import SENSORS, STATS

DEFAULT_GAIN_PCTS = (0.5, 1.0, 2.0, 3.0, 5.0)


def _validate_features(X: pd.DataFrame) -> None:
    if not isinstance(X, pd.DataFrame) or X.empty:
        raise ValueError("Expected a nonempty DataFrame of sensor features")
    expected = {f"{sensor}_{stat}" for sensor in SENSORS for stat in STATS}
    if not expected.issubset(X.columns):
        raise ValueError("Missing expected 30 sensor features")
    if not np.isfinite(X[list(expected)].to_numpy(dtype=float)).all():
        raise ValueError("Sensor features must be finite")


def stress_test_gain(
    model,
    X: pd.DataFrame,
    *,
    gain_pcts=DEFAULT_GAIN_PCTS,
    predicted_green_code=100,
) -> pd.DataFrame:
    """Evaluate 5 sensors x 2 directions x N gain magnitudes, vectorized.

    Returns one row per input with base P(green), worst challenged P(green),
    sensitivity (largest probability drop), and fraction of challenged predictions
    switching away from green. Ground-truth labels are neither accepted nor used.

    The scenarios represent bounded multiplicative sensor-calibration errors.
    They do NOT change actual valve condition or simulate progressing failure.
    """
    _validate_features(X)
    gains = tuple(float(v) for v in gain_pcts)
    if not gains or any(not np.isfinite(p) or not 0 < p < 100 for p in gains):
        raise ValueError("Gain percentages must be finite and between 0 and 100")
    if not hasattr(model, 'predict_proba') or not hasattr(model, 'classes_'):
        raise ValueError("Classifier with probabilities and classes_ is required")
    classes = list(model.classes_)
    if predicted_green_code not in classes:
        raise ValueError("Green class missing from model")
    target_ix = classes.index(predicted_green_code)
    base_probs = model.predict_proba(X)
    if base_probs.shape != (len(X), len(classes)) or not np.isfinite(base_probs).all():
        raise ValueError("Invalid base probabilities")
    if not np.all(np.argmax(base_probs, axis=1) == target_ix):
        raise ValueError("Only predicted-green rows may be stress tested")

    variations = []
    scenarios = []
    for sensor in SENSORS:
        cols = [f"{sensor}_{stat}" for stat in STATS]
        for pct in gains:
            for sign in (-1, 1):
                multiplier = 1 + sign * pct / 100
                variant = X.copy()
                variant.loc[:, cols] = variant.loc[:, cols] * multiplier
                variations.append(variant)
                scenarios.append((sensor, sign * pct))
    stacked = pd.concat(variations, ignore_index=True)
    probabilities = model.predict_proba(stacked)
    if not np.isfinite(probabilities).all():
        raise ValueError("Invalid stress-test probabilities")
    n = len(X)
    challenged = probabilities.reshape(len(scenarios), n, len(classes))
    p_optimal = challenged[:, :, target_ix]
    predicted = np.argmax(challenged, axis=2)
    base_green = base_probs[:, target_ix]
    min_p = p_optimal.min(axis=0)
    worst_idx = p_optimal.argmin(axis=0)

    return pd.DataFrame({
        "base_p_optimal": base_green,
        "min_stressed_p_optimal": min_p,
        "max_p_optimal_drop": np.clip(base_green - min_p, 0.0, 1.0),
        "switch_fraction": (predicted != target_ix).mean(axis=0),
        "worst_case_sensor": [scenarios[i][0] for i in worst_idx],
        "worst_case_gain_pct": [scenarios[i][1] for i in worst_idx],
        "scenarios_tested": len(scenarios),
    }, index=X.index)


def challenge_priority(base_p_optimal, max_p_optimal_drop, *, sensitivity_weight=0.5):
    """Prespecified, label-blind risk score: higher = investigate sooner.

    0.5 is a fixed *demonstration hypothesis*, not a fitted/optimal weight.
    Do not tune it against the final blocked folds and then present an unbiased
    improvement claim. An optimized weighting requires nested calibration.
    """
    w = float(sensitivity_weight)
    if not 0 <= w <= 1:
        raise ValueError("Sensitivity weight must be between 0 and 1")
    base = np.asarray(base_p_optimal, dtype=float)
    drop = np.asarray(max_p_optimal_drop, dtype=float)
    if np.any(~np.isfinite(base)) or np.any(~np.isfinite(drop)) or np.any((base < 0) | (base > 1)) or np.any((drop < 0) | (drop > 1)):
        raise ValueError("Expected finite scores between zero and one")
    return (1 - w) * (1 - base) + w * drop
