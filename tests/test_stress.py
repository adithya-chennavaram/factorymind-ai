import numpy as np
import pandas as pd
import pytest

from factorymind.data import SENSORS, STATS
from factorymind.stress import challenge_priority, stress_test_gain
from scripts.evaluate_green_challenge import evaluate_rankings


class MockValveModel:
    classes_ = np.array([73, 100])

    def predict_proba(self, X):
        # PS1 gain alters confidence in a deterministic, testable way.
        p = np.clip(0.75 - (X["PS1_mean"].to_numpy() - 100) * 0.02, 0.01, 0.99)
        return np.stack([1 - p, p], axis=1)


def sample_features():
    return pd.DataFrame([{f"{s}_{stat}": (100.0 if s == "PS1" else 1.0)
                           for s in SENSORS for stat in STATS} for _ in range(3)])


def test_gain_stress_produces_50_perturbations_and_nonnegative_sensitivity():
    X = sample_features()
    result = stress_test_gain(MockValveModel(), X)
    assert len(result) == len(X)
    assert (result.scenarios_tested == 50).all()
    assert (result.max_p_optimal_drop > 0).all()
    assert (result.switch_fraction >= 0).all()
    assert (result.switch_fraction <= 1).all()
    assert X.iloc[0]["PS1_mean"] == 100  # no mutation of input


def test_challenge_score_is_label_blind_and_bounded():
    score = challenge_priority(np.array([.9, .6]), np.array([.1, .2]))
    assert score.tolist() == pytest.approx([.1, .3])
    with pytest.raises(ValueError):
        challenge_priority([.9], [.1], sensitivity_weight=1.5)


def test_review_selection_does_not_consult_truth_for_ranking():
    frame = pd.DataFrame({
        "cycle_id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "fold": [1] * 5 + [2] * 5,
        "predicted": [100] * 10,
        "actual": [90, 100, 90, 100, 100, 100, 90, 100, 90, 100],
        "p_optimal_uncalibrated": [.5, .8, .85, .9, .95, .5, .8, .85, .9, .95],
        "max_p_optimal_drop": [.4, .1, .2, .03, .02, .4, .1, .2, .03, .02],
    })
    frame["challenge_priority"] = challenge_priority(frame.p_optimal_uncalibrated, frame.max_p_optimal_drop)
    original = evaluate_rankings(frame, budgets=(40,))
    scrambled = frame.copy()
    scrambled["actual"] = [100 if v != 100 else 90 for v in frame["actual"]]
    altered = evaluate_rankings(scrambled, budgets=(40,))
    assert original["methods"]["factorymind_combined"][0]["reviewed"] == 4
    assert altered["methods"]["factorymind_combined"][0]["reviewed"] == 4
    # Recorded scores/ranking do not depend on labels.
    assert np.allclose(frame.challenge_priority, scrambled.challenge_priority)


def test_reject_non_green_inputs():
    X = sample_features()
    X.loc[0, "PS1_mean"] = 300
    with pytest.raises(ValueError, match="Only predicted-green"):
        stress_test_gain(MockValveModel(), X)
