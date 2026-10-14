import pandas as pd
import pytest
from scripts.evaluate_green_review import evaluate


def sample():
    return pd.DataFrame({
        "cycle_id": list(range(1, 11)),
        "actual": [100, 90, 100, 90, 100, 100, 90, 100, 90, 100],
        "predicted": [100] * 10,
        "fold": [1]*5+[2]*5,
        "p_optimal_uncalibrated": [.95,.60,.70,.55,.90,.95,.40,.85,.20,.99],
    })


def test_review_selection_never_uses_ground_truth_to_rank():
    df = sample()
    original = evaluate(df, budgets=(40,))
    altered = df.copy()
    altered["actual"] = [90,100,90,100,90,90,100,90,100,90]
    changed = evaluate(altered, budgets=(40,))
    assert original["budgets"][0]["reviewed"] == changed["budgets"][0]["reviewed"] == 4
    assert original["budgets"][0]["false_green_caught"] == 4
    assert changed["budgets"][0]["false_green_caught"] == 0


def test_missing_proba_fails_closed():
    with pytest.raises(ValueError, match="retrain"):
        evaluate(sample().drop(columns=["p_optimal_uncalibrated"]))


def test_non_probability_fails_closed():
    df = sample()
    df.loc[1, "p_optimal_uncalibrated"] = 10
    with pytest.raises(ValueError): evaluate(df)
