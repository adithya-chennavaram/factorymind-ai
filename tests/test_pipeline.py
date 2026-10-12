from pathlib import Path
import numpy as np
from factorymind.data import summarize_cycle, extract_features, feature_columns
from factorymind.train import train
from factorymind.core import predict_cycle, investigation_prompt
from scripts.generate_test_fixture import generate
import joblib


def test_summary():
    vals = np.array([1., 2., 3., 4.])
    result = summarize_cycle(vals, "PS1")
    assert result["PS1_mean"] == 2.5
    assert result["PS1_change"] == 2.0


def test_end_to_end_synthetic_fixture(tmp_path: Path):
    raw = tmp_path / "raw"
    generate(raw, n=48)
    df = extract_features(raw)
    cols = feature_columns(df)
    assert len(df) == 48
    assert len(cols) == 30
    assert "valve_condition" not in cols
    assert "stable_flag" not in cols
    assert "cycle_id" not in cols
    artifacts = tmp_path / "artifacts"
    metrics = train(df, artifacts)
    assert 0 <= metrics["macro_f1"] <= 1
    model = joblib.load(artifacts / "valve_model.joblib")
    case = predict_cycle(model, df, 1)
    assert case["predicted_valve_condition_code"] in (73, 80, 90, 100)
    assert "model_confidence_uncalibrated" in case
    messages = investigation_prompt(case)
    assert "Do not fabricate" in messages[0]["content"]
    assert "sensor_evidence" in messages[1]["content"]
