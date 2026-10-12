import numpy as np
import pytest
from factorymind.data import extract_features, feature_columns, summarize_cycle
from factorymind.train import blocked_splits
from scripts.generate_test_fixture import generate


def test_alignment_uses_second_profile_column(tmp_path):
    generate(tmp_path, n=48)
    frame = extract_features(tmp_path)
    labels = np.loadtxt(tmp_path / "profile.txt")
    assert frame.valve_condition.tolist() == labels[:, 1].astype(int).tolist()
    assert frame.cycle_id.tolist() == list(range(1, 49))
    # Even extra annotation columns can never become predictors.
    for name in ("cooler", "pump", "accumulator", "stability"):
        frame[name] = 123
    assert len(feature_columns(frame)) == 30


@pytest.mark.parametrize("kind", ["short", "extra", "nonfinite", "malformed"])
def test_bad_sensor_rows_rejected(tmp_path, kind):
    generate(tmp_path, n=48)
    path = tmp_path / "PS1.txt"
    rows = path.read_text().splitlines()
    if kind == "short":
        rows.pop()
    elif kind == "extra":
        rows.append(rows[-1])
    elif kind == "nonfinite":
        rows[0] = "nan 1 2"
    else:
        rows[0] += " bad-token"
    path.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError):
        extract_features(tmp_path)


def test_strict_training_rejects_toy_data(tmp_path):
    generate(tmp_path, n=48)
    with pytest.raises(ValueError, match="2,205"):
        extract_features(tmp_path, strict_uci=True)


def test_blocked_split_coverage_and_purge():
    seen = []
    for train, test in blocked_splits(2205):
        assert not set(train) & set(test)
        assert np.all((train < test[0] - 10) | (train > test[-1] + 10))
        assert np.all(np.diff(test) == 1)
        seen.extend(test)
    assert sorted(seen) == list(range(2205))


def test_invalid_samples():
    with pytest.raises(ValueError):
        summarize_cycle(np.array([]), "PS1")
