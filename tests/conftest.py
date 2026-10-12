import joblib
import pytest
from scripts.generate_test_fixture import generate
from factorymind.data import extract_features
from factorymind.train import train

@pytest.fixture(scope="session")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("synthetic")
    generate(root / "raw", n=48)
    features = extract_features(root / "raw")
    train(features, root / "artifacts")
    return joblib.load(root / "artifacts/valve_model.joblib"), features
