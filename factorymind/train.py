"""Reproducible random holdout and purged contiguous-block evaluation."""
from pathlib import Path
import json
import hashlib
import platform
from time import perf_counter
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from .data import TARGET_LABELS, extract_features, feature_columns

ROOT = Path(__file__).resolve().parent.parent
ORDER = sorted(TARGET_LABELS)


def make_model():
    return Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("classifier", RandomForestClassifier(n_estimators=180, min_samples_leaf=2,
                      random_state=42, class_weight="balanced_subsample", n_jobs=-1))])


def evaluate(actual, predicted):
    return {"test_samples": len(actual), "accuracy": float(accuracy_score(actual, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
            "macro_f1": float(f1_score(actual, predicted, labels=ORDER, average="macro", zero_division=0)),
            "per_class": classification_report(actual, predicted, labels=ORDER,
                target_names=[TARGET_LABELS[v] for v in ORDER], output_dict=True, zero_division=0),
            "class_order": ORDER, "confusion_matrix": confusion_matrix(actual, predicted, labels=ORDER).tolist()}


def blocked_splits(n, gap=10):
    """Five contiguous held-out blocks; remove neighboring cycles from training."""
    for train_idx, test_idx in KFold(n_splits=5, shuffle=False).split(np.arange(n)):
        train_idx = train_idx[(train_idx < test_idx[0] - gap) | (train_idx > test_idx[-1] + gap)]
        yield train_idx, test_idx


def train(df: pd.DataFrame, artifacts_dir: str | Path, *, source="unspecified / test data", blocked=False) -> dict:
    artifacts_dir = Path(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    cols = feature_columns(df)
    if len(cols) != 30 or df.cycle_id.duplicated().any():
        raise ValueError("Expected 30 sensor features and unique cycle IDs")
    df = df.sort_values("cycle_id").reset_index(drop=True)
    X, y = df[cols], df.valve_condition.astype(int)
    if not set(y).issubset(ORDER) or y.value_counts().min() < 2:
        raise ValueError("Expected supported classes with at least two examples each")
    train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=.25, random_state=42, stratify=y)
    model = make_model()
    start = perf_counter()
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    predicted = model.predict(X.iloc[test_idx])
    metrics = evaluate(y.iloc[test_idx], predicted)
    dummy = DummyClassifier(strategy="most_frequent").fit(X.iloc[train_idx], y.iloc[train_idx])
    metrics.update({"dataset": source, "total_cycles": len(df), "train_samples": len(train_idx),
        "class_counts": {str(k): int(v) for k, v in y.value_counts().sort_index().items()},
        "evaluation_note": "Random stratified 75/25 split, seed 42. Correlated rig cycles can inflate performance. No external validation.",
        "feature_columns": cols, "train_cycle_ids": df.iloc[train_idx].cycle_id.astype(int).tolist(),
        "test_cycle_ids": df.iloc[test_idx].cycle_id.astype(int).tolist(),
        "majority_baseline": evaluate(y.iloc[test_idx], dummy.predict(X.iloc[test_idx])),
        "fit_and_holdout_seconds": round(perf_counter() - start, 3),
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "features_sha256": hashlib.sha256(df[cols + ["cycle_id", "valve_condition"]].to_csv(index=False).encode()).hexdigest()})
    all_predictions = model.predict(X)
    cases = pd.DataFrame({"cycle_id": df.cycle_id, "actual": y, "predicted": all_predictions,
                          "split": np.where(np.isin(np.arange(len(df)), test_idx), "held_out", "training")})
    cases.to_csv(artifacts_dir / "predictions.csv", index=False)
    if blocked:
        out_of_fold = np.zeros(len(df), dtype=int)
        oof_p_optimal = np.zeros(len(df), dtype=float)
        oof_confidence = np.zeros(len(df), dtype=float)
        oof_fold = np.zeros(len(df), dtype=int)
        folds = []
        for fold, (tr, te) in enumerate(blocked_splits(len(df)), 1):
            candidate = make_model().fit(X.iloc[tr], y.iloc[tr])
            probabilities = candidate.predict_proba(X.iloc[te])
            optimum_column = list(candidate.classes_).index(100)
            oof_p_optimal[te] = probabilities[:, optimum_column]
            oof_confidence[te] = probabilities.max(axis=1)
            oof_fold[te] = fold
            out_of_fold[te] = candidate.predict(X.iloc[te])
            folds.append({"fold": fold, "first_test_cycle": int(df.iloc[te[0]].cycle_id),
                          "last_test_cycle": int(df.iloc[te[-1]].cycle_id), "train_samples": len(tr),
                          "train_classes": sorted(y.iloc[tr].unique().tolist()), **evaluate(y.iloc[te], out_of_fold[te])})
        metrics["blocked"] = {**evaluate(y, out_of_fold), "folds": folds, "purge_cycles": 10,
            "note": "Five contiguous folds with 10-cycle boundary purge; pooled out-of-fold predictions. Uses both earlier and later training blocks, not forecasting or independent-rig validation."}
        pd.DataFrame({"cycle_id": df.cycle_id, "actual": y, "predicted": out_of_fold,
                      "fold": oof_fold, "p_optimal_uncalibrated": oof_p_optimal,
                      "max_confidence_uncalibrated": oof_confidence}).to_csv(artifacts_dir / "blocked_predictions.csv", index=False)
    joblib.dump(model, artifacts_dir / "valve_model.joblib")
    metrics["model_sha256"] = hashlib.sha256((artifacts_dir / "valve_model.joblib").read_bytes()).hexdigest()
    (artifacts_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main():
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--raw-dir", default=str(ROOT / "data/raw"))
    parser.add_argument("--features", default=str(ROOT / "data/processed/features.csv"))
    parser.add_argument("--artifacts", default=str(ROOT / "artifacts"))
    args = parser.parse_args()
    print("Extracting and validating real UCI data...", flush=True)
    df = extract_features(args.raw_dir, strict_uci=True)
    print(f"Validated {len(df)} cycles and {len(feature_columns(df))} features", flush=True)
    metrics = train(df, args.artifacts, source="UCI 447 hydraulic experimental test rig; real downloaded sensor data", blocked=True)
    df["evaluation_split"] = np.where(df.cycle_id.isin(metrics["test_cycle_ids"]), "held_out", "training")
    path = Path(args.features)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(json.dumps({k: metrics[k] for k in ("accuracy", "balanced_accuracy", "macro_f1", "test_samples")}, indent=2))
    print("Blocked pooled macro F1:", metrics["blocked"]["macro_f1"])


if __name__ == "__main__":
    main()
