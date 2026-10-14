"""Leakage-aware, blocked out-of-fold *hypothesis test* of FactoryMind scoring.

Run after `python -m factorymind.train`, which writes the feature CSV and the
blocked predictions with model probabilities. Refit the same five blocked models
and challenge predicted-green held-out rows only. No validation labels are used
for scoring, stress-testing, selecting cases, or modifying the model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from factorymind.data import feature_columns
from factorymind.stress import DEFAULT_GAIN_PCTS, challenge_priority, stress_test_gain
from factorymind.train import blocked_splits, make_model

BUDGETS = (10, 20, 30)
REQUIRED = {"cycle_id", "actual", "predicted", "fold", "p_optimal_uncalibrated"}


def evaluate_rankings(scored: pd.DataFrame, *, budgets=BUDGETS) -> dict:
    """Compute outcomes only AFTER label-blind scoring and case selection."""
    needed = REQUIRED | {"max_p_optimal_drop", "challenge_priority"}
    if not needed.issubset(scored.columns):
        raise ValueError("Missing ranking columns: " + ", ".join(sorted(needed - set(scored.columns))))
    if scored.cycle_id.duplicated().any() or (scored.predicted != 100).any():
        raise ValueError("Expected unique predicted-Optimal cycles only")
    if scored.empty:
        raise ValueError("No predicted-green cases")
    total_false = int((scored.actual != 100).sum())
    by_method = {}
    for method, column in (("confidence_only", "p_optimal_uncalibrated"),
                           ("stress_only", "max_p_optimal_drop"),
                           ("factorymind_combined", "challenge_priority")):
        sort_ascending = method == "confidence_only"
        entries = []
        for budget in budgets:
            reviewed = 0
            caught = 0
            random_expected = 0.0
            folds = []
            for fold, group in scored.groupby("fold", sort=True):
                k = int(np.floor(len(group) * budget / 100))
                # Stable cycle-ID tie breaker. Selection never uses ground truth.
                chosen = group.sort_values([column, "cycle_id"], ascending=[sort_ascending, True], kind="stable").head(k)
                fold_caught = int((chosen.actual != 100).sum())
                fold_false = int((group.actual != 100).sum())
                reviewed += k
                caught += fold_caught
                random_expected += k * fold_false / len(group)
                folds.append({"fold": int(fold), "green": len(group), "reviewed": k,
                              "false_green": fold_false, "caught": fold_caught})
            entries.append({"budget_pct_of_predicted_green_per_fold": int(budget),
                            "reviewed": reviewed, "false_green_caught": caught,
                            "false_green_catch_rate": round(caught / total_false, 4) if total_false else None,
                            "inspection_precision": round(caught / reviewed, 4) if reviewed else None,
                            "false_green_remaining": total_false - caught,
                            "random_expected_caught": round(random_expected, 2),
                            "folds": folds})
        by_method[method] = entries
    return {"predicted_green": len(scored), "false_green": total_false, "methods": by_method}


def produce_scores(features: pd.DataFrame, blocked: pd.DataFrame) -> pd.DataFrame:
    """Reproduce OOF models; stress only held-out green rows, with no label leakage."""
    if not REQUIRED.issubset(blocked.columns):
        raise ValueError("Missing OOF probabilities; rerun python -m factorymind.train")
    cols = feature_columns(features)
    if len(cols) != 30:
        raise ValueError("Expected 30 sensor features")
    if features.cycle_id.duplicated().any() or blocked.cycle_id.duplicated().any():
        raise ValueError("Duplicate cycle IDs")
    features = features.sort_values("cycle_id").reset_index(drop=True)
    blocked = blocked.sort_values("cycle_id").reset_index(drop=True)
    if not np.array_equal(features.cycle_id.to_numpy(), blocked.cycle_id.to_numpy()):
        raise ValueError("Cycle IDs don't match between features and OOF predictions")
    if (features.valve_condition.to_numpy() != blocked.actual.to_numpy()).any():
        raise ValueError("Ground truth inconsistent between input files")
    X = features[cols]
    y = features.valve_condition.astype(int)
    reports = []
    for fold, (train_idx, test_idx) in enumerate(blocked_splits(len(features)), start=1):
        if not (blocked.iloc[test_idx].fold == fold).all():
            raise ValueError("Blocked split does not match stored fold")
        print(f"Fold {fold}/5: train={len(train_idx)}, test={len(test_idx)}", flush=True)
        model = make_model().fit(X.iloc[train_idx], y.iloc[train_idx])
        predicted = model.predict(X.iloc[test_idx])
        if not np.array_equal(predicted, blocked.iloc[test_idx].predicted.to_numpy()):
            raise ValueError("Stored out-of-fold predictions differ from refitted model")
        current_p = model.predict_proba(X.iloc[test_idx])[:, list(model.classes_).index(100)]
        recorded_p = blocked.iloc[test_idx].p_optimal_uncalibrated.to_numpy(dtype=float)
        if not np.allclose(current_p, recorded_p, rtol=1e-4, atol=1e-4):
            raise ValueError("Stored out-of-fold probabilities differ; rerun training")
        green_positions = np.flatnonzero(predicted == 100)
        if not len(green_positions):
            continue
        chosen = test_idx[green_positions]
        green_features = X.iloc[chosen]
        stress = stress_test_gain(model, green_features)
        selection = blocked.iloc[chosen].copy().reset_index(drop=True)
        result = stress.reset_index(drop=True)
        for c in result.columns:
            selection[c] = result[c].to_numpy()
        selection["challenge_priority"] = challenge_priority(selection.p_optimal_uncalibrated, selection.max_p_optimal_drop)
        reports.append(selection)
    if not reports:
        raise ValueError("No green predictions in any held-out fold")
    result = pd.concat(reports, ignore_index=True).sort_values("cycle_id").reset_index(drop=True)
    if len(result) != (blocked.predicted == 100).sum():
        raise ValueError("Missing predicted-green cycles")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=Path("data/processed/features.csv"))
    parser.add_argument("--predictions", type=Path, default=Path("artifacts/blocked_predictions.csv"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/green_challenge_results.json"))
    args = parser.parse_args()
    features = pd.read_csv(args.features)
    blocked = pd.read_csv(args.predictions)
    scores = produce_scores(features, blocked)
    report = evaluate_rankings(scores)
    report.update({
        "scope": "Purged blocked OOF evaluation of experimental hydraulic test-rig valve conditions; NOT prospective validation",
        "challenge": "50 simulated multiplicative sensor-gain scenarios: 5 sensors × 2 signs × 5 magnitudes",
        "gain_percentages": list(DEFAULT_GAIN_PCTS),
        "ranking": "Prespecified 0.5*(1 - P_optimal_uncalibrated) + 0.5*max(P_optimal - challenged_P_optimal)",
        "labels": "Labels used only for blocked-fold model training and post-selection evaluation; never for scoring held-out cases",
        "limitations": ["Gain perturbations are illustrative, not sensor-spec-validated", "Blocked folds still share the same experimental rig", "No validated maintenance savings, hardware fault simulation, or physical root-cause diagnosis", "Do not tune weights after inspecting these results and claim independent test superiority"],
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    score_path = args.output.with_name("green_challenge_scored.csv")
    scores.to_csv(score_path, index=False)
    print(json.dumps({"predicted_green": report["predicted_green"], "false_green": report["false_green"],
                      "budget_summary": {method: [{k: v for k, v in row.items() if k != "folds"} for row in entries]
                                         for method, entries in report["methods"].items()},
                      "scores_file": str(score_path)}, indent=2))


if __name__ == "__main__":
    main()
