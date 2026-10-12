"""Descriptive analysis of existing blocked predictions; never retrains on labels."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main():
    pred = pd.read_csv(ROOT / "artifacts/blocked_predictions.csv")
    metrics = json.loads((ROOT / "artifacts/metrics.json").read_text())
    profile = np.loadtxt(ROOT / "data/raw/profile.txt", dtype=int)
    assert pred.cycle_id.tolist() == list(range(1, len(profile) + 1))
    assert pred.actual.tolist() == profile[:, 1].tolist()
    pred["error"] = pred.actual != pred.predicted
    for fold in metrics["blocked"]["folds"]:
        mask = pred.cycle_id.between(fold["first_test_cycle"], fold["last_test_cycle"])
        pred.loc[mask, "fold"] = fold["fold"]
    for i, name in enumerate(["cooler", "valve", "pump_leakage", "accumulator", "stability"]):
        pred[name] = profile[:, i]
    pairs = pred[pred.error].groupby(["actual", "predicted"]).size().sort_values(ascending=False)
    report = {"total_cycles": len(pred), "errors": int(pred.error.sum()),
              "missed_nonoptimal": int(((pred.actual != 100) & (pred.predicted == 100)).sum()),
              "false_nonoptimal": int(((pred.actual == 100) & (pred.predicted != 100)).sum()),
              "pairs": [{"actual": int(a), "predicted": int(p), "count": int(n)} for (a,p),n in pairs.items()],
              "folds": [], "annotation_strata": {}}
    for fold, group in pred.groupby("fold"):
        report["folds"].append({"fold": int(fold), "cycles": len(group), "errors": int(group.error.sum()),
                               "error_rate": float(group.error.mean()),
                               "class_counts": {str(k): int(v) for k,v in group.actual.value_counts().sort_index().items()}})
    for name in ["cooler", "pump_leakage", "accumulator", "stability"]:
        report["annotation_strata"][name] = [{"code": int(code), "cycles": len(g), "errors": int(g.error.sum()),
                                              "error_rate": float(g.error.mean())} for code,g in pred.groupby(name)]
    report["limits"] = "Annotations are used only for post-hoc audit, never predictors. Strata overlap and differ in class mix and cycle block; associations are not causal explanations. No new model selection performed."
    (ROOT / "artifacts/blocked_error_analysis.json").write_text(json.dumps(report, indent=2))
    lines = ["# Blocked-validation error analysis", "", "Computed by `python scripts/analyze_errors.py` from retained out-of-fold predictions and original profile rows. This analysis does not refit the model or use annotations as predictors.", "",
             f"There are **{report['errors']} errors out of {len(pred)} cycles**. Of these, **{report['missed_nonoptimal']} ({report['missed_nonoptimal']/report['errors']:.2%}) are non-optimal labels predicted optimal**. Another {report['false_nonoptimal']} are optimal labels predicted non-optimal; 77 confuse one non-optimal class with another.", "",
             "## Largest confusion directions", "", "| Dataset label → prediction | Cycles | Share of all errors |", "|---|---:|---:|"]
    for pair in report['pairs']:
        lines.append(f"| {pair['actual']} → {pair['predicted']} | {pair['count']} | {pair['count']/report['errors']:.2%} |")
    lines += ["", "Codes: 100 optimal, 90 small lag, 80 severe lag, 73 near failure. Small-lag recall is 176/360 = 48.89%; severe-lag recall is 253/360 = 70.28%. The 154 small-lag → optimal errors alone account for 40.74% of all errors. A good aggregate accuracy would conceal this sensitivity problem.", "", "## Block concentration", "", "| Fold | Errors / 441 | Error rate | Class counts (73 / 80 / 90 / 100) |", "|---|---:|---:|---|"]
    for f in report['folds']:
        counts=' / '.join(str(f['class_counts'].get(str(c),0)) for c in [73,80,90,100])
        lines.append(f"| {f['fold']} | {f['errors']} | {f['error_rate']:.2%} | {counts} |")
    lines += ["", "Fold 5 (cycles 1765–2205) contributes 137/378 = 36.24% of all errors: 131 of its errors predict optimal for non-optimal labels. In that fold all 148 optimal examples are correct, while 58 small-lag, 40 severe-lag and 33 near-failure examples are predicted optimal. Fold 2 contains all 41 optimal → non-optimal errors. Failure behavior therefore varies strongly by cycle block.", "", "## Post-hoc annotation associations", "", "| Annotation | Code | Cycles | Errors | Error rate |", "|---|---|---:|---:|---:|"]
    for name,strata in report['annotation_strata'].items():
        for s in strata:
            lines.append(f"| {name} | {s['code']} | {s['cycles']} | {s['errors']} | {s['error_rate']:.2%} |")
    lines += ["", "These overlapping strata are descriptive, not causal attribution. Class balance and cycle ordering differ across strata. Stability and the other component labels are excluded from model features. Do not claim that a particular cooler setting, stability flag or accumulator value caused a valve error.", "", "## Plausible explanations and next validation steps", "", "The random/blocked performance gap is consistent with correlated neighboring cycles and changing condition distributions. The matrix establishes a bias toward optimal predictions on several unseen blocks; it cannot identify the physical mechanism. Whole-cycle summaries may hide short switching events, and sensor readings can reflect multiple component settings. Those are hypotheses, not tested causes.", "", "Next experiments: use an untouched block for model selection assessment; compare event-window features and class-weight choices using inner blocked validation; report per-class recall and false-alarm rates; consider an abstention/review policy evaluated independently. Do not tune on these reported folds and then present their reused scores as unbiased generalization. Ultimately obtain independent-rig or plant validation. No feature-causality, calibrated failure probability or maintenance effectiveness is established."]
    (ROOT / "docs/BLOCKED_ERROR_ANALYSIS.md").write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:report[k] for k in ['total_cycles','errors','missed_nonoptimal','false_nonoptimal']},indent=2))

if __name__ == "__main__": main()
