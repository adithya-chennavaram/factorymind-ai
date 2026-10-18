# Green-Light Audit: confidence-only baseline

The original upload `blocked_predictions.csv` contains only cycle IDs, ground truth and class predictions. Its fixed blocked-fold results show **2,205** experimental cycles, **1,344** predicted Optimal and **260** non-optimal cases predicted Optimal. Therefore 19.35% of the predicted-green population was wrong in this blocked evaluation. This is **not** a real-world failure rate.

A 20% review budget **among predicted-green cycles** produces 266 reviews after deterministic per-fold floors and catches 136 false-green cases (52.31%). Random-review values remain expectations, not verified catches.

The current training artifacts contain out-of-fold `p_optimal_uncalibrated` and fold IDs. The completed baseline can be regenerated with:

```bash
python -m factorymind.train
python -m scripts.evaluate_green_review
```

The evaluator ranks only predictions classified as Optimal and uses the **lowest out-of-fold P(Optimal)** as a simple baseline at 10%, 20%, and 30% review budgets per fold. Labels are consulted only to score the resulting review selections, not to create their rankings.

**Result:** confidence-only is the recommended evaluated baseline. At the same 20% budget, stress-only catches 5 false-green cases and the original combined policy catches 66; neither improves the baseline. Stress-test instability alone does not establish that a prediction is wrong. All sensor-gain assumptions are simulations, not physical failures.
