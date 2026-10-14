# Green-Light Audit: precise definition and next measurement

The original upload `blocked_predictions.csv` contains only cycle IDs, ground truth and class predictions. Its fixed blocked-fold results show **2,205** experimental cycles, **1,344** predicted Optimal and **260** non-optimal cases predicted Optimal. Therefore 19.35% of the predicted-green population was wrong in this blocked evaluation. This is **not** a real-world failure rate.

A 20% review budget **among predicted-green cycles** means approximately 269 inspections overall (per-fold floors may differ). Random review would be expected to find roughly 52 of the 260 incorrect-green predictions if chosen uniformly across all greens; these are *expectations*, not verified catches.

The provided old CSV contains no out-of-fold confidence scores, so it cannot support a performance claim for confidence review or a combined reliability gate. We amended `factorymind/train.py` to save the out-of-fold `p_optimal_uncalibrated` and fold IDs in `artifacts/blocked_predictions.csv` when training is run again. Run:

```bash
python -m factorymind.train
python scripts/evaluate_green_review.py
```

The evaluator ranks only predictions classified as Optimal and uses the **lowest out-of-fold P(Optimal)** as a simple baseline at 10%, 20%, and 30% review budgets per fold. Labels are consulted only to score the resulting review selections, not to create their rankings.

**Important:** A new combined reliability gate must be evaluated fairly against this baseline, not tuned on its test outcomes. Stress-test instability alone does not establish that a prediction is wrong. All sensor-noise assumptions must be clearly identified as simulations, not physical failures.
