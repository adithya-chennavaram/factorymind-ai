# FactoryMind — Green-Light Challenge, experiment 1

## Objective

**Hypothesis, not a proven improvement:** a label-blind, fixed-weight ranking that combines `1 - P(Optimal)` with simulated sensor-gain sensitivity can flag more incorrectly healthy predictions than the low-confidence-only baseline at **the same review budget**.

Only predicted-Optimal (green) cycles are candidates for review; existing predicted non-Optimal cases are outside this experiment. The review budget is the percent of green predictions **per held-out blocked fold**, not a percent of all 2,205 cycles.

### Existing baseline (already measured by the owner)

- Green: 1,344; false green: 260 (19.35% among green).
- 10% review: 132 inspected, 91 false green caught (35.0%).
- 20% review: 266 inspected, 136 false green caught (52.31%).
- 30% review: 400 inspected, 167 false green caught (64.23%).
- Random-review expected catch: 25.47 / 51.45 / 77.44 respectively.

Do not attribute any of these improvements to the new method: **this is the confidence-only reference**.

## Experiment

1. Read `data/processed/features.csv` and `artifacts/blocked_predictions.csv` created by `python -m factorymind.train`.
2. Refit the same Random Forest on each purged, blocked training fold and verify the stored out-of-fold predictions and `P(Optimal)` values match.
3. For each **predicted-Optimal held-out cycle**, challenge its sensor features with 50 **hypothetical multiplicative sensor-gain scenarios**: 5 sensors (`PS1`, `PS2`, `FS1`, `TS1`, `VS1`) × 2 directions × 5 magnitudes (0.5, 1, 2, 3, 5%). One sensor at a time is scaled; all its statistical features are multiplied by the same gain. This is consistent with a constant gain change in raw samples.
4. Compute the largest decline in `P(Optimal)`, plus the fraction of perturbed cases whose predicted class changes from Optimal.
5. Compare the following **fixed, label-blind** review rankings:
   - **confidence_only**: lowest original `P(Optimal)` first.
   - **stress_only**: largest drop in `P(Optimal)` first.
   - **factorymind_combined**: highest `0.5 * (1 - P(Optimal)) + 0.5 * max_probability_drop` first.
6. At 10%, 20%, and 30% of each fold's green cycles, count the held-out false-green errors selected for human review. Actual labels are accessed only after the selection.

The 0.5 weight is a **prespecified exploratory hypothesis**. Do not tune it using these held-out fold results, then advertise the tuned result as independent performance. If the combined method does not beat the confidence baseline, report that honestly and keep the Crash Test Lab as sensitivity diagnostics.

## Commands

```bash
cd ~/Downloads/factorymind-ai
source .venv/bin/activate
python -m pytest -q tests/test_stress.py
python scripts/evaluate_green_challenge.py
```

The complete evaluation may take time because it refits five random forests and predicts across 50 scenarios per green test cycle. It does not call AMD, does not use API key/quota, and does not change the deployed model artifact.

Outputs:

- `artifacts/green_challenge_results.json` — review-budget comparisons and limitations.
- `artifacts/green_challenge_scored.csv` — per-green-cycle confidence, stress sensitivity, and review priority for UI integration. Contains actual labels for **offline evaluation only**. **Do not send ground-truth labels to an AMD LLM investigation or reveal them before the operator decision.**

## Interpretation and limitations

- Stability under stress testing does not imply correctness. An incorrect prediction can remain stable under every scenario.
- Chosen gain magnitudes are illustrative and **not validated** against these sensors' specifications, uncertainties, or factory calibration drift.
- These folds use the **same experimental hydraulic test rig**, with nearby cycles purged. They are neither prospective factory validation nor true future failure forecasting.
- Some features capture valve condition, but the experiment does not identify physical root causes.
- Catch rate = false-green predictions reviewed / all false-green predictions among green predictions.
- Inspection precision = false-green predictions reviewed / all green predictions reviewed.
- Random expectation is conditioned on the individual held-out folds and their class mix, and must not be confused with a measured random trial.
- Do not claim downtime, financial savings, calibrated fault probabilities, or physical safety improvements from these results alone.
