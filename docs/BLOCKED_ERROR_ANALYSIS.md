# Blocked-validation error analysis

Computed by `python -m scripts.analyze_errors` from retained out-of-fold predictions and original profile rows. This analysis does not refit the model or use annotations as predictors.

There are **378 errors out of 2205 cycles**. Of these, **260 (68.78%) are non-optimal labels predicted optimal**. Another 41 are optimal labels predicted non-optimal; 77 confuse one non-optimal class with another.

## Largest confusion directions

| Dataset label → prediction | Cycles | Share of all errors |
|---|---:|---:|
| 90 → 100 | 154 | 40.74% |
| 80 → 100 | 73 | 19.31% |
| 73 → 100 | 33 | 8.73% |
| 100 → 90 | 32 | 8.47% |
| 90 → 80 | 27 | 7.14% |
| 80 → 90 | 24 | 6.35% |
| 73 → 80 | 12 | 3.17% |
| 80 → 73 | 10 | 2.65% |
| 100 → 80 | 7 | 1.85% |
| 90 → 73 | 3 | 0.79% |
| 100 → 73 | 2 | 0.53% |
| 73 → 90 | 1 | 0.26% |

Codes: 100 optimal, 90 small lag, 80 severe lag, 73 near failure. Small-lag recall is 176/360 = 48.89%; severe-lag recall is 253/360 = 70.28%. The 154 small-lag → optimal errors alone account for 40.74% of all errors. A good aggregate accuracy would conceal this sensitivity problem.

## Block concentration

| Fold | Errors / 441 | Error rate | Class counts (73 / 80 / 90 / 100) |
|---|---:|---:|---|
| 1 | 63 | 14.29% | 60 / 55 / 50 / 276 |
| 2 | 82 | 18.59% | 60 / 65 / 70 / 246 |
| 3 | 85 | 19.27% | 90 / 90 / 90 / 171 |
| 4 | 11 | 2.49% | 57 / 50 / 50 / 284 |
| 5 | 137 | 31.07% | 93 / 100 / 100 / 148 |

Fold 5 (cycles 1765–2205) contributes 137/378 = 36.24% of all errors: 131 of its errors predict optimal for non-optimal labels. In that fold all 148 optimal examples are correct, while 58 small-lag, 40 severe-lag and 33 near-failure examples are predicted optimal. Fold 2 contains all 41 optimal → non-optimal errors. Failure behavior therefore varies strongly by cycle block.

## Post-hoc annotation associations

| Annotation | Code | Cycles | Errors | Error rate |
|---|---|---:|---:|---:|
| cooler | 3 | 732 | 144 | 19.67% |
| cooler | 20 | 732 | 89 | 12.16% |
| cooler | 100 | 741 | 145 | 19.57% |
| pump_leakage | 0 | 1221 | 218 | 17.85% |
| pump_leakage | 1 | 492 | 76 | 15.45% |
| pump_leakage | 2 | 492 | 84 | 17.07% |
| accumulator | 90 | 808 | 98 | 12.13% |
| accumulator | 100 | 399 | 71 | 17.79% |
| accumulator | 115 | 399 | 70 | 17.54% |
| accumulator | 130 | 599 | 139 | 23.21% |
| stability | 0 | 1449 | 358 | 24.71% |
| stability | 1 | 756 | 20 | 2.65% |

These overlapping strata are descriptive, not causal attribution. Class balance and cycle ordering differ across strata. Stability and the other component labels are excluded from model features. Do not claim that a particular cooler setting, stability flag or accumulator value caused a valve error.

## Plausible explanations and next validation steps

The random/blocked performance gap is consistent with correlated neighboring cycles and changing condition distributions. The matrix establishes a bias toward optimal predictions on several unseen blocks; it cannot identify the physical mechanism. Whole-cycle summaries may hide short switching events, and sensor readings can reflect multiple component settings. Those are hypotheses, not tested causes.

Next experiments: use an untouched block for model selection assessment; compare event-window features and class-weight choices using inner blocked validation; report per-class recall and false-alarm rates; consider an abstention/review policy evaluated independently. Do not tune on these reported folds and then present their reused scores as unbiased generalization. Ultimately obtain independent-rig or plant validation. No feature-causality, calibrated failure probability or maintenance effectiveness is established.
