# Real UCI evaluation — October 12, 2026

Generated from artifacts/metrics.json after executing python -m factorymind.train on the official download.

Python 3.13.5; scikit-learn 1.9.1. Seed 42; 180 trees; minimum leaf size 2; balanced_subsample class weighting. The served model was fitted on 1,653 training cycles and remains unchanged for the 552-cycle holdout demonstration.

Class counts: 73 (near failure): 360; 80 (severe lag): 360; 90 (small lag): 360; 100 (optimal switching): 1,125.

All 2,205 rows were checked against profile.txt. Predictors are six statistics per sensor (mean, population standard deviation, minimum, maximum, 95th percentile and second-half mean minus first-half mean), across PS1, PS2, FS1, TS1 and VS1. Profile annotations, stability and cycle ID are excluded.

## Random stratified holdout

Accuracy: 0.98007246; balanced accuracy: 0.97890071; macro F1: 0.97451482; evaluated cycles: 552.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Near failure | 0.989011 | 1.000000 | 0.994475 | 90 |
| Severe lag | 0.977273 | 0.955556 | 0.966292 | 90 |
| Small lag | 0.916667 | 0.977778 | 0.946237 | 90 |
| Optimal switching | 1.000000 | 0.982270 | 0.991055 | 282 |

Confusion matrix: rows = dataset labels, columns = predictions. Class order 73, 80, 90, 100.

| Actual / predicted | 73 | 80 | 90 | 100 |
|---|---:|---:|---:|---:|
| 73 | 90 | 0 | 0 | 0 |
| 80 | 0 | 86 | 4 | 0 |
| 90 | 0 | 2 | 88 | 0 |
| 100 | 1 | 0 | 4 | 277 |

## Five contiguous folds with 10-cycle purge

Accuracy: 0.82857143; balanced accuracy: 0.75686111; macro F1: 0.78774403; evaluated cycles: 2205.

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Near failure | 0.954407 | 0.872222 | 0.911466 | 360 |
| Severe lag | 0.846154 | 0.702778 | 0.767830 | 360 |
| Small lag | 0.755365 | 0.488889 | 0.593592 | 360 |
| Optimal switching | 0.806548 | 0.963556 | 0.878088 | 1125 |

Confusion matrix: rows = dataset labels, columns = predictions. Class order 73, 80, 90, 100.

| Actual / predicted | 73 | 80 | 90 | 100 |
|---|---:|---:|---:|---:|
| 73 | 314 | 12 | 1 | 33 |
| 80 | 10 | 253 | 24 | 73 |
| 90 | 3 | 27 | 176 | 154 |
| 100 | 2 | 7 | 32 | 1084 |

## Block details

| Fold | Held-out cycle range | Training cycles after purge | Accuracy | Macro F1 |
|---|---|---:|---:|---:|
| 1 | 1–441 | 1754 | 0.857143 | 0.728373 |
| 2 | 442–882 | 1744 | 0.814059 | 0.775930 |
| 3 | 883–1323 | 1744 | 0.807256 | 0.789579 |
| 4 | 1324–1764 | 1744 | 0.975057 | 0.962009 |
| 5 | 1765–2205 | 1754 | 0.689342 | 0.683022 |

Block boundaries follow file row order. Each cycle appears in exactly one held-out block; ten neighboring cycles are removed from training at each boundary. Training can use both earlier and later blocks. This is a stress test for local cycle correlation, not forecasting, leave-one-rig-out validation or proof of field performance. Other component conditions can be confounded with valve condition even though their labels are excluded.

## Baseline and demo errors

Most-frequent-class baseline on the same random holdout: accuracy 0.510870, macro F1 0.169065. No hyperparameter search or calibration was performed. The blocked results were not used to choose a revised model.

Random holdout: 541 correct, 11 incorrect. Cycle 4: actual 100, predicted 100. Cycle 213: actual 73, predicted 73. Cycle 248: actual 100, predicted 73. These are all held-out examples. Blocked evaluation produces 378 errors; small-lag recall is 48.89%.

## Provenance

Model SHA-256: `55d84ff82f43b97fb7151d7f8800786ae99f15cd816d95b660b2350b71ee04fe`.

Canonical sensor/target table SHA-256: `b4f13d23507a3d62f2a4412fc20e18da3ae4c4a49d4fb6c07c105d8bf3fdfc74`.

Local archive and extracted-file SHA-256 hashes are saved in data/raw/manifest.json. Prediction tables are artifacts/predictions.csv and artifacts/blocked_predictions.csv. Full split membership, all fold reports and environment versions are retained in artifacts/metrics.json. Hashes are reproducibility identifiers, not independent authenticity signatures.

Dataset: Helwig, Pignanelli & Schütze (2015), UCI Condition monitoring of hydraulic systems, [DOI 10.24432/C5CW21](https://doi.org/10.24432/C5CW21), CC BY 4.0. No external plant data or fabricated fixture metrics are used here.
