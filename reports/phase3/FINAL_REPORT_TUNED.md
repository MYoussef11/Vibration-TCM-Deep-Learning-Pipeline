# Phase 3 – GroupKFold & Tuning Results

## Data & Validation
- GroupKFold is used for every evaluation, grouping by source file.  
- Default dataset: `(1641, 20, 6)` windows; spectrograms `(1641, 9, 1, 1)`.  
- Max-spec dataset: `(304, 100, 6)` windows; spectrograms `(304, 33, 2, 1)` (64-point STFT).

## Classical Models (GroupKFold, tuned)
Best model from `scripts/tune_classical_models.py`:

| Model | Mean CV Accuracy | Macro F1 | Key Params |
| --- | --- | --- | --- |
| Gradient Boosting | **0.504** | 0.462 | n_estimators=250, learning_rate=0.06, max_depth=2, min_samples_split=4, min_samples_leaf=2, subsample=0.8 |

Per-class F1: Good 0.531, Moderate 0.263, Bad 0.594. Artifacts: `reports/phase3/classical_tuning/`.

## Deep Learning (GroupKFold, default dataset)
Aggregate results from `reports/phase3/dl/<model>/metrics.json`:

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| 2D CNN | 0.491 | 0.382 |
| 1D CNN | 0.425 | 0.383 |
| LSTM | 0.340 | 0.316 |
| Hybrid | 0.311 | 0.270 |

## Max-Spec Trial (GroupKFold, 100-sample windows)
Aggregate results from `reports/phase3/dl_maxspec/<model>/metrics.json`:

| Model | Accuracy | Macro F1 |
| --- | --- | --- |
| 1D CNN | 0.503 | 0.478 |
| 2D CNN | 0.495 | 0.334 |
| LSTM | 0.411 | 0.354 |
| Hybrid | 0.377 | 0.275 |

## Comparison to Original Report (Random Split)

| Phase | Classical Best Acc. | DL Best Acc. | Validation |
| --- | --- | --- | --- |
| Original (`FINAL_REPORT.md`) | Gradient Boosting (PyCaret) – 0.918 | 1D CNN – 0.862 | Random split |
| Tuned (GroupKFold) | Gradient Boosting – 0.504 | 1D CNN (max-spec) – 0.503 | GroupKFold |

## Files
- Classical tuning outputs: `reports/phase3/classical_tuning/`  
- DL GroupKFold (default): `reports/phase3/dl/`  
- DL GroupKFold (max-spec): `reports/phase3/dl_maxspec/`  
- Legacy random-split report: `reports/phase3/FINAL_REPORT.md`
