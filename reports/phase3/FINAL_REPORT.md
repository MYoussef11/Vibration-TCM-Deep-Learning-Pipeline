# Phase 3 – Supervised Modeling Results

## Datasets & Inputs

- Filtered band-pass (4th order, 1–4 Hz) at 10 Hz sampling; 20-sample windows with 50 % overlap.
- Time-series tensor: `(1641, 20, 6)` windows → train/val/test splits (1184/210/247).
- Spectrogram tensor: `(1641, 9, 1, 1)` (9 FFT bins from 16-point STFT).

## Baseline: PyCaret Feature Models

| Metric | Value |
| --- | --- |
| Best model | Gradient Boosting Classifier |
| Accuracy | 0.918 |
| Macro F1 | 0.914 |

Per-class F1: Good 0.926, Moderate 0.892, Bad 0.925. Artifacts in `reports/phase3/pycaret/`.

## Deep Learning Models

| Model | Test Accuracy | Macro F1 (from report) |
| --- | --- | --- |
| 1D CNN | **0.862** | 0.848 |
| LSTM | 0.745 | 0.740 |
| Hybrid (1D + 2D) | 0.741 | 0.732 |
| 2D CNN (spectrogram only) | 0.494 | 0.399 |

### Highlights

- **1D CNN** (best DL model): strong precision on Good (0.92) and Bad (0.83) classes, Moderate remains tougher (F1 0.755). Confusion matrix + prediction-vs-truth slice identify occasional Moderate↔Bad swaps.
- **LSTM**: better Moderate recall (0.74) but lower overall accuracy; useful as ensemble candidate.
- **Hybrid**: similar to LSTM, indicating spectrogram branch adds minimal signal with current STFT resolution.
- **2D CNN**: underperforms likely due to extremely short spectrograms (9×1); consider longer windows or mel spectrograms if this branch is required.

All models include training/validation curves, normalized confusion matrices, residual histograms, and raw predictions in `reports/phase3/dl/<model>/`.

## Operational Checks

- Model summaries logged (`model_summary.txt`) showing parameter counts and layer layouts.
- Shape printouts confirm `(20,6)` time windows and `(9,1,1)` spectrograms feeding every model.
- Early stopping (patience 8) prevented overfitting; best weights restored before evaluation.
- Best-model diagnostics: `prediction_vs_truth.png` (continuous 100-window slice) and `residual_histogram.png` for confidence auditing.

## Recommendations

1. **Production Candidate**: 1D CNN from `reports/phase3/dl/cnn1d/` – convert to TensorFlow SavedModel or TFLite for deployment; add post-training calibration if needed.
2. **Moderate Class**: Despite improvements, Moderate remains the least separable. Consider increased window length or richer spectral features before final signoff.
3. **Future Enhancements**:
   - Tune PyCaret’s top models (GBC, RF, LightGBM) using the DL features as inputs.
   - Experiment with mel-spectrograms or CWT scalograms to strengthen the 2D branch.
   - Investigate data augmentation (noise injection, mixup) to bolster Moderate samples.

## Next Steps

- Package `cnn1d` model with preprocessing pipeline for inference.
- Merge classical (PyCaret) and DL insights into a final executive summary or slide deck.
- Decide on Moderate handling (retain 3 classes vs. merge) based on business tolerance for misclassification.
