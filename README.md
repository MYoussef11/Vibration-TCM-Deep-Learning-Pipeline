# Vibration Tool Condition Monitoring Pipeline

End-to-end deep learning pipeline for classifying machine tool condition (Binary: Good vs Faulty) from vibration sensor data. The work is organized into three phases:

1. **Data Verification & Preprocessing** – validate sampling rates, clean raw signals, and define the preprocessing recipe.
2. **Unsupervised Analysis** – extract handcrafted features, explore clusters (PCA/UMAP + KMeans/GMM), and confirm the final labeling strategy.
3. **Supervised Modeling** – train classical ML baselines plus 1D/2D/Hybrid deep networks on windows and spectrograms.

## Repository Layout

```
.
├── Data/                  # Provided CSV files grouped by condition
├── scripts/
│   ├── run_phase1.py      # Sampling-frequency checks & quality reports
│   └── run_phase2.py      # Feature extraction, clustering, and plots
├── tcm_pipeline/
│   ├── data_loader.py     # DataLoader class + validation
│   ├── preprocessing.py   # Preprocessor + quality checks + config builder
│   ├── feature_extraction.py
│   ├── unsupervised.py
│   ├── supervised.py
│   └── reporting.py
├── requirements.txt
└── README.md
```

## Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate 
pip install --upgrade pip
pip install -r requirements.txt
```

Key libraries: `pandas`, `numpy`, `scipy`, `scikit-learn`, `umap-learn`, `tensorflow`, `xgboost`, plus plotting packages for future visualization.

## Phase 1 – Data Verification

Run sampling-rate checks, quality audits, and emit the preprocessing config:

```bash
python scripts/run_phase1.py
```

Outputs in `reports/phase1/`.

## Phase 2 – Feature Extraction & Unsupervised Analysis

Generate normalized windows, handcrafted features, and clustering visuals:

```bash
python scripts/run_phase2.py
```

Outputs in `reports/phase2/` (`window_features.csv`, `unsupervised_summary.json`, `figures/*.png`).

## Phase 3 – Supervised Modeling (GroupKFold)

- Build group-aware datasets (includes file-based group IDs):

```bash
python scripts/build_phase3_datasets.py
```

- Run classical tuning with GroupKFold (GBC/XGBoost/RF):

```bash
python scripts/tune_classical_models.py
```

Artifacts in `reports/phase3/classical_tuning/`.

- Train and Tune Deep Learning Models (Optuna + GroupKFold):
  
  This script handles hyperparameter tuning, final model training, and evaluation (Confusion Matrix, Classification Report).

```bash
# Train 1D CNN
python scripts/tune_dl_models.py --dataset artifacts/phase3/phase3_datasets.npz --model cnn1d --trials 15 --epochs 40

# Train LSTM
python scripts/tune_dl_models.py --dataset artifacts/phase3/phase3_datasets.npz --model lstm --trials 15 --epochs 40

# Train 2D CNN (Spectrograms)
python scripts/tune_dl_models.py --dataset artifacts/phase3/phase3_datasets.npz --model cnn2d --trials 15 --epochs 40
```

Artifacts in `reports/phase3/dl_tuning/`.

- (Optional) Baseline Training without Tuning:

```bash
python scripts/train_phase3_models.py
```

## Reports

- `reports/phase3/FINAL_REPORT.md` – legacy results (random splits).
- `reports/phase3/FINAL_REPORT_TUNED.md` – GroupKFold/tuning outcomes, including max-spec summary.
