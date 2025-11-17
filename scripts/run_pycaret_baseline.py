"""PyCaret baseline comparison for Phase 3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from pycaret.classification import compare_models, finalize_model, predict_model, pull, save_model, setup
from sklearn.metrics import classification_report, confusion_matrix

LABEL_MAP = {"good": 0, "moderate": 1, "bad": 2}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PyCaret baseline on engineered features.")
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("reports") / "phase2" / "window_features.csv",
        help="Path to the feature matrix CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "phase3" / "pycaret",
        help="Directory to store comparison tables and metrics.",
    )
    parser.add_argument("--train-size", type=float, default=0.8, help="Fraction for training during setup.")
    parser.add_argument("--session-id", type=int, default=42, help="Random seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.features)
    if "label" not in df.columns:
        raise ValueError("Feature CSV must contain a 'label' column.")

    df["label_id"] = df["label"].str.lower().map(LABEL_MAP)
    if df["label_id"].isna().any():
        missing = df[df["label_id"].isna()]["label"].unique()
        raise ValueError(f"Unrecognized labels: {missing}")

    feature_df = df.drop(columns=["file", "label", "window_index"])

    s = setup(
        data=feature_df,
        target="label_id",
        train_size=args.train_size,
        session_id=args.session_id,
        normalize=True,
        transformation=True,
        verbose=False,
    )

    best_model = compare_models()
    comparison_table = pull()

    finalized_model = finalize_model(best_model)
    predictions = predict_model(finalized_model)
    prediction_table = pull()

    y_true = predictions["label_id"]
    y_pred = predictions["prediction_label"]
    report = classification_report(y_true, y_pred, output_dict=True)
    confusion = confusion_matrix(y_true, y_pred)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_table.to_csv(output_dir / "pycaret_comparison.csv", index=False)
    prediction_table.to_csv(output_dir / "best_model_predictions.csv", index=False)
    (output_dir / "best_model_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    pd.DataFrame(confusion).to_csv(output_dir / "best_model_confusion_matrix.csv", index=False)
    save_model(finalized_model, str(output_dir / "best_model"))

    metrics_md = "\n".join(
        [
            "# PyCaret Baseline Results",
            "",
            f"- Training size fraction: **{args.train_size}**",
            f"- Session ID: **{args.session_id}**",
            f"- Best model: **{finalized_model}**",
            "",
            "See `pycaret_comparison.csv` for the full leaderboard and "
            "`best_model_report.json` for precision/recall/F1 details.",
        ]
    )
    (output_dir / "README.md").write_text(metrics_md, encoding="utf-8")

    print("PyCaret baseline completed.")
    print(f"Artifacts saved to {output_dir}")


if __name__ == "__main__":
    main()
