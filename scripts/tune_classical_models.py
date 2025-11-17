"""Hyperparameter tuning for classical models with GroupKFold."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import (
    GroupKFold,
    RandomizedSearchCV,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

LABEL_MAP = {"good": 0, "moderate": 1, "bad": 2}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tune classical ML models with GroupKFold validation.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=Path("reports") / "phase2" / "window_features.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports") / "phase3" / "classical_tuning",
    )
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def load_dataset(features_path: Path) -> pd.DataFrame:
    if not features_path.exists():
        raise FileNotFoundError(f"Feature file not found: {features_path}")
    df = pd.read_csv(features_path)
    required = {"file", "label"}
    if not required.issubset(df.columns):
        raise ValueError(f"Feature file must contain columns: {required}")
    df["label_id"] = df["label"].str.lower().map(LABEL_MAP)
    if df["label_id"].isna().any():
        raise ValueError("Found labels outside {good, moderate, bad}.")
    return df


def to_serializable(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def get_models(random_state: int) -> Dict[str, Dict]:
    models: Dict[str, Dict] = {
        "gbc": {
            "estimator": GradientBoostingClassifier(random_state=random_state),
            "param_distributions": {
                "clf__n_estimators": np.linspace(50, 400, 15, dtype=int),
                "clf__learning_rate": np.linspace(0.01, 0.2, 20),
                "clf__max_depth": [2, 3, 4, 5, 6],
                "clf__min_samples_split": [2, 3, 4, 5],
                "clf__min_samples_leaf": [1, 2, 3, 4],
                "clf__subsample": np.linspace(0.6, 1.0, 9),
            },
        },
        "xgboost": {
            "estimator": XGBClassifier(
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=random_state,
                tree_method="hist",
            ),
            "param_distributions": {
                "clf__n_estimators": np.linspace(100, 600, 20, dtype=int),
                "clf__max_depth": [3, 4, 5, 6, 7],
                "clf__learning_rate": np.logspace(-3, -1, 10),
                "clf__subsample": np.linspace(0.6, 1.0, 9),
                "clf__colsample_bytree": np.linspace(0.6, 1.0, 9),
                "clf__reg_lambda": np.logspace(-2, 2, 10),
            },
        },
        "rf": {
            "estimator": RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1,
            ),
            "param_distributions": {
                "clf__n_estimators": np.linspace(100, 600, 20, dtype=int),
                "clf__max_depth": [None, 10, 20, 30, 40],
                "clf__min_samples_split": [2, 3, 4, 5],
                "clf__min_samples_leaf": [1, 2, 3, 4],
                "clf__max_features": ["sqrt", "log2", 0.5, 0.7, 0.9],
            },
        },
    }
    return models


def tune_model(
    model_id: str,
    estimator,
    param_distributions: Dict,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    cv: GroupKFold,
    n_iter: int,
    random_state: int,
) -> Dict:
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", estimator),
        ]
    )
    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        n_jobs=-1,
        verbose=1,
        random_state=random_state,
        scoring="accuracy",
    )
    search.fit(X, y, groups=groups)
    best_estimator = search.best_estimator_
    predictions = cross_val_predict(
        best_estimator,
        X,
        y,
        cv=cv,
        groups=groups,
        n_jobs=-1,
    )
    report = classification_report(y, predictions, output_dict=True)
    results = {
        "model": model_id,
        "best_score": float(search.best_score_),
        "best_params": search.best_params_,
        "classification_report": report,
    }
    return results, best_estimator


def main() -> None:
    args = parse_args()
    df = load_dataset(args.features)

    feature_cols = [col for col in df.columns if col not in {"file", "label", "window_index", "label_id"}]
    X = df[feature_cols]
    y = df["label_id"].to_numpy()
    groups = df["file"].astype("category").cat.codes.to_numpy()

    cv = GroupKFold(n_splits=args.folds)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    models = get_models(args.random_state)
    summaries: List[Dict] = []
    best_global = None

    for model_id, spec in models.items():
        print(f"\n=== Tuning {model_id.upper()} ===")
        summary, estimator = tune_model(
            model_id=model_id,
            estimator=spec["estimator"],
            param_distributions=spec["param_distributions"],
            X=X,
            y=y,
            groups=groups,
            cv=cv,
            n_iter=args.n_iter,
            random_state=args.random_state,
        )
        summaries.append(summary)
        (output_dir / f"{model_id}_summary.json").write_text(
            json.dumps(summary, indent=2, default=to_serializable),
            encoding="utf-8",
        )
        if best_global is None or summary["best_score"] > best_global["best_score"]:
            best_global = {**summary, "estimator_repr": repr(estimator)}

    (output_dir / "model_comparison.json").write_text(
        json.dumps(summaries, indent=2, default=to_serializable),
        encoding="utf-8",
    )
    if best_global:
        (output_dir / "best_model.json").write_text(
            json.dumps(best_global, indent=2, default=to_serializable),
            encoding="utf-8",
        )
        print(f"\nBest model: {best_global['model']} (accuracy={best_global['best_score']:.3f})")


if __name__ == "__main__":
    main()
