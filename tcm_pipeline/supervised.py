"""Supervised modeling strategies for tool condition monitoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


@dataclass
class SupervisedResult:
    """Holds metrics for a trained supervised model."""

    report: Dict[str, Dict[str, float]]
    confusion: np.ndarray


class SupervisedModel:
    """Trains baseline classifiers on engineered features."""

    def __init__(self, n_splits: int = 5, random_state: int = 42) -> None:
        self.n_splits = n_splits
        self.random_state = random_state

    def build_model(self) -> Pipeline:
        """Return a simple scalable scikit-learn pipeline."""
        classifier = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            objective="multi:softprob",
            eval_metric="mlogloss",
        )
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("clf", classifier),
            ]
        )

    def train_and_evaluate(self, features: pd.DataFrame, labels: pd.Series) -> SupervisedResult:
        """Cross-validate the baseline classifier and aggregate metrics."""
        kfold = StratifiedKFold(
            n_splits=self.n_splits,
            shuffle=True,
            random_state=self.random_state,
        )
        y_true_all: list[int] = []
        y_pred_all: list[int] = []
        for train_idx, test_idx in kfold.split(features, labels):
            X_train, X_test = features.iloc[train_idx], features.iloc[test_idx]
            y_train, y_test = labels.iloc[train_idx], labels.iloc[test_idx]
            model = self.build_model()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_true_all.extend(y_test.tolist())
            y_pred_all.extend(y_pred.tolist())
        report = classification_report(y_true_all, y_pred_all, output_dict=True)
        confusion = confusion_matrix(y_true_all, y_pred_all)
        return SupervisedResult(report=report, confusion=confusion)
