"""LightGBM compatibility shim for environments without native LightGBM."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .basic import LightGBMError


SUPPORTED_PARAMS = {
    "learning_rate",
    "max_depth",
    "min_samples_leaf",
    "l2_regularization",
    "max_leaf_nodes",
    "max_bins",
    "validation_fraction",
    "n_iter_no_change",
    "tol",
    "random_state",
}


def _translate_params(params: Dict[str, Any]) -> Dict[str, Any]:
    translated: Dict[str, Any] = {}
    for key, value in params.items():
        if key == "num_leaves":
            translated["max_leaf_nodes"] = value
        elif key == "n_estimators":
            translated["max_iter"] = value
        elif key == "feature_fraction":
            translated["max_features"] = value
        elif key in ("device", "max_bin", "max_bins"):
            continue
        elif key in SUPPORTED_PARAMS:
            translated[key] = value
    return translated


class LGBMClassifier:
    """Minimal drop-in replacement backed by HistGradientBoostingClassifier."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.model: Optional[HistGradientBoostingClassifier] = None

    def fit(self, X: np.ndarray, y: np.ndarray, **fit_params: Any) -> "LGBMClassifier":
        params = _translate_params({**self.kwargs, **fit_params})
        self.model = HistGradientBoostingClassifier(**params)
        self.model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise LightGBMError("Model not fitted.")
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise LightGBMError("Model not fitted.")
        return self.model.predict_proba(X)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        if self.model is None:
            raise LightGBMError("Model not fitted.")
        return self.model.score(X, y)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        return dict(self.kwargs)

    def set_params(self, **params: Any) -> "LGBMClassifier":
        self.kwargs.update(params)
        return self
