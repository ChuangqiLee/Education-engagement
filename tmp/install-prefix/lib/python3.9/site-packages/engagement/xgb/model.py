"""XGBoost training/persistence wrapper with the paper parameter preset."""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np


PAPER_PARAMS = {
    "learning_rate": 0.1,
    "n_estimators": 100,
    "max_depth": 5,
    "min_child_weight": 3,
    "subsample": 0.9,
    "colsample_bytree": 0.8,
}


class LandmarkXGBoost:
    def __init__(self, mode: str = "practical_reproduction", **params: Any):
        if mode not in ("practical_reproduction", "paper_faithful"):
            raise ValueError("mode must be practical_reproduction or paper_faithful")
        self.mode = mode
        self.params = dict(PAPER_PARAMS)
        self.params.update(params)
        self.model = None
        self.label_encoder = None

    def _check_mode(self) -> None:
        if self.mode == "paper_faithful":
            raise NotImplementedError(
                "[UNRESOLVED] The paper does not specify how XGBoost detects 468 landmarks. "
                "Paper-faithful training cannot be completed without original targets/features."
            )

    def fit(self, features: np.ndarray, labels: Sequence[Any]):
        self._check_mode()
        try:
            from sklearn.preprocessing import LabelEncoder
            from xgboost import XGBClassifier
        except ImportError as exc:
            raise RuntimeError("Install xgboost and scikit-learn to train this module") from exc
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[0] != len(labels):
            raise ValueError("features must be [samples,features] and align with labels")
        self.label_encoder = LabelEncoder().fit(list(labels))
        y = self.label_encoder.transform(list(labels))
        params = dict(self.params)
        params.setdefault("random_state", 42)
        params.setdefault("n_jobs", 1)
        if len(self.label_encoder.classes_) > 2:
            params.setdefault("objective", "multi:softprob")
            params.setdefault("num_class", len(self.label_encoder.classes_))
            params.setdefault("eval_metric", "mlogloss")
        else:
            params.setdefault("objective", "binary:logistic")
            params.setdefault("eval_metric", "logloss")
        self.model = XGBClassifier(**params)
        self.model.fit(x, y)
        return self

    def _require_fitted(self):
        if self.model is None or self.label_encoder is None:
            raise RuntimeError("model is not fitted")

    def predict(self, features: np.ndarray) -> np.ndarray:
        self._require_fitted()
        encoded = self.model.predict(np.asarray(features, dtype=np.float32))
        return self.label_encoder.inverse_transform(np.asarray(encoded, dtype=int))

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        self._require_fitted()
        return np.asarray(self.model.predict_proba(np.asarray(features, dtype=np.float32)))

    def evaluate(self, features: np.ndarray, labels: Sequence[Any]) -> Dict[str, float]:
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support

        prediction = self.predict(features)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, prediction, average="macro", zero_division=0)
        return {
            "accuracy": float(accuracy_score(labels, prediction)),
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
        }

    def feature_importance(self) -> np.ndarray:
        self._require_fitted()
        return np.asarray(self.model.feature_importances_)

    def save(self, path: Union[str, Path]) -> None:
        self._require_fitted()
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("joblib is required for model persistence") from exc
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: Union[str, Path]):
        try:
            import joblib
        except ImportError as exc:
            raise RuntimeError("joblib is required for model persistence") from exc
        loaded = joblib.load(path)
        if not isinstance(loaded, cls):
            raise TypeError("checkpoint is not a LandmarkXGBoost")
        return loaded

