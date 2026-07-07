import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from typing import Optional

from src.config import MODELS_DIR, RANDOM_STATE, ANOMALY_CONTAMINATION


ANOMALY_MODEL_PATH = MODELS_DIR / "anomaly_model.pkl"


class AnomalyDetector:
    def __init__(self):
        self._model: Optional[IsolationForest] = None
        self._feature_cols: list[str] = []

    def train(self, df: pd.DataFrame, feature_cols: list[str]) -> dict:
        self._feature_cols = feature_cols
        X = df[feature_cols].values

        model = IsolationForest(
            n_estimators=200,
            contamination=ANOMALY_CONTAMINATION,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model.fit(X)

        self._model = model
        joblib.dump(model, ANOMALY_MODEL_PATH)
        joblib.dump(feature_cols, MODELS_DIR / "anomaly_features.pkl")

        preds = model.predict(X)

        n_anomalies = int((preds == -1).sum())
        return {
            "total_checked": len(X),
            "anomalies_found": n_anomalies,
            "anomaly_rate": round(n_anomalies / len(X), 4),
        }

    def load(self):
        if ANOMALY_MODEL_PATH.exists():
            self._model = joblib.load(ANOMALY_MODEL_PATH)
            self._feature_cols = joblib.load(MODELS_DIR / "anomaly_features.pkl")
        return self

    def detect(self, feature_vector: dict) -> dict:
        if self._model is None:
            self.load()
        X = np.array([[feature_vector[f] for f in self._feature_cols]])
        score = float(self._model.decision_function(X)[0])
        pred = int(self._model.predict(X)[0])
        return {
            "customer_id": feature_vector.get("CustomerID", "unknown"),
            "anomaly_score": round(float(score), 6),
            "is_anomaly": pred == -1,
            "reconstruction_error": round(float(-score + 0.5), 6),
        }

    def detect_batch(self, df: pd.DataFrame) -> list[dict]:
        if self._model is None:
            self.load()
        X = df[self._feature_cols].values
        scores = self._model.decision_function(X)
        preds = self._model.predict(X)
        results = []
        for i, (_, row) in enumerate(df.iterrows()):
            results.append({
                "customer_id": row.get("CustomerID", "unknown"),
                "anomaly_score": round(float(scores[i]), 6),
                "is_anomaly": preds[i] == -1,
                "reconstruction_error": round(float(-scores[i] + 0.5), 6),
            })
        return results

    def is_loaded(self) -> bool:
        return self._model is not None


anomaly_detector = AnomalyDetector()
