import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import shap
import numpy as np
import joblib
from typing import Optional

from src.config import MODEL_FILES, PERSONA_MAP


class PersonaExplainer:
    def __init__(self):
        self._explainer: Optional[shap.Explainer] = None
        self._feature_names: list[str] = []

    def load(self):
        kmeans = joblib.load(MODEL_FILES["kmeans"])
        scaler = joblib.load(MODEL_FILES["scaler"])
        pca = joblib.load(MODEL_FILES["pca"])
        selected_features = joblib.load(MODEL_FILES["selected_features"])
        self._feature_names = selected_features

        def model_pipeline(X: np.ndarray) -> np.ndarray:
            X_scaled = scaler.transform(X)
            X_pca = pca.transform(X_scaled)
            return kmeans.predict(X_pca)

        class Wrapper:
            def __init__(self, predict_fn, scaler, pca, kmeans):
                self.predict_fn = predict_fn
                self.scaler = scaler
                self.pca = pca
                self.kmeans = kmeans

            def predict(self, X: np.ndarray) -> np.ndarray:
                X_scaled = self.scaler.transform(X)
                X_pca = self.pca.transform(X_scaled)
                return self.kmeans.predict(X_pca)

        self._wrapped = Wrapper(model_pipeline, scaler, pca, kmeans)
        background = np.random.randn(100, len(selected_features))
        self._explainer = shap.KernelExplainer(self._wrapped.predict, background)
        return self

    def explain(self, feature_vector: dict) -> dict:
        if self._explainer is None:
            self.load()
        X = np.array([[feature_vector[f] for f in self._feature_names]])
        shap_values = self._explainer.shap_values(X)
        cluster = int(self._wrapped.predict(X)[0])
        persona = PERSONA_MAP.get(cluster, "Unknown")

        if isinstance(shap_values, list):
            cluster_shap = shap_values[cluster]
        else:
            cluster_shap = shap_values[0]

        contributions = []
        for i, feat in enumerate(self._feature_names):
            val = float(cluster_shap[i])
            contributions.append({
                "feature": feat,
                "importance": round(abs(val), 6),
                "direction": "positive" if val > 0 else "negative",
            })
        contributions.sort(key=lambda x: x["importance"], reverse=True)

        return {
            "customer_id": feature_vector.get("CustomerID", "unknown"),
            "persona": persona,
            "base_value": float(self._explainer.expected_value[cluster] if isinstance(self._explainer.expected_value, list) else self._explainer.expected_value),
            "contributions": contributions,
        }

    def is_loaded(self) -> bool:
        return self._explainer is not None


persona_explainer = PersonaExplainer()
