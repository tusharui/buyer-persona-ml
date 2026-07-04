from pathlib import Path
from typing import Optional
import joblib
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from src.cache import cache


_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


class ModelLoader:
    def __init__(self):
        self._scaler: Optional[StandardScaler] = None
        self._pca: Optional[PCA] = None
        self._kmeans: Optional[KMeans] = None
        self._features: Optional[list[str]] = None
        self._version: str = "unknown"

    def load(self):
        scaler_path = _MODELS_DIR / "scaler.pkl"
        pca_path = _MODELS_DIR / "pca.pkl"
        kmeans_path = _MODELS_DIR / "kmeans.pkl"
        features_path = _MODELS_DIR / "selected_features.pkl"

        if not all(p.exists() for p in [scaler_path, pca_path, kmeans_path, features_path]):
            raise FileNotFoundError("Run 'python -m src.pipeline' first to train models.")

        self._scaler = joblib.load(scaler_path)
        self._pca = joblib.load(pca_path)
        self._kmeans = joblib.load(kmeans_path)
        self._features = joblib.load(features_path)
        self._version = f"kmeans_k{self._kmeans.n_clusters}"
        return self

    async def load_active_version(self):
        version = await cache.get("model:active")
        if version:
            self._version = version

    def is_loaded(self) -> bool:
        return all(v is not None for v in [self._scaler, self._pca, self._kmeans, self._features])

    @property
    def scaler(self) -> StandardScaler:
        return self._scaler

    @property
    def pca(self) -> PCA:
        return self._pca

    @property
    def kmeans(self) -> KMeans:
        return self._kmeans

    @property
    def features(self) -> list[str]:
        return self._features

    @property
    def version(self) -> str:
        return self._version


model_loader = ModelLoader()
