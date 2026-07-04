import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, entropy
from sklearn.preprocessing import StandardScaler

from src.config import MODELS_DIR, PROCESSED_DIR
from src.cache import cache

warnings.filterwarnings("ignore", category=RuntimeWarning)

PSI_THRESHOLD = 0.25
KS_THRESHOLD = 0.05
KL_THRESHOLD = 0.1


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    expected = np.array(expected).flatten()
    actual = np.array(actual).flatten()
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    if max_val - min_val < 1e-10:
        return 0.0
    bins = np.linspace(min_val, max_val, buckets + 1)
    expected_counts = np.histogram(expected, bins=bins)[0].astype(np.float64)
    actual_counts = np.histogram(actual, bins=bins)[0].astype(np.float64)
    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()
    expected_pct = np.clip(expected_pct, 1e-10, 1.0)
    actual_pct = np.clip(actual_pct, 1e-10, 1.0)
    psi_val = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return float(psi_val)


def kl_divergence(expected: np.ndarray, actual: np.ndarray, buckets: int = 20) -> float:
    expected = np.array(expected).flatten()
    actual = np.array(actual).flatten()
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())
    if max_val - min_val < 1e-10:
        return 0.0
    bins = np.linspace(min_val, max_val, buckets + 1)
    expected_pdf = np.histogram(expected, bins=bins, density=True)[0] + 1e-10
    actual_pdf = np.histogram(actual, bins=bins, density=True)[0] + 1e-10
    return float(entropy(expected_pdf, actual_pdf))


def ks_test(expected: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    expected = np.array(expected).flatten()
    actual = np.array(actual).flatten()
    stat, pval = ks_2samp(expected, actual)
    return float(stat), float(pval)


class DriftDetector:
    def __init__(self):
        self._baseline: Optional[pd.DataFrame] = None
        self._feature_cols: list[str] = []
        self._baseline_path: Path = PROCESSED_DIR / "drift_baseline_features.csv"
        self._results: dict = {}

    def load_baseline(self) -> bool:
        if self._baseline_path.exists():
            self._baseline = pd.read_csv(self._baseline_path)
            self._feature_cols = [c for c in self._baseline.columns if c != "CustomerID"]
            return True
        personas_path = PROCESSED_DIR / "customer_personas.csv"
        if personas_path.exists():
            df = pd.read_csv(personas_path)
            self._feature_cols = [c for c in df.columns if c not in (
                "CustomerID", "Cluster", "Persona", "PC1", "PC2"
            )]
            self._baseline = df[["CustomerID"] + self._feature_cols]
            self._save_baseline()
            return True
        return False

    def _save_baseline(self):
        if self._baseline is not None:
            self._baseline.to_csv(self._baseline_path, index=False)

    def set_baseline(self, df: pd.DataFrame):
        self._baseline = df
        self._feature_cols = [c for c in df.columns if c != "CustomerID"]
        self._save_baseline()

    def detect_drift(self, new_data: pd.DataFrame) -> dict:
        if self._baseline is None:
            return {"error": "No baseline loaded", "drift_detected": False}
        new_features = [c for c in self._feature_cols if c in new_data.columns]
        if not new_features:
            return {"error": "No matching features found in new data", "drift_detected": False}

        results = {}
        drift_count = 0
        total_checks = 0

        for col in new_features:
            expected = self._baseline[col].dropna().values
            actual = new_data[col].dropna().values
            if len(expected) < 5 or len(actual) < 5:
                continue

            psi_val = psi(expected, actual)
            kl_val = kl_divergence(expected, actual)
            ks_stat, ks_pval = ks_test(expected, actual)

            psi_drift = psi_val > PSI_THRESHOLD
            ks_drift = ks_pval < KS_THRESHOLD
            kl_drift = kl_val > KL_THRESHOLD

            col_result = {
                "psi": round(psi_val, 6),
                "kl_divergence": round(kl_val, 6),
                "ks_statistic": round(ks_stat, 6),
                "ks_pvalue": round(ks_pval, 6),
                "psi_drift": psi_drift,
                "ks_drift": ks_drift,
                "kl_drift": kl_drift,
                "drift_detected": psi_drift or ks_drift or kl_drift,
            }
            if col_result["drift_detected"]:
                drift_count += 1
            total_checks += 1
            results[col] = col_result

        overall = {
            "features_checked": total_checks,
            "features_with_drift": drift_count,
            "drift_detected": drift_count > 0,
            "drift_ratio": round(drift_count / max(total_checks, 1), 4),
        }
        self._results = {"overall": overall, "features": results}
        return self._results

    @property
    def results(self) -> dict:
        return self._results

    @property
    def baseline_loaded(self) -> bool:
        return self._baseline is not None

    @property
    def n_features(self) -> int:
        return len(self._feature_cols)


drift_detector = DriftDetector()
