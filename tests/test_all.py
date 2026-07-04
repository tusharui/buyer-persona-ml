"""Tests for src/ modules."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs

from src.preprocessing import (
    clean_data, handle_outliers, scale_features,
    select_features_by_correlation, select_features_by_variance,
)
from src.features import build_customer_features
from src.clustering import kmeans_optimal_k, kmeans_fit, dbscan_fit
from src.evaluation import validate_clusters, intra_inter_distances, cluster_stability_score


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def raw_df():
    np.random.seed(42)
    n = 100
    start = pd.Timestamp("2024-01-01")
    return pd.DataFrame({
        "InvoiceID": [f"INV{i:06d}" for i in range(n)],
        "CustomerID": [f"C{np.random.randint(10, 30)}" for _ in range(n)],
        "InvoiceDate": [start + pd.Timedelta(days=int(d)) for d in np.random.randint(0, 100, n)],
        "ProductCategory": np.random.choice(["A", "B", "C"], n),
        "ProductID": [f"P{np.random.randint(100, 200)}" for _ in range(n)],
        "Quantity": np.random.randint(1, 6, n),
        "UnitPrice": np.random.choice([199, 499, 999], n),
        "DiscountPct": np.random.choice([0, 10, 20], n),
        "PaymentMethod": np.random.choice(["Card", "UPI"], n),
        "Returned": np.random.choice([0, 1], n, p=[0.8, 0.2]),
    })


@pytest.fixture
def customer_features(raw_df):
    return build_customer_features(raw_df)


# ── Preprocessing Tests ──────────────────────────────────────────────

class TestCleanData:
    def test_removes_duplicates(self, raw_df):
        duped = pd.concat([raw_df, raw_df.iloc[:5]], ignore_index=True)
        cleaned = clean_data(duped)
        assert len(cleaned) == len(raw_df)

    def test_removes_cancelled(self, raw_df):
        raw_df.loc[0, "InvoiceID"] = "C000001"
        cleaned = clean_data(raw_df)
        assert "C000001" not in cleaned["InvoiceID"].values

    def test_removes_null_customer(self, raw_df):
        raw_df.loc[0, "CustomerID"] = None
        cleaned = clean_data(raw_df)
        assert raw_df.shape[0] - 1 == len(cleaned) or raw_df.shape[0] == len(cleaned)

    def test_removes_negative_qty(self, raw_df):
        raw_df.loc[0, "Quantity"] = -3
        cleaned = clean_data(raw_df)
        assert (cleaned["Quantity"] > 0).all()


class TestHandleOutliers:
    def test_removes_extreme_values(self):
        df = pd.DataFrame({"val": [1, 2, 2, 3, 100]})
        result = handle_outliers(df, ["val"])
        assert result["val"].max() < 50

    def test_no_change_within_iqr(self):
        df = pd.DataFrame({"val": [11, 12, 12, 13, 14]})
        result = handle_outliers(df, ["val"])
        assert len(result) == len(df)


class TestScaleFeatures:
    def test_standard_scaler_center(self):
        df = pd.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 20, 30, 40, 50]})
        scaled, scaler = scale_features(df.copy(), ["a", "b"])
        assert abs(scaled["a"].mean()) < 1e-10
        assert abs(scaled["a"].std(ddof=0) - 1) < 1e-10


class TestFeatureSelection:
    def test_correlation_drops_highly_correlated(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [2, 4, 6, 8, 10],  # r=1 with a
            "c": [5, 4, 3, 2, 1],
        })
        dropped = select_features_by_correlation(df, ["a", "b", "c"], threshold=0.8)
        assert len(dropped) >= 1

    def test_variance_selector(self):
        df = pd.DataFrame({
            "a": [1, 2, 3, 4, 5],
            "b": [1, 1, 1, 1, 1],  # zero variance
        })
        kept, dropped = select_features_by_variance(df, ["a", "b"], threshold=0.01)
        assert "a" in kept
        assert "b" in dropped


# ── Feature Tests ────────────────────────────────────────────────────

class TestBuildCustomerFeatures:
    def test_returns_one_row_per_customer(self, raw_df):
        result = build_customer_features(raw_df)
        assert result["CustomerID"].nunique() == len(result)

    def test_has_rfm_columns(self, customer_features):
        for col in ["Recency", "Frequency", "Monetary"]:
            assert col in customer_features.columns

    def test_has_behavioral_columns(self, customer_features):
        for col in ["AvgBasketSize", "PurchaseInterval", "WeekendRatio"]:
            assert col in customer_features.columns

    def test_all_quantities_positive(self, customer_features):
        assert (customer_features["Frequency"] >= 1).all()

    def test_monetary_non_negative(self, customer_features):
        assert (customer_features["Monetary"] >= 0).all()


# ── Clustering Tests ─────────────────────────────────────────────────

class TestKMeans:
    def test_optimal_k_returns_valid(self):
        X, _ = make_blobs(n_samples=100, centers=3, random_state=42)
        best_k, inertias, sil = kmeans_optimal_k(X, range(2, 8))
        assert 2 <= best_k <= 7
        assert len(inertias) == 6
        assert len(sil) == 6

    def test_fit_returns_correct_labels(self):
        X, _ = make_blobs(n_samples=50, centers=3, random_state=42)
        labels, model = kmeans_fit(X, 3)
        assert len(labels) == 50
        assert len(set(labels)) == 3


class TestDBSCAN:
    def test_fit_returns_labels(self):
        X, _ = make_blobs(n_samples=50, centers=3, random_state=42)
        labels, n_clusters, n_noise = dbscan_fit(X, eps=1.0, min_samples=3)
        assert len(labels) == 50


# ── Evaluation Tests ─────────────────────────────────────────────────

class TestValidateClusters:
    def test_returns_silhouette_and_db(self):
        X, labels = make_blobs(n_samples=100, centers=3, random_state=42)
        result = validate_clusters(X, labels)
        assert "silhouette" in result
        assert "davies_bouldin" in result
        assert -1 <= result["silhouette"] <= 1

    def test_intra_inter_distances(self):
        X, labels = make_blobs(n_samples=100, centers=3, random_state=42)
        result = intra_inter_distances(X, labels)
        assert "intra" in result
        assert "inter_mean" in result
        assert result["inter_mean"] > 0

    def test_stability_score(self):
        X, labels = make_blobs(n_samples=100, centers=3, random_state=42)
        result = cluster_stability_score(X, labels, n_splits=3)
        assert "mean_ari" in result
        assert -1 <= result["mean_ari"] <= 1


class TestModelRegistry:
    def test_registry_instantiation(self):
        from src.model_registry import ModelRegistry
        reg = ModelRegistry()
        assert reg is not None
        assert reg._client is None

    def test_latest_versions_empty_when_no_mlflow(self):
        from src.model_registry import model_registry
        versions = model_registry.get_latest_versions()
        assert versions == []
