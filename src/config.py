"""Project-wide configuration using pathlib — no hardcoded absolute paths."""

from pathlib import Path
import os

# Project root: auto-detected (two levels up from src/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data directories
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# Models
MODELS_DIR = PROJECT_ROOT / "models"

# Reports
REPORTS_DIR = PROJECT_ROOT / "reports"

# Raw data file (source transactions)
RAW_DATA_PATH = RAW_DIR / "transactions.csv"

# Processed data files
PROCESSED_FILES = {
    "features": PROCESSED_DIR / "customer_features.csv",
    "scaled": PROCESSED_DIR / "customer_features_scaled.csv",
    "selected": PROCESSED_DIR / "customer_features_selected.csv",
    "with_clusters": PROCESSED_DIR / "customer_features_with_clusters.csv",
    "personas": PROCESSED_DIR / "customer_personas.csv",
}

# Serialized models
MODEL_FILES = {
    "scaler": MODELS_DIR / "scaler.pkl",
    "pca": MODELS_DIR / "pca.pkl",
    "kmeans": MODELS_DIR / "kmeans.pkl",
    "selected_features": MODELS_DIR / "selected_features.pkl",
}

# Pipeline config
RANDOM_STATE = 42
TEST_HOLDOUT_SIZE = 0.2
CORRELATION_THRESHOLD = 0.85
VARIANCE_THRESHOLD = 0.01
PCA_VARIANCE_TARGET = 0.95
KMEANS_K = 4
DBSCAN_EPS = 1.5
DBSCAN_MIN_SAMPLES = 5

# Persona mapping
PERSONA_MAP = {
    0: "VIP Loyal Customers",
    1: "Discount Hunters",
    2: "Churn Risk",
    3: "One-Time Buyers",
}

PERSONA_DESCRIPTIONS = {
    "VIP Loyal Customers": "High spending, frequent purchases, low recency. Core revenue drivers.",
    "Discount Hunters": "Low spending, only during sales/discounts. Price-sensitive segment.",
    "Churn Risk": "Inactive for months, low frequency, high recency. Need re-engagement.",
    "One-Time Buyers": "Bought once, never returned. Low monetary & frequency.",
}

BUSINESS_RECOMMENDATIONS = {
    "VIP Loyal Customers": [
        "Exclusive rewards & VIP loyalty program",
        "Early access to new products",
        "Personalized recommendations",
    ],
    "Discount Hunters": [
        "Targeted coupon campaigns",
        "Flash sale notifications",
        "Bundle deals to increase basket size",
    ],
    "Churn Risk": [
        "Win-back email campaigns",
        "Special reactivation discounts",
        "Survey to understand inactivity",
    ],
    "One-Time Buyers": [
        "Cross-selling on complementary products",
        "Post-purchase follow-up sequence",
        "Free shipping on second purchase",
    ],
}

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
