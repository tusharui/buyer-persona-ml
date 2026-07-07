
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"

RAW_DATA_PATH = RAW_DIR / "transactions.csv"

PROCESSED_FILES = {
    "features": PROCESSED_DIR / "customer_features.csv",
    "scaled": PROCESSED_DIR / "customer_features_scaled.csv",
    "selected": PROCESSED_DIR / "customer_features_selected.csv",
    "with_clusters": PROCESSED_DIR / "customer_features_with_clusters.csv",
    "personas": PROCESSED_DIR / "customer_personas.csv",
}

MODEL_FILES = {
    "scaler": MODELS_DIR / "scaler.pkl",
    "pca": MODELS_DIR / "pca.pkl",
    "kmeans": MODELS_DIR / "kmeans.pkl",
    "selected_features": MODELS_DIR / "selected_features.pkl",
}

RANDOM_STATE = 42
TEST_HOLDOUT_SIZE = 0.2
CORRELATION_THRESHOLD = 0.85
VARIANCE_THRESHOLD = 0.01
PCA_VARIANCE_TARGET = 0.95
KMEANS_K = 4
DBSCAN_EPS = 1.5
DBSCAN_MIN_SAMPLES = 5

OPTUNA_N_TRIALS = 50
CHURN_TEST_SIZE = 0.3
FORECAST_DAYS = 90
ANOMALY_CONTAMINATION = 0.05

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/buyer_persona_ml",
)
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
PREDICTION_CACHE_TTL = int(os.getenv("PREDICTION_CACHE_TTL", "86400"))
FEATURE_CACHE_TTL = int(os.getenv("FEATURE_CACHE_TTL", "3600"))

LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_API_BASE = os.getenv("LLM_API_BASE", "https://api.groq.com/openai/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "buyer-persona-transactions")
KAFKA_CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "buyer-persona-ml")

CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(PROJECT_ROOT / "chroma_db"))

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

for d in [RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)
