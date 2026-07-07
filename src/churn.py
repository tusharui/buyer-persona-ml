import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score

from src.config import MODELS_DIR, CHURN_TEST_SIZE, RANDOM_STATE


CHURN_MODEL_PATH = MODELS_DIR / "churn_model.pkl"
CHURN_FEATURES_PATH = MODELS_DIR / "churn_features.pkl"


def prepare_churn_labels(cust_df: pd.DataFrame) -> pd.DataFrame:
    df = cust_df.copy()
    high_recency_threshold = df["Recency"].quantile(0.75)
    low_frequency_threshold = df["Frequency"].quantile(0.25)
    df["ChurnLabel"] = (
        (df["Recency"] > high_recency_threshold) &
        (df["Frequency"] <= low_frequency_threshold)
    ).astype(int)
    return df


def train_churn_model(feature_df: pd.DataFrame, feature_cols: list[str]) -> dict:
    df = prepare_churn_labels(feature_df)

    X = df[feature_cols].values
    y = df["ChurnLabel"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CHURN_TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "roc_auc": round(roc_auc_score(y_test, y_prob), 4),
        "avg_precision": round(average_precision_score(y_test, y_prob), 4),
        "churn_rate": float(y.mean()),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    joblib.dump(model, CHURN_MODEL_PATH)
    joblib.dump(feature_cols, CHURN_FEATURES_PATH)

    importances = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    return {"model": model, "metrics": metrics, "importance": importances}


def predict_churn(feature_vector: dict, model=None, feature_cols: list[str] = None) -> dict:
    if model is None:
        model = joblib.load(CHURN_MODEL_PATH)
    if feature_cols is None:
        feature_cols = joblib.load(CHURN_FEATURES_PATH)

    X = np.array([[feature_vector[f] for f in feature_cols]])
    prob = float(model.predict_proba(X)[:, 1][0])

    if prob < 0.3:
        risk = "Low"
    elif prob < 0.6:
        risk = "Medium"
    else:
        risk = "High"

    if hasattr(model, "feature_importances_"):
        top_indices = np.argsort(model.feature_importances_)[-3:][::-1]
        top_factors = [
            {"feature": feature_cols[i], "importance": float(model.feature_importances_[i]), "direction": "positive"}
            for i in top_indices
        ]
    else:
        top_factors = []

    return {
        "customer_id": feature_vector.get("CustomerID", "unknown"),
        "churn_probability": round(prob, 4),
        "churn_risk": risk,
        "top_factors": top_factors,
    }
