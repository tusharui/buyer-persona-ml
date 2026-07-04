import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import joblib

from src.config import MODEL_FILES, RAW_DATA_PATH, PERSONA_MAP
from src.preprocessing import clean_data
from src.features import build_customer_features, FEATURE_COLS


def load_artifacts():
    scaler = joblib.load(MODEL_FILES["scaler"])
    pca = joblib.load(MODEL_FILES["pca"])
    kmeans = joblib.load(MODEL_FILES["kmeans"])
    selected_features = joblib.load(MODEL_FILES["selected_features"])
    return scaler, pca, kmeans, selected_features


def predict_personas(df_transactions, scaler, pca, kmeans, selected_features):
    df_clean = clean_data(df_transactions)
    cust = build_customer_features(df_clean)

    scale_cols = [c for c in FEATURE_COLS if c in cust.columns]
    cust_scaled = cust.copy()
    cust_scaled[scale_cols] = scaler.transform(cust[scale_cols])

    X = cust_scaled[selected_features].values
    X_pca = pca.transform(X)

    clusters = kmeans.predict(X_pca)
    cust["Cluster"] = clusters
    cust["Persona"] = cust["Cluster"].map(PERSONA_MAP)

    return cust


def main():
    parser = argparse.ArgumentParser(description="Predict buyer personas from transaction data.")
    parser.add_argument("--input", type=str, default=str(RAW_DATA_PATH),
                        help="Path to new transaction CSV")
    parser.add_argument("--output", type=str, default="predictions.csv",
                        help="Path to save predictions CSV")
    args = parser.parse_args()

    print("Loading artifacts...")
    scaler, pca, kmeans, selected_features = load_artifacts()
    print(f"  Loaded: scaler, PCA ({pca.n_components_} components), "
          f"KMeans (k={kmeans.n_clusters}), {len(selected_features)} features")

    print(f"Reading transactions from: {args.input}")
    df = pd.read_csv(args.input, parse_dates=["InvoiceDate"])
    print(f"  Raw rows: {len(df)}")

    predictions = predict_personas(df, scaler, pca, kmeans, selected_features)
    predictions.to_csv(args.output, index=False)
    print(f"  Predictions: {len(predictions)} customers")
    print("  Persona distribution:")
    print(predictions["Persona"].value_counts().to_string())
    print(f"\nSaved to: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"ERROR: File not found — {e}")
        print("Run 'python -m src.pipeline' first to train and save models.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Prediction failed — {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
