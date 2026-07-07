import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

from src.config import (
    PROCESSED_FILES, MODEL_FILES,
    RANDOM_STATE, KMEANS_K,
    PCA_VARIANCE_TARGET, PERSONA_MAP,
)
from src.preprocessing import (
    load_raw_data, clean_data, handle_outliers,
    scale_features, select_features_by_correlation,
)
from src.features import build_customer_features, FEATURE_COLS
from src.clustering import kmeans_fit, kmeans_optimal_k
from src.tracking import ExperimentLogger
from src.database import AsyncSessionLocal
from sqlalchemy import text


async def load_from_neon():
    async with AsyncSessionLocal() as session:
        result = await session.execute(text("""
            SELECT invoice_id, customer_id, invoice_date, product_category,
                   product_id, quantity, unit_price, discount_pct,
                   payment_method, returned
            FROM transactions
            ORDER BY invoice_date
        """))
        rows = result.fetchall()
        df = pd.DataFrame(rows, columns=[
            "InvoiceID", "CustomerID", "InvoiceDate", "ProductCategory",
            "ProductID", "Quantity", "UnitPrice", "DiscountPct",
            "PaymentMethod", "Returned",
        ])
        df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
        return df


async def write_clusters_to_neon(cust_df: pd.DataFrame):
    rows = cust_df[["CustomerID", "Cluster", "Persona"]].values
    values = ", ".join(
        f"('{r[0]}', {int(r[1])}, '{r[2]}')" for r in rows
    )
    sql = text(f"""
        UPDATE customer_features cf
        SET cluster = tmp.cluster,
            persona = tmp.persona,
            updated_at = NOW()
        FROM (VALUES {values}) AS tmp(customer_id, cluster, persona)
        WHERE cf.customer_id = tmp.customer_id::varchar
    """)
    async with AsyncSessionLocal() as session:
        await session.execute(sql)
        await session.commit()
        print(f"  Updated {len(cust_df)} customer records in Neon.")


async def async_main(source: str):
    logger = ExperimentLogger(name="buyer_persona_pipeline")
    print("=" * 60)
    print("BUYER PERSONA ML — FULL PIPELINE")
    print("=" * 60)
    logger.start_run()

    print(f"\n[1/7] Loading raw data (source: {source})...")
    if source == "neon":
        print("  Reading transactions from Neon...")
        async with AsyncSessionLocal() as session:
            r = await session.execute(text("SELECT count(*) FROM transactions"))
            print(f"  Neon transactions available: {r.scalar()}")
        df = await load_from_neon()
        print(f"  Loaded {len(df)} rows into DataFrame.")
    else:
        df = load_raw_data()
    logger.log_param("raw_rows", len(df))
    logger.log_param("source", source)
    print(f"  Raw: {df.shape}")

    df = clean_data(df)
    df = handle_outliers(df, ["UnitPrice", "Quantity", "DiscountPct"])
    print(f"  Cleaned: {df.shape}")
    logger.log_param("cleaned_rows", len(df))

    print("\n[2/7] Building customer features...")
    cust = build_customer_features(df)
    print(f"  Customers: {cust.shape}")
    logger.log_param("n_customers", len(cust))

    cust.to_csv(PROCESSED_FILES["features"], index=False)

    print("\n[3/7] Scaling features...")
    scale_cols = [c for c in FEATURE_COLS if c in cust.columns]
    cust_scaled, scaler = scale_features(cust.copy(), scale_cols,
                                          save_path=MODEL_FILES["scaler"])
    cust_scaled.to_csv(PROCESSED_FILES["scaled"], index=False)
    print(f"  Scaler saved: {MODEL_FILES['scaler']}")

    print("\n[4/7] Selecting features...")
    auto_drop = select_features_by_correlation(cust_scaled, scale_cols)
    keep_cols = ["Recency", "Frequency", "Monetary", "AvgBasketSize",
                 "PurchaseInterval", "WeekendRatio", "NightRatio",
                 "DiscountUsage", "ReturnRate", "ProductDiversity"]
    final_cols = [c for c in keep_cols if c in cust_scaled.columns]
    final_cols = [c for c in final_cols if c not in auto_drop or c in
                  ("Recency", "Frequency", "Monetary")]
    final_cols = sorted(set(final_cols))

    logger.log_param("auto_dropped_features", auto_drop)
    logger.log_param("selected_features", final_cols)
    print(f"  Auto-dropped: {auto_drop}")
    print(f"  Final features: {final_cols}")

    X = cust_scaled[final_cols].values
    joblib.dump(final_cols, MODEL_FILES["selected_features"])

    print("\n[5/7] PCA + clustering...")
    pca = PCA(n_components=PCA_VARIANCE_TARGET, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X)
    logger.log_param("pca_components", pca.n_components_)
    logger.log_param("pca_explained_variance",
                     sum(pca.explained_variance_ratio_))
    print(f"  PCA components: {pca.n_components_} "
          f"({sum(pca.explained_variance_ratio_):.2%} variance)")
    joblib.dump(pca, MODEL_FILES["pca"])

    print("\n[5b/7] Comparing clustering algorithms...")
    approaches = {}

    labels_pca, km_pca = kmeans_fit(X_pca, KMEANS_K)
    approaches["KMeans+PCA"] = {
        "model": km_pca, "labels": labels_pca,
        "silhouette": silhouette_score(X_pca, labels_pca),
    }

    labels_orig, km_orig = kmeans_fit(X, KMEANS_K)
    approaches["KMeans+Original"] = {
        "model": km_orig, "labels": labels_orig,
        "silhouette": silhouette_score(X, labels_orig),
    }

    best_k_tune, _, _ = kmeans_optimal_k(X, range(2, 8))
    labels_tuned, km_tuned = kmeans_fit(X, best_k_tune)
    approaches[f"KMeans(k={best_k_tune})"] = {
        "model": km_tuned, "labels": labels_tuned,
        "silhouette": silhouette_score(X, labels_tuned),
    }

    print(f"  {'Method':<25} {'Silhouette':>12}")
    print(f"  {'-'*37}")
    best_method = None
    best_sil = -1
    for name, info in sorted(approaches.items(), key=lambda x: -x[1]["silhouette"]):
        sil = info["silhouette"]
        print(f"  {name:<25} {sil:>12.4f}")
        if sil > best_sil:
            best_sil = sil
            best_method = name

    print(f"\n  Best: {best_method} (silhouette={best_sil:.4f})")
    logger.log_param("best_method", best_method)

    best_info = approaches[best_method]

    k = best_info["model"].n_clusters
    km_final = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(X)
    full_labels = km_final.predict(X)
    joblib.dump(km_final, MODEL_FILES["kmeans"])
    print(f"  Saved model: {MODEL_FILES['kmeans']}")

    print("\n[6/7] Validation...")
    val = silhouette_score(X, full_labels)
    logger.log_metric("silhouette", val)
    print(f"  Silhouette Score: {val:.4f}")

    full_pca_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    cust["Cluster"] = full_labels
    cust["PC1"] = full_pca_2d[:, 0]
    cust["PC2"] = full_pca_2d[:, 1]

    cust["Persona"] = cust["Cluster"].map(PERSONA_MAP)
    cust.to_csv(PROCESSED_FILES["personas"], index=False)
    print(f"\n  Personas saved: {PROCESSED_FILES['personas']}")
    print(f"  Distribution:\n{cust['Persona'].value_counts()}")

    if source == "neon":
        print("\n  Writing cluster assignments to Neon...")
        await write_clusters_to_neon(cust)
        logger.log_param("neon_updated", True)

    log_path = logger.save()
    logger.end_run()
    print(f"\n  Experiment log: {log_path}")
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    return cust


if __name__ == "__main__":
    import asyncio
    parser = argparse.ArgumentParser(description="Buyer Persona ML pipeline.")
    parser.add_argument("--csv", action="store_true", help="Use CSV file instead of Neon")
    args = parser.parse_args()

    try:
        asyncio.run(async_main(source="csv" if args.csv else "neon"))
    except FileNotFoundError as e:
        print(f"ERROR: File not found — {e}")
        print("Make sure data/raw/transactions.csv exists.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Pipeline failed — {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
