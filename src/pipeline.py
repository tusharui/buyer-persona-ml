"""
End-to-end pipeline: raw transactions → clustered personas + saved models.
Run: python -m src.pipeline
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import joblib

from src.config import (
    RAW_DATA_PATH, PROCESSED_FILES, MODEL_FILES,
    RANDOM_STATE, TEST_HOLDOUT_SIZE, KMEANS_K,
    PCA_VARIANCE_TARGET,
)
from src.preprocessing import (
    load_raw_data, clean_data, handle_outliers,
    scale_features, select_features_by_correlation,
)
from src.features import build_customer_features, FEATURE_COLS
from src.clustering import kmeans_fit, kmeans_optimal_k
from src.evaluation import validate_clusters, intra_inter_distances, cluster_stability_score
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
from src.tracking import ExperimentLogger

logger = ExperimentLogger(name="buyer_persona_pipeline")


def run_pipeline():
    print("=" * 60)
    print("BUYER PERSONA ML — FULL PIPELINE")
    print("=" * 60)

    # 1. Load & clean
    print("\n[1/7] Loading raw data...")
    df = load_raw_data()
    logger.log_param("raw_rows", len(df))
    print(f"  Raw: {df.shape}")

    df = clean_data(df)
    df = handle_outliers(df, ["UnitPrice", "Quantity", "DiscountPct"])
    print(f"  Cleaned: {df.shape}")
    logger.log_param("cleaned_rows", len(df))

    # 2. Feature engineering
    print("\n[2/7] Building customer features...")
    cust = build_customer_features(df)
    print(f"  Customers: {cust.shape}")
    logger.log_param("n_customers", len(cust))

    cust.to_csv(PROCESSED_FILES["features"], index=False)
    logger.log_artifact(str(PROCESSED_FILES["features"]))

    # 3. Scale
    print("\n[3/7] Scaling features...")
    scale_cols = [c for c in FEATURE_COLS if c in cust.columns]
    cust_scaled, scaler = scale_features(cust.copy(), scale_cols,
                                          save_path=MODEL_FILES["scaler"])
    cust_scaled.to_csv(PROCESSED_FILES["scaled"], index=False)
    logger.log_artifact(str(PROCESSED_FILES["scaled"]))
    logger.log_artifact(str(MODEL_FILES["scaler"]))
    print(f"  Scaler saved: {MODEL_FILES['scaler']}")

    # 4. Feature selection (automated)
    print("\n[4/7] Selecting features...")
    auto_drop = select_features_by_correlation(cust_scaled, scale_cols)
    # Always keep these core RFM + behavioral regardless
    keep_cols = ["Recency", "Frequency", "Monetary", "AvgBasketSize",
                 "PurchaseInterval", "WeekendRatio", "NightRatio",
                 "DiscountUsage", "ReturnRate", "ProductDiversity"]
    final_cols = [c for c in keep_cols if c in cust_scaled.columns]
    # Remove any that were auto-flagged for dropping, except core RFM
    final_cols = [c for c in final_cols if c not in auto_drop or c in
                  ("Recency", "Frequency", "Monetary")]
    final_cols = sorted(set(final_cols))

    logger.log_param("auto_dropped_features", auto_drop)
    logger.log_param("selected_features", final_cols)
    print(f"  Auto-dropped: {auto_drop}")
    print(f"  Final features: {final_cols}")

    X = cust_scaled[final_cols].values
    joblib.dump(final_cols, MODEL_FILES["selected_features"])
    logger.log_artifact(str(MODEL_FILES["selected_features"]))

    # 5. Hold-out split for stability validation
    print("\n[5/7] Hold-out split (stability check)...")
    X_train, X_hold = train_test_split(X, test_size=TEST_HOLDOUT_SIZE,
                                        random_state=RANDOM_STATE)
    print(f"  Train: {X_train.shape}, Hold-out: {X_hold.shape}")
    logger.log_param("holdout_size", TEST_HOLDOUT_SIZE)

    # 6. PCA
    print("\n[6/7] PCA + clustering...")
    pca = PCA(n_components=PCA_VARIANCE_TARGET, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_train)
    logger.log_param("pca_components", pca.n_components_)
    logger.log_param("pca_explained_variance",
                     sum(pca.explained_variance_ratio_))
    print(f"  PCA components: {pca.n_components_} "
          f"({sum(pca.explained_variance_ratio_):.2%} variance)")
    joblib.dump(pca, MODEL_FILES["pca"])
    logger.log_artifact(str(MODEL_FILES["pca"]))

    # Try multiple clustering approaches, pick the best
    print("\n[6b/7] Comparing clustering algorithms...")
    approaches = {}

    # 1. KMeans on PCA (default)
    labels_pca, km_pca = kmeans_fit(X_pca, KMEANS_K)
    approaches["KMeans+PCA"] = {
        "model": km_pca, "labels": labels_pca,
        "silhouette": silhouette_score(X_pca, labels_pca),
        "db": davies_bouldin_score(X_pca, labels_pca),
    }

    # 2. KMeans on original features (no PCA)
    labels_orig, km_orig = kmeans_fit(X_train, KMEANS_K)
    approaches["KMeans+Original"] = {
        "model": km_orig, "labels": labels_orig,
        "silhouette": silhouette_score(X_train, labels_orig),
        "db": davies_bouldin_score(X_train, labels_orig),
    }

    # 3. Agglomerative on PCA
    agg = AgglomerativeClustering(n_clusters=KMEANS_K)
    labels_agg = agg.fit_predict(X_pca)
    approaches["Agglomerative+PCA"] = {
        "model": agg, "labels": labels_agg,
        "silhouette": silhouette_score(X_pca, labels_agg),
        "db": davies_bouldin_score(X_pca, labels_agg),
    }

    # 4. GMM on PCA
    gmm = GaussianMixture(n_components=KMEANS_K, random_state=RANDOM_STATE)
    labels_gmm = gmm.fit_predict(X_pca)
    approaches["GMM+PCA"] = {
        "model": gmm, "labels": labels_gmm,
        "silhouette": silhouette_score(X_pca, labels_gmm),
        "db": davies_bouldin_score(X_pca, labels_gmm),
    }

    # 5. KMeans on original features with optimal k
    best_k_tune, _, _ = kmeans_optimal_k(X_train, range(2, 8))
    labels_tuned, km_tuned = kmeans_fit(X_train, best_k_tune)
    approaches[f"KMeans(k={best_k_tune})"] = {
        "model": km_tuned, "labels": labels_tuned,
        "silhouette": silhouette_score(X_train, labels_tuned),
        "db": davies_bouldin_score(X_train, labels_tuned),
    }

    # Comparison table
    print(f"  {'Method':<25} {'Silhouette':>12} {'DB Index':>10}")
    print(f"  {'-'*47}")
    best_method = None
    best_sil = -1
    for name, info in sorted(approaches.items(), key=lambda x: -x[1]["silhouette"]):
        sil = info["silhouette"]
        db = info["db"]
        print(f"  {name:<25} {sil:>12.4f} {db:>10.4f}")
        if sil > best_sil:
            best_sil = sil
            best_method = name

    print(f"\n  Best: {best_method} (silhouette={best_sil:.4f})")
    logger.log_param("best_method", best_method)

    # Use best approach for final
    best_info = approaches[best_method]
    km_model = best_info["model"]
    labels_train = best_info["labels"]

    # Re-fit on full data using best method
    if best_method == "KMeans+PCA":
        full_pca = pca.transform(X)
        full_labels = KMeans(n_clusters=KMEANS_K, random_state=RANDOM_STATE, n_init=10).fit_predict(full_pca)
    elif best_method == "KMeans+Original" or best_method.startswith("KMeans(k="):
        k = km_model.n_clusters
        full_labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
    elif best_method == "Agglomerative+PCA":
        full_pca = pca.transform(X)
        full_labels = AgglomerativeClustering(n_clusters=KMEANS_K).fit_predict(full_pca)
    else:  # GMM
        full_pca = pca.transform(X)
        full_labels = GaussianMixture(n_components=KMEANS_K, random_state=RANDOM_STATE).fit_predict(full_pca)

    logger.log_param("final_labels", best_method)
    logger.log_metric("best_silhouette", best_sil)
    print(f"  Final labels on full data: {len(set(full_labels))} clusters")

    # Save best KMeans model if applicable
    if "KMeans" in best_method:
        km_final = KMeans(n_clusters=len(set(full_labels)), random_state=RANDOM_STATE, n_init=10).fit(X if "Original" in best_method else pca.transform(X))
        joblib.dump(km_final, MODEL_FILES["kmeans"])
        print(f"  Saved model: {MODEL_FILES['kmeans']}")

    # 7. Validate
    print("\n[7/7] Validation...")
    val = silhouette_score(X if "Original" in best_method else pca.transform(X), full_labels), davies_bouldin_score(X if "Original" in best_method else pca.transform(X), full_labels)
    val = {"silhouette": val[0], "davies_bouldin": val[1]}
    logger.log_metrics(val)
    print(f"  Silhouette Score:      {val['silhouette']:.4f}")
    print(f"  Davies-Bouldin Index:  {val['davies_bouldin']:.4f}")

    intra_inter = intra_inter_distances(X_pca, labels_train)
    logger.log_metric("avg_intra_distance", np.mean(intra_inter["intra"]))
    logger.log_metric("avg_inter_distance", intra_inter["inter_mean"])

    stability = cluster_stability_score(X_pca, labels_train)
    logger.log_metrics({f"stability_{k}": v for k, v in stability.items()
                        if k != "scores"})
    print(f"  Stability (ARI):       {stability['mean_ari']:.4f} ± "
          f"{stability['std_ari']:.4f}")

    # Build full output DataFrame
    if "Original" in best_method:
        full_pca_2d = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    else:
        full_pca_2d = pca.transform(X)[:, :2]
    cust["Cluster"] = full_labels
    cust["PC1"] = full_pca_2d[:, 0]
    cust["PC2"] = full_pca_2d[:, 1]

    from src.config import PERSONA_MAP
    cust["Persona"] = cust["Cluster"].map(PERSONA_MAP)
    cust.to_csv(PROCESSED_FILES["personas"], index=False)
    logger.log_artifact(str(PROCESSED_FILES["personas"]))
    print(f"\n  Personas saved: {PROCESSED_FILES['personas']}")
    print(f"  Distribution:\n{cust['Persona'].value_counts()}")

    # Save experiment log
    log_path = logger.save()
    print(f"\n  Experiment log: {log_path}")
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)

    return cust


if __name__ == "__main__":
    try:
        run_pipeline()
    except FileNotFoundError as e:
        print(f"ERROR: File not found — {e}")
        print("Make sure data/raw/transactions.csv exists. Run the data generation script first.")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Pipeline failed — {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
