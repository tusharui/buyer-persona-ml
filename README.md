# Buyer Persona ML

> **Note:** The transaction data in this repository is **synthetically generated** for demonstration purposes. Results and persona distributions may differ with real-world data.

Unsupervised customer segmentation engine that transforms raw transaction logs into actionable business personas using RFM + behavioral features, dimensionality reduction, and clustering.

## Overview

This project implements an end-to-end ML pipeline to segment customers into behavioral personas without any labeled data. The system ingests raw invoice-level transaction data and outputs interpretable customer segments (e.g., VIP Loyal, Discount Hunters, Churn Risk, One-Time Buyers) along with business recommendations and an interactive Streamlit dashboard.

## Pipeline Architecture

```
Raw Transactions
    │
    ▼
Data Cleaning (duplicates, nulls, outliers, cancellations)
    │
    ▼
Feature Engineering (RFM + behavioral metrics)
    │
    ▼
Feature Scaling (StandardScaler / RobustScaler)
    │
    ▼
Feature Selection (correlation threshold + variance filter)
    │
    ▼
Dimensionality Reduction (PCA @ 95% variance, UMAP)
    │
    ▼
Clustering (KMeans + DBSCAN)
    │
    ▼
Validation (Silhouette, Davies-Bouldin, intra/inter distance)
    │
    ▼
Persona Assignment + Business Recommendations
    │
    ▼
Streamlit Dashboard (6 pages)
```

## Feature Engineering

The engine derives 14 customer-level features from raw transaction data:

| Category | Features |
|----------|----------|
| **RFM** | Recency (days since last purchase), Frequency (order count), Monetary (total spend) |
| **Behavioral** | Avg Basket Size, Purchase Interval, Weekend/Night purchase ratio, Discount Usage, Return Rate |
| **Diversity** | Product Diversity (unique products), Category Diversity (unique categories) |
| **Value** | Avg Quantity, Max Order Value, Total Quantity |

Feature selection removes highly correlated pairs (|r| > 0.85) and near-zero variance features, retaining 10 predictive features for clustering.

## Modeling Approach

The pipeline compares five clustering strategies and auto-selects the best by Silhouette score:

| Method | Description |
|--------|-------------|
| KMeans+PCA | KMeans on PCA-reduced features |
| KMeans+Original | KMeans on original (untransformed) features |
| Tuned KMeans (k=2..10) | Grid search over k on PCA features |
| Agglomerative+PCA | Hierarchical clustering on PCA features |
| GMM+PCA | Gaussian Mixture Model on PCA features |

### KMeans
- Optimal k determined via Elbow method + Silhouette Score (k=2..10)
- Final clusters validated with Silhouette, Davies-Bouldin, intra/inter distance, and hold-out stability (ARI)
- Centroids interpretable for persona assignment

### DBSCAN
- Eps tuned via k-distance graph (5-NN)
- Grid search over eps × min_samples
- Naturally detects noise/outliers without pre-specifying k

## Dimensionality Reduction

- **PCA**: Captures 95% explained variance (typically 4-6 components)
- **UMAP**: Non-linear 2D projection for visualization, preserves local neighborhood structure

## Results

| Cluster | Persona | Characteristics | Share |
|---------|---------|----------------|-------|
| 0 | VIP Loyal Customers | Low recency, high frequency, high monetary | ~45% |
| 1 | Discount Hunters | Low recency, low basket, high discount usage | ~11% |
| 2 | Churn Risk | High recency, low frequency, low monetary | ~28% |
| 3 | One-Time Buyers | Medium recency, very low frequency, high return rate | ~17% |

## Business Impact

Each persona is paired with targeted recommendations:

| Persona | Strategy |
|---------|----------|
| VIP Loyal | Exclusive rewards, VIP tiers, personalized concierge |
| Discount Hunters | Flash sales, coupon campaigns, bundle deals |
| Churn Risk | Win-back emails, reactivation discounts, feedback surveys |
| One-Time Buyers | Cross-selling, post-purchase nurture, retargeting ads |

## Project Structure

```
buyer-persona-ml/
├── data/
│   ├── raw/                     # Source transaction data (CSV)
│   └── processed/               # Cleaned, featured, clustered datasets
├── notebooks/                   # Jupyter notebooks (pipeline stages)
│   ├── 01_EDA                   # Exploratory data analysis
│   ├── 02_Feature_Engineering   # RFM + behavioral feature creation
│   ├── 03_PCA_UMAP              # Dimensionality reduction analysis
│   ├── 04_KMeans_DBSCAN         # Clustering + validation
│   └── 05_Persona_Insights      # Persona assignment + business interpretation
├── src/                         # Reusable Python modules
│   ├── config.py                # Paths & constants (pathlib-based)
│   ├── preprocessing.py         # Data cleaning, scaling, feature selection
│   ├── features.py              # Customer feature engineering
│   ├── clustering.py            # KMeans / DBSCAN wrappers
│   ├── evaluation.py            # Validation metrics & stability
│   ├── pipeline.py              # End-to-end pipeline orchestrator
│   ├── predict.py               # CLI inference for new transactions
│   ├── tracking.py              # Experiment logger
│   └── visualization.py         # Plotting utilities
├── dashboard/
│   └── app.py                   # Streamlit dashboard (6 pages)
├── models/                      # Serialized models (scaler, PCA, KMeans)
├── reports/                     # Generated experiment logs & reports
├── tests/
│   └── test_all.py              # 20 unit tests (pytest)
├── .gitignore
├── Dockerfile
├── Makefile
├── MODEL_CARD.md
├── requirements.txt
└── README.md
```

## Setup

```bash
# Clone and navigate
cd buyer-persona-ml

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python -m src.pipeline

# Or using Make
make all

# Run tests
make test
# or
python -m pytest tests/ -v

# Run notebooks (optional)
jupyter lab notebooks/

# Launch dashboard
streamlit run dashboard/app.py

# Predict on new transaction data
python -m src.predict --input data/raw/transactions.csv --output predictions.csv
```

## Tests

20 unit tests covering preprocessing, feature engineering, clustering, and evaluation:

```bash
python -m pytest tests/ -v
```

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Dataset Overview** | Key metrics, descriptive stats, persona distribution |
| **Feature Engineering** | Feature histograms, correlation heatmap |
| **PCA & UMAP** | 2D/3D PCA scatter, UMAP projection |
| **Clustering Results** | Validation scores, feature heatmap, cluster sizes |
| **Persona Explorer** | Drill into persona profiles, radar chart, PCA highlight |
| **Business Recommendations** | Downloadable PDF/CSV reports per persona |

## Dependencies

- Python 3.12+
- scikit-learn, pandas, numpy (core ML)
- matplotlib, seaborn (visualization)
- umap-learn (non-linear dimensionality reduction)
- streamlit (dashboard)
- pytest (testing)
- openpyxl (Excel I/O)

See [requirements.txt](requirements.txt) for exact pinned versions.

## License

MIT
