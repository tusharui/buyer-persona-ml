# Buyer Persona ML

<p align="center">
  <b>Production-grade unsupervised customer segmentation engine</b><br>
  RFM + behavioral clustering → FastAPI REST API → Streamlit dashboard → MLOps
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-blue?logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/FastAPI-0.115-009688?logo=fastapi">
  <img src="https://img.shields.io/badge/Streamlit-1.44-FF4B4B?logo=streamlit">
  <img src="https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikit-learn">
  <img src="https://img.shields.io/badge/PostgreSQL-Neon-4169E1?logo=postgresql">
  <img src="https://img.shields.io/badge/Redis-7-DC382D?logo=redis">
  <img src="https://img.shields.io/badge/MLflow-2.20-0194E2?logo=mlflow">
  <img src="https://img.shields.io/badge/Celery-5.5-37814A?logo=celery">
  <img src="https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker">
  <img src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## Elevator Pitch

**Unsupervised ML pipeline** that ingests raw transaction data and outputs **interpretable customer personas** (VIP Loyal, Discount Hunters, Churn Risk, One-Time Buyers) with targeted business recommendations. Designed for production — shipped with a FastAPI REST API (Redis-cached, async), interactive Streamlit dashboard, Celery async training, MLflow experiment tracking, automated drift detection, and full Docker Compose orchestration.

> **Data:** synthetically generated (10K transactions, 1K customers) for demonstration.

---

## Table of Contents

- [Why This Stands Out](#-why-this-stands-out)
- [System Architecture](#-system-architecture)
- [Pipeline Overview](#-pipeline-overview)
- [Components Deep Dive](#-components-deep-dive)
  - [Feature Engineering](#1-feature-engineering)
  - [Preprocessing & Dimensionality Reduction](#2-preprocessing--dimensionality-reduction)
  - [Clustering Engine](#3-clustering-engine)
  - [Validation Framework](#4-validation-framework)
  - [Persona Assignment](#5-persona-assignment--business-strategy)
  - [REST API](#6-rest-api)
  - [Dashboard](#7-dashboard)
  - [Async Task Queue](#8-async-task-queue)
  - [MLflow Tracking & Model Registry](#9-mlflow-tracking--model-registry)
  - [Drift Detection](#10-drift-detection)
  - [Caching & Feature Store](#11-caching--feature-store)
- [Database Schema](#-database-schema)
- [Infrastructure & DevOps](#-infrastructure--devops)
- [Project Structure](#-project-structure)
- [Quick Start](#-quick-start)
- [API Reference](#-api-reference)
- [Dashboard Pages](#-dashboard-pages)
- [Testing](#-testing)
- [Tech Stack](#-tech-stack)

---

## 🏆 Why This Stands Out

| Area | What This Project Demonstrates |
|------|-------------------------------|
| **ML Engineering** | End-to-end unsupervised pipeline: feature engineering, dimensionality reduction, multi-model comparison, rigorous validation |
| **MLOps** | MLflow tracking + model registry with staging/production lifecycle, data drift detection, experiment reproducibility |
| **Software Engineering** | Clean modular architecture, async Python, dependency injection, middleware, structured error handling, type hints |
| **API Design** | FastAPI with Pydantic v2 validation, Redis caching, health checks, versioned model deployments, rollback support |
| **Data Engineering** | SQLAlchemy 2.0 async ORM, Alembic migrations, Redis caching layer, batch feature store with upsert semantics |
| **DevOps** | Multi-stage Docker builds, Docker Compose (8 services), production/development profiles, Prometheus/Grafana monitoring |
| **System Design** | Distributed architecture with async workers, separate read/write paths, cache hierarchy, graceful degradation |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             CLIENT LAYER                                        │
│                                                                                 │
│   ┌──────────────────────┐   ┌──────────────────────┐   ┌───────────────────┐  │
│   │  Streamlit Dashboard │   │  API Clients / curl   │   │  CLI / Notebooks  │  │
│   │  (port 8501)         │   │  (port 8000)          │   │  (batch predict)  │  │
│   └──────────┬───────────┘   └──────────┬───────────┘   └─────────┬─────────┘  │
└───────────────┼──────────────────────────┼─────────────────────────┼────────────┘
                │                          │                         │
                ▼                          ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             SERVICE LAYER                                       │
│                                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                       FastAPI Application                               │   │
│   │                                                                         │   │
│   │   ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────┐  │   │
│   │   │ /predict │  │ /train   │  │ /models  │  │ /health  │  │ /drift │  │   │
│   │   │  (POST)  │  │  (POST)  │  │ (GET/DEL)│  │  (GET)   │  │ (GET)  │  │   │
│   │   └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬───┘  │   │
│   └────────┼──────────────┼─────────────┼──────────────┼────────────┼──────┘   │
│            │              │             │              │            │          │
│   ┌────────┴──────────────┴─────────────┴──────────────┴────────────┴──────┐   │
│   │                        Celery Worker (async)                          │   │
│   │                   train_pipeline_task (auto-retry ×2)                  │   │
│   └────────────────────────────────────────────────────────────────────────┘   │
└───────────────┼──────────────────────────┬─────────────────────────┬────────────┘
                │                          │                         │
                ▼                          ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DATA & ML LAYER                                       │
│                                                                                 │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────────┐  │
│   │  Clean +     │──▶│  Feature     │──▶│  PCA (95%)   │──▶│  Multi-Model   │  │
│   │  Scale       │   │  Engineering │   │  + UMAP      │   │  Comparison    │  │
│   │  (14→10 fea) │   │  (RFM + beh) │   │              │   │  KMeans/GMM/  │  │
│   └──────────────┘   └──────────────┘   └──────────────┘   │  Agglo/DBSCAN  │  │
│                                                             └───────┬────────┘  │
│   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐          │           │
│   │  Drift       │   │  Feature     │   │  Model       │          │           │
│   │  Detector    │   │  Store       │   │  Registry    │◀─────────┘           │
│   │  (PSI/KS/KL) │   │  (Neon+Redis)│   │  (MLflow)    │                      │
│   └──────────────┘   └──────────────┘   └──────────────┘                      │
└───────────────┼──────────────────────────┬─────────────────────────┬────────────┘
                │                          │                         │
                ▼                          ▼                         ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          PERSISTENCE LAYER                                      │
│                                                                                 │
│   ┌────────────────────┐   ┌────────────────────┐   ┌──────────────────────┐   │
│   │  Neon (PostgreSQL) │   │  Redis (Cache +    │   │  MLflow Tracking     │   │
│   │  • transactions    │   │       Broker)      │   │  Server              │   │
│   │  • customers       │   │  • Predict cache   │   │  (port 5000)         │   │
│   │  • customer_feat.  │   │  • Feature cache   │   │  • experiments       │   │
│   │  (Alembic managed) │   │  • Model versions  │   │  • model registry    │   │
│   └────────────────────┘   │  • Celery backend  │   └──────────────────────┘   │
│                            └────────────────────┘                              │
│   ┌────────────────────┐   ┌────────────────────┐                              │
│   │  Local Filesystem  │   │  Prometheus +      │                              │
│   │  • models/*.pkl    │   │  Grafana           │                              │
│   │  • data/processed  │   │  (monitoring)      │                              │
│   │  • reports/        │   └────────────────────┘                              │
│   └────────────────────┘                                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Pipeline Overview

```
 [Raw Transactions]             [Feature Store]
       │                              │
       ▼                              ▼
 ┌─────────────┐             ┌──────────────┐
 │ Clean Data  │────────────▶│  ML Pipeline  │
 │             │  CSV/Neon   │              │
 └─────────────┘             │ 1. Load      │
                             │ 2. Feature   │
 ┌─────────────┐             │ 3. Scale     │
 │ Train Model │◀────────────│ 4. PCA       │
 │ (Celery)    │  async      │ 5. Cluster   │
 └─────────────┘             │ 6. Validate  │
        │                    │ 7. Assign    │
        ▼                    │ 8. Save      │
 ┌─────────────┐             └──────┬───────┘
 │ MLflow Log  │                    │
 │ (params +   │                    ▼
 │  metrics +  │             ┌──────────────┐
 │  artifacts) │             │ Model Export │
 └─────────────┘             │ scaler.pkl   │
                             │ pca.pkl      │
 ┌─────────────┐             │ kmeans.pkl   │
 │ Model       │◀────────────│ features.pkl │
 │ Registry    │  promote    └──────────────┘
 └─────────────┘
        │
        ▼
 ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
 │ FastAPI      │    │ Streamlit    │    │ Batch CLI   │
 │ /predict     │    │ Dashboard    │    │ predict.py  │
 │ (Redis cache)│    │ (6 pages)    │    │             │
 └──────────────┘    └──────────────┘    └─────────────┘
```

---

## 🔬 Components Deep Dive

### 1. Feature Engineering

**`src/features.py`** — Transforms raw invoice-level transactions into **14 customer-level features** spanning four behavioral dimensions:

| Category | Features | Business Signal |
|----------|----------|-----------------|
| **RFM** | Recency, Frequency, Monetary | Core customer value (who are your best customers?) |
| **Engagement** | AvgBasketSize, PurchaseInterval, TotalQty | Purchase depth and cadence |
| **Behavior** | WeekendRatio, NightRatio, DiscountUsage, ReturnRate | Shopping preferences and price sensitivity |
| **Diversity** | ProductDiversity, CategoryDiversity, AvgQuantity, MaxOrderValue | Breadth of appetite and spending ceiling |

Each feature is computed via vectorized `groupby().agg()` operations — no loops, production-ready performance.

---

### 2. Preprocessing & Dimensionality Reduction

**`src/preprocessing.py`** — Four-stage pipeline:

| Stage | Technique | Why |
|-------|-----------|-----|
| **Clean** | Drop duplicates, cancelled orders, null customers, negative quantities | Data quality gates |
| **Outliers** | IQR-based capping (1.5×IQR) on UnitPrice, Quantity, DiscountPct | Robust to extreme values |
| **Scale** | StandardScaler (z-score) — saved to `models/scaler.pkl` | Required for distance-based clustering |
| **Select** | Drop correlated pairs (\|r\| > 0.85, keeps higher-variance feature) + variance threshold (0.01) | Reduces noise, retains 10 predictive features |

**Dimensionality Reduction:**
- **PCA** — Captures 95% explained variance (auto-determines 4–6 components). Saved to `models/pca.pkl`.
- **UMAP** — Non-linear 2D projection for dashboard visualization (on-demand).

---

### 3. Clustering Engine

**`src/clustering.py`** — The pipeline compares **five strategies** head-to-head and auto-selects the winner by Silhouette score:

| Strategy | Space | k | Why Include It |
|----------|-------|---|----------------|
| **KMeans+PCA** | PCA-reduced | 4 | Fast, interpretable centroids |
| **KMeans+Original** | Scaled features | 4 | Baseline on original space |
| **Tuned KMeans** | Scaled features | 2–10 (grid search) | Finds optimal k automatically |
| **Agglomerative+PCA** | PCA-reduced | 4 | Hierarchical alternative |
| **GMM+PCA** | PCA-reduced | 4 | Soft clustering, probabilistic |

Additionally, **DBSCAN** is available for density-based clustering with eps tuning via k-distance graph.

**Winner auto-selection:** The method with the highest Silhouette score is used for the final model, ensuring optimal results without manual intervention.

---

### 4. Validation Framework

**`src/evaluation.py`** — Four-metric validation suite that goes beyond simple scoring:

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Silhouette Score** | Cluster cohesion vs. separation (–1 to 1) | Primary selection criterion |
| **Davies-Bouldin Index** | Ratio of intra to inter-cluster distances (lower = better) | Secondary validation |
| **Intra/Inter Distance** | Avg distance within clusters vs. between clusters | Interpretability check |
| **Stability (ARI)** | Adjusted Rand Index across 5 random 80% subsamples | Reproducibility guarantee |

The stability score (ARI) answers the critical question: *"If we re-ran this on slightly different data, would we get the same segments?"*

---

### 5. Persona Assignment & Business Strategy

Each cluster is mapped to a named persona with an actionable business strategy:

| Cluster | Persona | Characteristics | Strategy |
|---------|---------|-----------------|----------|
| 0 | 🏆 VIP Loyal Customers | Low recency, high frequency, high monetary | Exclusive rewards, VIP tiers, personalized concierge |
| 1 | 💰 Discount Hunters | Low recency, low basket, high discount usage | Flash sales, coupon campaigns, bundle deals |
| 2 | ⚠️ Churn Risk | High recency, low frequency, low monetary | Win-back emails, reactivation discounts, feedback surveys |
| 3 | 🆕 One-Time Buyers | Medium recency, very low frequency, high return rate | Cross-selling, post-purchase nurture, retargeting ads |

Business recommendations are centralized in `src/config.py` and surfaced in the dashboard for stakeholder consumption.

---

### 6. REST API

**`api/`** — Production-grade FastAPI application with automatic OpenAPI documentation at `/docs`.

**Endpoints:**

| Method | Path | Description | Cache |
|--------|------|-------------|-------|
| `GET` | `/health` | DB, Redis, model status | No |
| `POST` | `/predict` | Predict personas from transaction batch | Redis (24h, keyed by feature hash) |
| `POST` | `/train` | Queue async Celery training job | No |
| `GET` | `/train/{task_id}` | Poll training task status | No |
| `GET` | `/models` | List model versions | No |
| `POST` | `/models/deploy` | Deploy specific model version | Redis |
| `POST` | `/models/rollback` | Rollback to previous version | Redis |
| `GET` | `/health/drift` | Feature drift status vs. baseline | No |

**Engineering highlights:**
- **Lifespan management**: Models loaded once at startup via `ModelLoader` singleton; graceful shutdown with connection cleanup
- **Redis caching**: Prediction responses cached by SHA-256 feature hash — cache hit avoids full inference pipeline
- **Middleware**: Request logging with UUID + millisecond timing on every request
- **Exception handling**: Structured JSON error responses for HTTP, validation, and 500 errors
- **Graceful degradation**: Health endpoint reflects degraded state if DB/Redis/model is unavailable

---

### 7. Dashboard

**`dashboard/app.py`** — Streamlit dashboard (6 pages) with dual data source support:

| Page | What It Shows |
|------|---------------|
| **Dataset Overview** | KPI cards, sample data, descriptive stats, persona distribution |
| **Feature Engineering** | Histograms (2×4 grid), correlation heatmap |
| **PCA & UMAP** | 2D scatter, interactive 3D scatter, on-demand UMAP |
| **Clustering Results** | Silhouette + Davies-Bouldin scores, feature heatmap, cluster sizes |
| **Persona Explorer** | Per-persona feature profiles, radar chart, PCA-highlight view |
| **Business Recommendations** | Expandable persona cards with descriptions + strategies, downloadable reports |

**Caching layer:** `dashboard/cache.py` provides a Redis-backed cache for dataframes and matplotlib figures, keyed by SHA-256 hash of query parameters.

---

### 8. Async Task Queue

**`src/celery_app.py`** + **`src/tasks.py`** — Celery distributed task queue with Redis as broker/backend:

- **Task**: `train_pipeline_task` — wraps the full async ML pipeline (`async_main`) in a Celery task
- **Resilience**: `autoretry_for=(Exception,), max_retries=2`
- **Monitoring**: Task status pollable via `GET /train/{task_id}`
- **Return value**: JSON with status, customer count, cluster count, or error traceback

---

### 9. MLflow Tracking & Model Registry

**`src/tracking.py`** + **`src/model_registry.py`** — Full experiment lifecycle management.

**Every pipeline run tracks:**
- **Parameters:** raw/cleaned row counts, PCA components, selected features, best method, source
- **Metrics:** Silhouette, Davies-Bouldin, intra/inter distances, stability ARI (mean ± std)
- **Artifacts:** All serialized models + final personas CSV

**Model Registry lifecycle:**

```
register_model(run_id) ──▶ Staging ──▶ Production ──▶ Archived
                                 │
                                 └── Rollback (fallback)
```

Active version synced to Redis — the API reads the active version at startup for transparent rollouts.

---

### 10. Drift Detection

**`src/drift_detector.py`** — Automated feature drift monitoring against training baseline:

| Test | What It Detects | Threshold |
|------|-----------------|-----------|
| **PSI** (Population Stability Index) | Distribution shift magnitude | > 0.25 |
| **KL Divergence** | Information loss between distributions | > 0.1 |
| **KS Test** | Two-sample Kolmogorov-Smirnov | p < 0.05 |

Drift is detected **per feature** and reported as a ratio of drifted features over total checked. Exposed via `GET /health/drift` for integration with monitoring alerts.

---

### 11. Caching & Feature Store

**Cache layers (Redis):**

| Cache | Key Pattern | TTL | Purpose |
|-------|-------------|-----|---------|
| Prediction | `predict:{cid}:{feat_hash}` | 24h | Avoid re-inference for same features |
| Feature | `features:{customer_id}` | 1h | Feature vector lookup |
| Model Active | `model:active` | ∞ | Active deployment pointer |
| Dashboard | `dashboard:{prefix}:{hash}` | 1h | Cached figures + dataframes |
| Celery | Standard | Configurable | Task broker + result backend |

**Feature Store (`src/feature_store.py`):**
1. Loads all transactions from Neon
2. Computes feature vectors (upsert semantics — insert new, update existing)
3. Populates Redis cache for fast API lookups

---

## 🗄 Database Schema

Managed via **Alembic** (`alembic/versions/613d0d2bf62a_initial_schema.py`):

```
┌───────────────────┐       ┌──────────────────────────┐
│    customers      │       │      transactions        │
├───────────────────┤       ├──────────────────────────┤
│ PK customer_id    │◀──────│ FK customer_id           │
│    created_at     │  1:N  │    invoice_id (INDEX)    │
└───────────────────┘       │    invoice_date (INDEX)  │
                            │    product_category      │
┌───────────────────┐       │    product_id            │
│ customer_features │       │    quantity              │
├───────────────────┤       │    unit_price            │
│ PK id             │       │    discount_pct          │
│ FK customer_id    │◀──────│    payment_method        │
│    (UNIQUE,INDEX) │  1:1  │    returned              │
│    recency        │       └──────────────────────────┘
│    frequency      │
│    monetary       │       14 feature columns  + cluster + persona
│    ...            │       `customer_features` stores the computed
│    cluster        │       feature vectors + pipeline output
│    persona        │
│    updated_at     │
└───────────────────┘
```

---

## ☁ Infrastructure & DevOps

### Docker Compose Services (dev)

| Service | Technology | Purpose | Dependencies |
|---------|-----------|---------|--------------|
| `redis` | redis:7-alpine | Cache + Celery broker | — |
| `postgres` | postgres:16-alpine | Primary database | — |
| `mlflow` | Custom | Experiment tracking | — |
| `api` | FastAPI + Uvicorn | REST API | redis, postgres |
| `dashboard` | Streamlit | UI | api |
| `worker` | Celery | Async training | redis, postgres, mlflow |
| `prometheus` | prom/prometheus | Metrics collection | — |
| `grafana` | grafana/grafana | Monitoring dashboards | prometheus |

### Dockerfile (multi-stage)

```
FROM python:3.12-slim AS base     # pip install + source copy
       ├── api                    # uvicorn api.main:app
       ├── dashboard              # streamlit run dashboard/app.py
       └── worker                 # celery -A src.celery_app worker
```

### Production Profile

`docker-compose.prod.yml` strips local Postgres (uses external Neon `DATABASE_URL`) and removes Prometheus/Grafana for a lean production deployment.

---

## 📁 Project Structure

```
buyer-persona-ml/
│
├── api/                          # FastAPI REST API
│   ├── main.py                   # App factory, lifespan, error handlers
│   ├── schemas.py                # Pydantic v2 models
│   ├── dependencies.py           # ModelLoader singleton
│   ├── middleware.py             # Request logging with UUID + timing
│   ├── exception_handlers.py     # Structured JSON errors
│   └── routes/
│       ├── predict.py            # POST /predict (Redis-cached)
│       ├── training.py           # POST /train + GET /train/{id}
│       ├── models.py             # Model version management
│       └── drift.py              # GET /health/drift
│
├── dashboard/
│   ├── app.py                    # Streamlit (6 pages)
│   └── cache.py                  # Redis-backed figure/DF cache
│
├── src/                          # Core ML + infrastructure
│   ├── config.py                 # Paths, constants, persona definitions
│   ├── database.py               # Async SQLAlchemy engine (asyncpg)
│   ├── models.py                 # ORM: Customer, Transaction, CustomerFeature
│   ├── preprocessing.py          # Load → clean → scale → select
│   ├── features.py               # 14 RFM + behavioral features
│   ├── clustering.py             # KMeans, DBSCAN, k-distance tuning
│   ├── evaluation.py             # Silhouette, DB, intra/inter, stability
│   ├── pipeline.py               # Async end-to-end orchestrator
│   ├── predict.py                # CLI batch inference
│   ├── tracking.py               # MLflow experiment logger
│   ├── model_registry.py         # Model version lifecycle
│   ├── feature_store.py          # Batch compute + store features
│   ├── drift_detector.py         # PSI, KS, KL drift detection
│   ├── cache.py                  # Redis async client
│   ├── celery_app.py             # Celery configuration
│   ├── tasks.py                  # Async training task
│   ├── data_generator.py         # Synthetic data → Neon
│   └── visualization.py          # PCA scatter, heatmap
│
├── models/                       # Serialized artifacts
│   ├── MODEL_CARD.md             # Model documentation
│   ├── scaler.pkl                # StandardScaler
│   ├── pca.pkl                   # PCA transformer
│   ├── kmeans.pkl                # KMeans model
│   └── selected_features.pkl     # Feature list
│
├── data/
│   ├── raw/                      # Source CSV
│   └── processed/                # Generated datasets
│
├── alembic/                      # Database migrations
│   ├── env.py
│   └── versions/
│       └── 613d0d2bf62a_initial_schema.py
│
├── notebooks/                    # Jupyter notebooks (5 stages)
├── tests/
│   └── test_all.py               # 20 unit tests (pytest)
├── reports/                      # Experiment logs
│
├── docker-compose.yml            # Full dev stack (8 services)
├── docker-compose.prod.yml       # Production stack
├── Dockerfile                    # Multi-stage build
├── Makefile                      # 15+ convenience targets
├── .env.example                  # Env template
├── requirements.txt              # Pinned dependencies
├── alembic.ini
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone & install
cd buyer-persona-ml
pip install -r requirements.txt

# 2. Run the full pipeline (CSV mode)
python -m src.pipeline --csv

# 3. Launch dashboard
streamlit run dashboard/app.py

# 4. Start API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs

# 5. Batch predict
python -m src.predict --input data/raw/transactions.csv --output predictions.csv

# 6. Full Docker stack
docker compose up --build -d

# 7. Run tests
python -m pytest tests/ -v
```

### Makefile Quick Reference

```bash
make pipeline       # Full ML pipeline
make test           # Run tests
make run            # Launch dashboard
make run-api        # Start API server
make generate-data  # Generate synthetic data
make feature-store  # Compute + store features
make run-worker     # Start Celery worker
make docker-up      # Full stack up
make docker-down    # Full stack down
```

---

## 📋 API Reference

### Health Check
```
GET /health
→ { "status": "healthy", "database": "connected", "redis": "connected", "model": "loaded", "model_version": "kmeans_k4" }
```

### Predict Personas
```
POST /predict
Body: { "transactions": [{ "invoice_id", "customer_id", "invoice_date", "product_category", ... }] }
→ { "predictions": [{ "customer_id": "C1001", "cluster": 0, "persona": "VIP Loyal Customers" }], "model_version": "kmeans_k4" }
```

### Async Training
```
POST /train?source=neon
→ { "task_id": "uuid", "status": "queued", "message": "Training job queued successfully." }

GET /train/{task_id}
→ { "task_id": "uuid", "status": "SUCCESS", "message": "{'customers': 1000, 'clusters': 4}" }
```

### Model Management
```
GET /models
→ [{ "version": "kmeans_k4", "status": "production" }]

POST /models/deploy?version=v2
→ { "version": "v2", "status": "production" }

POST /models/rollback
→ { "version": "v1", "status": "production" }
```

### Drift Detection
```
GET /health/drift
→ { "status": "healthy", "drift_detected": false, "features_checked": 10, "features_with_drift": 0, ... }
```

Full interactive documentation available at `http://localhost:8000/docs` (Swagger UI) and `/redoc` (ReDoc).

---

## 📊 Dashboard Pages

| Page | Description |
|------|-------------|
| **Dataset Overview** | KPI cards, sample data, descriptive statistics, persona distribution bar chart |
| **Feature Engineering** | Per-feature histograms, interactive correlation heatmap |
| **PCA & UMAP** | 2D PCA scatter, interactive 3D PCA, on-demand UMAP projection |
| **Clustering Results** | Silhouette score, Davies-Bouldin index, feature heatmap per cluster, cluster sizes |
| **Persona Explorer** | Drill into any persona: feature profiles, radar chart comparison, PCA highlight |
| **Business Recommendations** | Expandable persona cards with descriptions + strategies, downloadable reports |

---

## ✅ Testing

**20 unit tests** covering every stage of the pipeline:

```
src/preprocessing   → CleanData, HandleOutliers, ScaleFeatures, FeatureSelection
src/features        → BuildCustomerFeatures (RFM columns, behavioral columns, shapes)
src/clustering      → KMeans optimal k, KMeans fit, DBSCAN fit
src/evaluation      → Silhouette + DB, intra/inter distances, stability ARI
src/model_registry  → Instantiation, MLflow fallback
```

```bash
python -m pytest tests/ -v
```

---

## 🛠 Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.12+ |
| **ML & Data** | scikit-learn, pandas, numpy, umap-learn |
| **Framework** | FastAPI, Uvicorn, Pydantic v2 |
| **Database** | PostgreSQL (Neon), SQLAlchemy 2.0 (async), asyncpg, Alembic |
| **Caching & Queue** | Redis (async), Celery |
| **Dashboard** | Streamlit, matplotlib, seaborn |
| **MLOps** | MLflow (tracking + model registry) |
| **Infrastructure** | Docker, Docker Compose, Prometheus, Grafana |
| **Testing** | pytest |
| **Other** | python-dotenv, joblib, openpyxl |

---

## 📄 License

MIT
