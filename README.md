# Buyer Persona ML

<p align="center">
  <b>AI-Powered Customer Segmentation Engine</b><br>
  Unsupervised ML → SHAP Explainability → Churn Prediction → Anomaly Detection →<br>
  LLM Narratives → RAG Chatbot → Time-Series Forecast → Kafka Streaming → Full MLOps
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
  <img src="https://img.shields.io/badge/Groq-LLM-10a37f?logo=groq">
  <img src="https://img.shields.io/badge/Apache_Kafka-2.8-231F20?logo=apache-kafka">
  <img src="https://img.shields.io/badge/license-MIT-green">
</p>

---

A production-grade unsupervised customer segmentation engine that ingests raw transaction data and outputs interpretable customer personas (VIP Loyal, Discount Hunters, Churn Risk, One-Time Buyers) with targeted business recommendations. The system spans the full ML lifecycle — from feature engineering and clustering to SHAP explainability, churn prediction, anomaly detection, time-series forecasting, LLM-powered narratives, RAG chatbot, Kafka streaming, and production MLOps.

> **Data:** 10K synthetic transactions, 1K customers for demonstration.

---

## Capability Overview

| Area | Features |
|------|----------|
| **Unsupervised ML** | RFM + behavioral feature engineering, multi-model comparison (KMeans, GMM, Agglomerative, DBSCAN), PCA/UMAP dimensionality reduction |
| **Model Interpretability** | SHAP KernelExplainer per-prediction feature attribution via `POST /predict/explain` |
| **Supervised Learning** | RandomForest churn classifier trained on persona features, probability + risk scoring |
| **Anomaly Detection** | IsolationForest for outlier customer detection with reconstruction error scoring |
| **Hyperparameter Optimization** | Optuna Bayesian TPE sampling for KMeans k, PCA components, DBSCAN eps/min_samples |
| **Time-Series Forecasting** | Prophet per-segment revenue forecasting with confidence intervals (fallback LinearRegression) |
| **LLM Integration** | Groq-powered persona narrative generator via `POST /persona/narrate` (llama-3.3-70b) |
| **RAG Chatbot** | LangChain retrieval chain over business docs + Chroma vector store, `POST /chat` |
| **Streaming Pipeline** | Async Kafka producer/consumer (aiokafka) for real-time transaction ingestion |
| **MLOps** | MLflow experiment tracking + model registry (Staging/Production/Archived), automated drift detection (PSI, KL, KS) |
| **API Design** | FastAPI with Pydantic v2, Redis caching, middleware, structured error handling, graceful degradation |
| **Data Engineering** | Async SQLAlchemy 2.0 ORM, Alembic migrations, feature store with upsert, Redis cache hierarchy |
| **DevOps** | Multi-stage Docker builds, Docker Compose (8 services), Prometheus/Grafana, CI/CD (GitHub Actions) |
| **System Design** | Distributed architecture: async workers, Celery task queue, separate read/write paths, cache hierarchy |

---

## 🏗 System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                      CLIENT LAYER                                            │
│                                                                                              │
│   ┌──────────────────────┐   ┌──────────────────────┐   ┌──────────────────┐   ┌─────────┐  │
│   │  Streamlit Dashboard │   │  API Clients / curl   │   │  CLI / Notebooks │   │  Kafka  │  │
│   │  (port 8501)         │   │  (port 8000)          │   │  (batch predict) │   │ Clients │  │
│   └──────────┬───────────┘   └──────────┬───────────┘   └────────┬─────────┘   └────┬────┘  │
└───────────────┼──────────────────────────┼─────────────────────────┼─────────────────┼───────┘
                │                          │                         │                 │
                ▼                          ▼                         ▼                 ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                   SERVICE LAYER                                              │
│                                                                                              │
│                          ┌─────────────────────────────────────┐                            │
│                          │         FastAPI Application          │                            │
│                          │                                       │                            │
│   ┌─────────┐  ┌──────────┐  ┌────────┐  ┌─────────┐  ┌──────┐  │  ┌──────────────────┐    │
│   │ /predict│  │ /predict │  │/predict│  │/anomalies│  │/chat │  │  │   Celery Worker  │    │
│   │(persona)│  │ /explain │  │ /churn │  │          │  │(RAG) │  │  │ (async training) │    │
│   └────┬────┘  └────┬─────┘  └───┬────┘  └────┬─────┘  └──┬───┘  │  └────────┬─────────┘    │
│   ┌────┴────┐  ┌────┴─────┐  ┌───┴────┐  ┌────┴─────┐  ┌──┴───┐  │           │              │
│   │/forecast│  │/train    │  │/models │  │/persona │  │/stream│  │           │              │
│   │         │  │(Celery)  │  │(deploy)│  │/narrate │  │(Kafka)│  │           │              │
│   └────┬────┘  └────┬─────┘  └────────┘  └─────────┘  └──────┘  │           │              │
│        │            │                                             │           │              │
│        └────────────┴─────────────────────────────────────────────┘           │              │
└───────────────────────────────┼───────────────────────────────────────────────┼──────────────┘
                                │                                               │
                                ▼                                               ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  DATA & ML LAYER                                            │
│                                                                                              │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌────────────┐   ┌──────────┐   ┌─────────┐ │
│   │  Clean + │──▶│ Feature  │──▶│ PCA (95%)│──▶│ Multi-Model│──▶│  SHAP    │──▶│  Churn  │ │
│   │  Scale   │   │ Engineer │   │ + UMAP   │   │ Comparison │   │ Explainer│   │ Predict │ │
│   └──────────┘   └──────────┘   └──────────┘   └─────┬──────┘   └──────────┘   └─────────┘ │
│                                                       │                                     │
│   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌─────┴──────┐   ┌──────────┐   ┌─────────┐ │
│   │ Anomaly  │   │ Forecast │   │  LLM     │   │   RAG      │   │  Drift   │   │  Optuna │ │
│   │ Detector │   │ (Prophet)│   │ Narrative│   │  Chatbot   │   │ Detection│   │  Tuning │ │
│   │(Isolation│   │          │   │ (Groq)   │   │ (ChromaDB) │   │(PSI/KL) │   │         │ │
│   │ Forest)  │   │          │   │          │   │            │   │          │   │         │ │
│   └──────────┘   └──────────┘   └──────────┘   └────────────┘   └──────────┘   └─────────┘ │
│                                                                                              │
│   ┌──────────────────────────────────────────────────────────────────────────────────┐      │
│   │                            Model Registry (MLflow)                                │      │
│   │            register() → Staging → Production → Archived → Rollback               │      │
│   └──────────────────────────────────────────────────────────────────────────────────┘      │
└───────────────────────────────┼───────────────────────────────────────────────────────┼──────┘
                                │                                                       │
                                ▼                                                       ▼
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                                PERSISTENCE LAYER                                            │
│                                                                                              │
│   ┌──────────────────────┐   ┌──────────────────────────┐   ┌───────────────────────────┐   │
│   │  Neon (PostgreSQL)   │   │  Redis (Cache + Broker)  │   │  MLflow Tracking Server   │   │
│   │  • transactions      │   │  • Predict cache (24h)   │   │  • Experiment tracking    │   │
│   │  • customers         │   │  • Feature cache (1h)    │   │  • Model registry         │   │
│   │  • customer_features │   │  • Active model pointer  │   │  • Artifact storage       │   │
│   │  (Alembic migrations)│   │  • Dashboard (figures)   │   └───────────────────────────┘   │
│   └──────────────────────┘   │  • Celery backend        │   ┌───────────────────────────┐   │
│                               └──────────────────────────┘   │  ChromaDB (RAG)           │   │
│   ┌──────────────────────┐   ┌──────────────────────────┐   │  • Persona docs indexed   │   │
│   │  Kafka (Streaming)   │   │  Prometheus + Grafana    │   │  • Semantic search        │   │
│   │  • Transactions topic│   │  • Custom ML metrics     │   └───────────────────────────┘   │
│   │  • Real-time consumer│   │  • Model KPIs dashboard  │                                  │
│   └──────────────────────┘   └──────────────────────────┘                                  │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                               ML PIPELINE FLOW                                           │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│  [Raw CSV] ──▶ [Clean & Scale] ──▶ [14 RFM+Behavioral Features] ──▶ [Feature Selection] │
│       │                              │                              │                   │
│       ▼                              ▼                              ▼                   │
│  [Neon DB] ────▶ [Feature Store] ──▶ [PCA (95% var)] ──▶ [Clustering Comparison]        │
│                                      │                     │                            │
│                                      ▼                     ▼                            │
│                               [Optuna Tune] ◀── [Validation Suite]                       │
│                                      │                     │                            │
│                                      ▼                     ▼                            │
│                          [Best Model → Register] ◀── [SHAP Explainer] ──▶ [Churn Model] │
│                                      │                     │              │              │
│                                      ▼                     ▼              ▼              │
│                          [Personas CSV] ──▶ [Forecast]  [Anomaly]     [RAG Index]        │
│                                      │                     │              │              │
│                                      ▼                     ▼              ▼              │
│                          [LLM Narratives]  [Streaming]    [Drift Monitor]                │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
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

All features computed via vectorized `groupby().agg()` — zero loops, production-ready.

### 2. Preprocessing & Dimensionality Reduction

**`src/preprocessing.py`** — Four-stage pipeline with robust data quality gates:

| Stage | Technique | Why |
|-------|-----------|-----|
| **Clean** | Drop duplicates, cancelled orders, null customers, negative quantities | Data quality gates |
| **Outliers** | IQR-based capping (1.5×IQR) on UnitPrice, Quantity, DiscountPct | Robust to extreme values |
| **Scale** | StandardScaler (z-score) — saved to `models/scaler.pkl` | Required for distance-based clustering |
| **Select** | Drop correlated pairs (\|r\| > 0.85, keeps higher-variance feature) + variance threshold (0.01) | Reduces noise, retains 10 predictive features |

**Dimensionality Reduction:**
- **PCA** — Captures 95% explained variance (auto-determines 4–6 components)
- **UMAP** — Non-linear 2D projection for dashboard visualization (on-demand)

### 3. Clustering Engine

**`src/clustering.py`** — Five strategies compared head-to-head, auto-selected by Silhouette score:

| Strategy | Space | k | Why Include It |
|----------|-------|---|----------------|
| **KMeans+PCA** | PCA-reduced | 4 | Fast, interpretable centroids |
| **KMeans+Original** | Scaled features | 4 | Baseline on original space |
| **Tuned KMeans** | Scaled features | 2–10 (grid search) | Finds optimal k automatically |
| **Agglomerative+PCA** | PCA-reduced | 4 | Hierarchical alternative |
| **GMM+PCA** | PCA-reduced | 4 | Soft clustering, probabilistic |

**DBSCAN** available for density-based clustering with k-distance graph eps tuning.

### 4. Optuna Hyperparameter Optimization

**`src/tuning.py`** — Bayesian hyperparameter optimization replacing manual grid search:

| Optimizer | Parameters Searched | Search Space |
|-----------|-------------------|--------------|
| `tune_kmeans_optuna` | n_clusters, init | 2–10, {k-means++, random} |
| `tune_pca_optuna` | n_components, k (joint) | 2–10, 2–8 |
| `tune_dbscan_optuna` | eps, min_samples | 0.1–3.0, 2–20 |

Uses Optuna's TPE (Tree-structured Parzen Estimator) sampler with 50 trials — 10x more efficient than grid search.

### 5. Validation Framework

**`src/evaluation.py`** — Four-metric validation suite:

| Metric | What It Measures | Why It Matters |
|--------|-----------------|----------------|
| **Silhouette Score** | Cluster cohesion vs. separation (–1 to 1) | Primary selection criterion |
| **Davies-Bouldin Index** | Ratio of intra to inter-cluster distances (lower = better) | Secondary validation |
| **Intra/Inter Distance** | Avg distance within clusters vs. between clusters | Interpretability check |
| **Stability (ARI)** | Adjusted Rand Index across 5 random 80% subsamples | Reproducibility guarantee |

### 6. Persona Assignment & Business Strategy

| Cluster | Persona | Characteristics | Strategy |
|---------|---------|-----------------|----------|
| 0 | 🏆 VIP Loyal Customers | Low recency, high frequency, high monetary | Exclusive rewards, VIP tiers, personalized concierge |
| 1 | 💰 Discount Hunters | Low recency, low basket, high discount usage | Flash sales, coupon campaigns, bundle deals |
| 2 | ⚠️ Churn Risk | High recency, low frequency, low monetary | Win-back emails, reactivation discounts, feedback surveys |
| 3 | 🆕 One-Time Buyers | Medium recency, very low frequency, high return rate | Cross-selling, post-purchase nurture, retargeting ads |

### 7. SHAP Explainability API

**`src/explainer.py`** + **`POST /predict/explain`** — Model interpretability powered by SHAP:

- Wraps the full pipeline (scaler → PCA → KMeans) as a single prediction function
- Uses `KernelExplainer` with background samples for model-agnostic explanations
- Returns per-feature attribution values with direction (positive/negative contribution)
- Enables answers to: *"Why was this customer labeled as Churn Risk?"*

```json
{
  "customer_id": "C1042",
  "persona": "Churn Risk",
  "base_value": -0.45,
  "contributions": [
    {"feature": "Recency", "importance": 0.32, "direction": "positive"},
    {"feature": "Frequency", "importance": 0.28, "direction": "negative"}
  ]
}
```

### 8. Supervised Churn Predictor

**`src/churn.py`** + **`POST /predict/churn`** — Binary classifier stacked on persona features:

- **Model:** RandomForestClassifier (200 estimators, max_depth=8, balanced class weights)
- **Labeling:** Heuristic — customers above 75th percentile Recency AND below 25th percentile Frequency
- **Outputs:** Churn probability (0–1), risk tier (Low/Medium/High), top 3 contributing factors
- **Training:** Hold-out split (30%), evaluated on ROC-AUC and average precision

### 9. Anomaly Detection

**`src/anomaly_detector.py`** + **`GET /anomalies`** — Unsupervised outlier detection:

- **Algorithm:** IsolationForest (200 estimators, 5% contamination)
- **Output:** Per-customer anomaly score, binary flag, reconstruction error
- **Uses:** Flag data entry errors, fraudulent accounts, unusual purchase patterns
- **Training:** On-demand via `POST /anomalies/train`

### 10. Time-Series Forecasting

**`src/forecast.py`** + **`GET /forecast`** — Revenue forecasting per customer segment:

- **Primary model:** Facebook Prophet with multiplicative seasonality, yearly seasonality
- **Fallback:** LinearRegression with 95% confidence intervals (when Prophet unavailable)
- **Granularity:** Monthly aggregation, 3-month forecast horizon
- **Caching:** Forecast results stored as JSON per persona, refreshed on-demand via `POST /forecast/refresh`

### 11. LLM Persona Narrative Generator

**`src/llm.py`** + **`POST /persona/narrate`** — AI-generated marketing persona descriptions:

- **Provider:** Groq API (OpenAI-compatible) — uses `llama-3.3-70b-versatile`
- **System prompt:** "Senior marketing analyst AI" generating 2-3 paragraph narratives
- **Input:** Persona name + optional behavioral profile (feature averages)
- **Output:** Rich narrative covering demographics, purchase drivers, channel preferences, LTV potential
- **Fallback:** Rule-based template when API key is absent

### 12. RAG Chatbot for Business Insights

**`src/rag.py`** + **`POST /chat`** — Natural language query over business documents:

- **Vector store:** ChromaDB with HuggingFace `all-MiniLM-L6-v2` embeddings
- **Retrieval:** LangChain `create_retrieval_chain` over persona definitions, reports, CSV data, markdown docs
- **LLM:** Groq via OpenAI-compatible endpoint — temperature 0.3 for factual answers
- **Sources:** Returns document sources for answer provenance
- **Query examples:** "What's the churn risk for VIP customers?" / "Which persona has the highest return rate?"

### 13. Kafka Streaming Pipeline

**`src/streaming.py`** + **`POST /stream/predict`** — Real-time transaction ingestion:

- **Producer:** `AIOKafkaProducer` with async `send()` and `send_batch()` methods
- **Consumer:** `AIOKafkaConsumer` with configurable group ID and auto-offset reset
- **Topic:** Configurable via `KAFKA_TOPIC` env var (default: `buyer-persona-transactions`)
- **Lifecycle:** Explicit connect/disconnect endpoints for container orchestration
- **Use case:** Stream transactions from webhook → Kafka → ML pipeline → persona update

### 14. REST API

**`api/`** — Production-grade FastAPI with 15+ endpoints and automatic OpenAPI docs at `/docs`:

| Method | Path | Description | Cache |
|--------|------|-------------|-------|
| `GET` | `/health` | DB, Redis, model status | No |
| `POST` | `/predict` | Predict personas from transaction batch | Redis (24h) |
| `POST` | `/predict/explain` | SHAP feature attribution per prediction | No |
| `POST` | `/predict/churn` | Churn probability with risk factors | No |
| `POST` | `/train` | Queue async Celery training job | No |
| `GET` | `/train/{task_id}` | Poll training task status | No |
| `POST` | `/train/churn` | Train churn prediction model | No |
| `POST` | `/anomalies/train` | Train anomaly detection model | No |
| `GET` | `/anomalies` | Batch anomaly detection on all customers | No |
| `GET` | `/models` | List model versions | No |
| `POST` | `/models/deploy` | Deploy specific model version | Redis |
| `POST` | `/models/rollback` | Rollback to previous version | Redis |
| `GET` | `/health/drift` | Feature drift status vs. baseline | No |
| `GET` | `/forecast` | Revenue forecast per persona | JSON cache |
| `POST` | `/forecast/refresh` | Refresh all forecast models | No |
| `POST` | `/persona/narrate` | LLM-generated persona narrative | No |
| `GET` | `/persona/narrate/all` | Narratives for all personas | No |
| `POST` | `/chat` | RAG chatbot query over business insights | No |
| `POST` | `/stream/predict` | Publish transactions to Kafka | No |
| `POST` | `/stream/connect` | Connect Kafka producer | No |
| `POST` | `/stream/disconnect` | Disconnect Kafka producer | No |

**Engineering highlights:**
- Asynchronous throughout — asyncpg, aioredis, aiokafka
- Lifespan-managed model loading with graceful shutdown
- Redis caching by feature hash (SHA-256) — 24h TTL
- Request ID middleware (UUID + ms timing on every request)
- Structured JSON error handling (HTTP, validation, 500)
- Graceful degradation — health reflects partial availability

### 15. Dashboard

**`dashboard/app.py`** — Streamlit dashboard (6 pages) with Neon + CSV dual data source:

| Page | What It Shows |
|------|---------------|
| **Dataset Overview** | KPI cards, sample data, descriptive stats, persona distribution |
| **Feature Engineering** | Histograms (2×4 grid), correlation heatmap |
| **PCA & UMAP** | 2D scatter, interactive 3D scatter, on-demand UMAP |
| **Clustering Results** | Silhouette + Davies-Bouldin scores, feature heatmap, cluster sizes |
| **Persona Explorer** | Per-persona feature profiles, radar chart, PCA-highlight view |
| **Business Recommendations** | Expandable persona cards with descriptions + strategies, downloadable reports |

Redis-backed caching (`dashboard/cache.py`) for dataframes and matplotlib figures.

### 16. Async Task Queue

**`src/celery_app.py`** + **`src/tasks.py`** — Distributed task queue with Redis broker/backend:

- `train_pipeline_task` wraps the full async ML pipeline in a Celery task
- Auto-retry on failure (`max_retries=2`)
- Status polling via `GET /train/{task_id}`

### 17. MLflow Tracking & Model Registry

**`src/tracking.py`** + **`src/model_registry.py`** — Full experiment lifecycle:

**Every run tracks:** parameters (PCA components, features, method), metrics (Silhouette, Davies-Bouldin, stability ARI), and artifacts (models, personas CSV).

**Registry lifecycle:**
```
register_model(run_id) ──▶ Staging ──▶ Production ──▶ Archived
                                 │
                                 └── Rollback (fallback)
```

Active version synced to Redis — API reads active version at startup for transparent rollouts.

### 18. Drift Detection

**`src/drift_detector.py`** — Automated feature drift monitoring:

| Test | What It Detects | Threshold |
|------|-----------------|-----------|
| **PSI** (Population Stability Index) | Distribution shift magnitude | > 0.25 |
| **KL Divergence** | Information loss between distributions | > 0.1 |
| **KS Test** | Two-sample Kolmogorov-Smirnov | p < 0.05 |

Per-feature drift reporting with overall drift ratio. Exposed via `GET /health/drift`.

### 19. Caching & Feature Store

**Redis cache layers:**

| Cache | Key Pattern | TTL | Purpose |
|-------|-------------|-----|---------|
| Prediction | `predict:{cid}:{feat_hash}` | 24h | Avoid re-inference for same features |
| Feature | `features:{customer_id}` | 1h | Feature vector lookup |
| Model Active | `model:active` | ∞ | Active deployment pointer |
| Dashboard | `dashboard:{prefix}:{hash}` | 1h | Cached figures + dataframes |
| Celery | Standard | Configurable | Task broker + result backend |

**Feature Store (`src/feature_store.py`):**
1. Load all transactions from Neon
2. Compute feature vectors (upsert — insert new, update existing)
3. Populate Redis for fast API lookups

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
| `mlflow` | Custom MLflow | Experiment tracking | — |
| `api` | FastAPI + Uvicorn | REST API (all endpoints) | redis, postgres |
| `dashboard` | Streamlit | UI dashboard | api |
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

### CI/CD (GitHub Actions)

| Pipeline | Trigger | Jobs |
|----------|---------|------|
| **CI** | Push/PR to `main` | Ruff lint → mypy type check → pytest (22 tests) → pipeline smoke test → Docker build |
| **CD** | Tag `v*` | Build & push API/dashboard/worker images to GitHub Container Registry |

### Production Profile

`docker-compose.prod.yml` — strips local Postgres (uses external Neon `DATABASE_URL`), removes Prometheus/Grafana for lean deployment.

---

## 📁 Project Structure

```
buyer-persona-ml/
│
├── api/                              # FastAPI REST API (21 endpoints)
│   ├── main.py                       # App factory, lifespan, 11 routers registered
│   ├── schemas.py                    # Pydantic v2 models (all request/response types)
│   ├── dependencies.py               # ModelLoader singleton
│   ├── middleware.py                 # Request logging with UUID + timing
│   ├── exception_handlers.py         # Structured JSON errors
│   └── routes/
│       ├── predict.py                # POST /predict (Redis-cached persona prediction)
│       ├── explain.py                # POST /predict/explain (SHAP attribution)
│       ├── churn.py                  # POST /predict/churn + POST /train/churn
│       ├── anomalies.py              # GET /anomalies + POST /anomalies/train
│       ├── forecast.py               # GET /forecast + POST /forecast/refresh
│       ├── persona.py                # POST /persona/narrate + GET /persona/narrate/all
│       ├── chat.py                   # POST /chat (RAG chatbot)
│       ├── stream.py                 # POST /stream/predict + connect/disconnect
│       ├── training.py               # POST /train + GET /train/{task_id}
│       ├── models.py                 # GET /models, POST /models/deploy, /rollback
│       └── drift.py                  # GET /health/drift
│
├── dashboard/
│   ├── app.py                        # Streamlit dashboard (6 pages)
│   └── cache.py                      # Redis-backed figure/DF cache
│
├── src/                              # Core ML + infrastructure
│   ├── config.py                     # Paths, constants, persona definitions, env vars
│   ├── database.py                   # Async SQLAlchemy engine (asyncpg)
│   ├── models.py                     # ORM: Customer, Transaction, CustomerFeature
│   ├── preprocessing.py              # Load → clean → scale → select
│   ├── features.py                   # 14 RFM + behavioral features
│   ├── clustering.py                 # KMeans, DBSCAN, GMM, Agglomerative
│   ├── tuning.py                     # Optuna hyperparameter optimization
│   ├── evaluation.py                 # Silhouette, DB, intra/inter, stability
│   ├── explainer.py                  # SHAP KernelExplainer wrapper
│   ├── churn.py                      # RandomForest churn prediction
│   ├── anomaly_detector.py           # IsolationForest anomaly detection
│   ├── forecast.py                   # Prophet time-series forecasting
│   ├── llm.py                        # Groq LLM persona narrative generator
│   ├── rag.py                        # LangChain + ChromaDB RAG chatbot
│   ├── streaming.py                  # AIOKafka producer/consumer
│   ├── pipeline.py                   # Async end-to-end orchestrator
│   ├── predict.py                    # CLI batch inference
│   ├── tracking.py                   # MLflow experiment logger
│   ├── model_registry.py             # Model version lifecycle
│   ├── feature_store.py              # Batch compute + store features
│   ├── drift_detector.py             # PSI, KS, KL drift detection
│   ├── cache.py                      # Redis async client
│   ├── celery_app.py                 # Celery configuration
│   ├── tasks.py                      # Celery training task
│   ├── data_generator.py             # Synthetic data → Neon
│   └── visualization.py              # PCA scatter, heatmap
│
├── models/                           # Serialized artifacts
│   ├── MODEL_CARD.md
│   ├── scaler.pkl, pca.pkl, kmeans.pkl, selected_features.pkl
│   ├── churn_model.pkl, anomaly_model.pkl
│   └── forecast/                     # Per-persona forecast JSON cache
│
├── data/                             # Source + processed data
│   ├── raw/transactions.csv
│   └── processed/
│
├── chroma_db/                        # ChromaDB persistent vector store (RAG)
├── alembic/                          # Database migrations
├── notebooks/                        # 5 Jupyter notebooks (EDA → Insights)
├── tests/test_all.py                 # 22 unit tests (pytest)
├── reports/                          # Experiment logs
│
├── docker-compose.yml                # Full dev stack (8 services)
├── docker-compose.prod.yml           # Production stack
├── Dockerfile                        # Multi-stage build
├── Makefile                          # 15+ convenience targets
├── .github/workflows/                # CI + CD pipelines
├── requirements.txt                  # 30 pinned dependencies
├── alembic.ini
└── README.md
```

---

## 🚀 Quick Start

```bash
# 1. Clone & install
cd buyer-persona-ml
pip install -r requirements.txt

# 2. Run the full pipeline (CSV mode — no database needed)
python -m src.pipeline --csv

# 3. Launch dashboard
streamlit run dashboard/app.py

# 4. Start API
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs

# 5. Batch predict
python -m src.predict --input data/raw/transactions.csv --output predictions.csv

# 6. Get SHAP explanations
curl -X POST http://localhost:8000/predict/explain \
  -H "Content-Type: application/json" \
  -d @data/raw/transactions_sample.json

# 7. Generate LLM persona narrative (set LLM_API_KEY in .env)
curl -X POST http://localhost:8000/persona/narrate \
  -H "Content-Type: application/json" \
  -d '{"persona": "VIP Loyal Customers"}'

# 8. Ask the RAG chatbot
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What marketing strategies work best for Churn Risk customers?"}'

# 9. Get revenue forecast
curl "http://localhost:8000/forecast?persona=VIP%20Loyal%20Customers"

# 10. Full Docker stack
docker compose up --build -d

# 11. Run tests
python -m pytest tests/ -v
```

### Makefile Quick Reference

```bash
make pipeline       # Full ML pipeline
make test           # Run tests (22 unit tests)
make run            # Launch dashboard
make run-api        # Start API server
make generate-data  # Generate synthetic data
make feature-store  # Compute + store features
make run-worker     # Start Celery worker
make docker-up      # Full stack up
make docker-down    # Full stack down
```

---

## 📋 Complete API Reference

Full interactive documentation at `http://localhost:8000/docs` (Swagger UI) and `/redoc` (ReDoc).

### Health & Monitoring
```
GET /health
→ { "status": "healthy", "database": "connected", "redis": "connected",
    "model": "loaded", "model_version": "kmeans_k4" }

GET /health/drift
→ { "status": "healthy", "drift_detected": false, "features_checked": 10,
    "features_with_drift": 0, "drift_ratio": 0 }
```

### Persona Prediction
```
POST /predict
Body: { "transactions": [{ "invoice_id", "customer_id", "invoice_date",
       "product_category", "product_id", "quantity", "unit_price",
       "discount_pct", "payment_method", "returned" }] }
→ { "predictions": [{ "customer_id": "C1001", "cluster": 0,
      "persona": "VIP Loyal Customers" }], "model_version": "kmeans_k4" }
```

### SHAP Explainability
```
POST /predict/explain
Body: [same transaction input as /predict]
→ [{ "customer_id": "C1001", "persona": "VIP Loyal Customers",
     "base_value": -0.32,
     "contributions": [{ "feature": "Recency", "importance": 0.28,
                         "direction": "negative" }, ...] }]
```

### Churn Prediction
```
POST /predict/churn
Body: [same transaction input as /predict]
→ { "predictions": [{ "customer_id": "C1001", "churn_probability": 0.12,
      "churn_risk": "Low",
      "top_factors": [{ "feature": "Recency", "importance": 0.31,
                        "direction": "positive" }] }],
    "model_version": "kmeans_k4" }

POST /train/churn
→ { "status": "completed", "metrics": { "roc_auc": 0.89,
      "avg_precision": 0.76, "churn_rate": 0.18 },
    "top_features": [{ "feature": "Recency", "importance": 0.24 }, ...] }
```

### Anomaly Detection
```
GET /anomalies
→ { "total_checked": 1000, "anomalies_found": 47,
    "results": [{ "customer_id": "C1042", "anomaly_score": -0.32,
                  "is_anomaly": true, "reconstruction_error": 0.67 }] }

POST /anomalies/train
→ { "status": "completed", "total_checked": 1000, "anomalies_found": 50,
    "anomaly_rate": 0.05 }
```

### LLM Persona Narratives
```
POST /persona/narrate
Body: { "persona": "VIP Loyal Customers",
        "profile": { "Recency": 5.2, "Frequency": 24, "Monetary": 12500 } }
→ { "persona": "VIP Loyal Customers",
    "narrative": "These are your highest-value customers...",
    "model_used": "llama-3.3-70b-versatile" }

GET /persona/narrate/all
→ [{ "persona": "VIP Loyal Customers", "narrative": "...",
     "model_used": "llama-3.3-70b-versatile" }, ...]
```

### RAG Chatbot
```
POST /chat
Body: { "query": "Which persona has the highest return rate?",
        "history": [] }
→ { "answer": "One-Time Buyers have the highest return rate at 22%...",
    "sources": ["src/config.py", "reports/experiment_log.json"] }
```

### Time-Series Forecast
```
GET /forecast?persona=VIP%20Loyal%20Customers&metric=monetary
→ [{ "persona": "VIP Loyal Customers", "metric": "monetary",
     "forecast": [{ "date": "2026-08-01", "predicted_value": 45230.0,
                    "lower_bound": 38900.0, "upper_bound": 51560.0 }] }]

POST /forecast/refresh
→ { "status": "completed", "personas_refreshed": ["VIP Loyal Customers",
    "Discount Hunters", "Churn Risk", "One-Time Buyers"] }
```

### Kafka Streaming
```
POST /stream/connect
→ { "status": "connected", "broker": "localhost:9092" }

POST /stream/predict
Body: { "transactions": [{ ... }] }
→ { "status": "produced", "message": "Sent 5/5 transactions to Kafka topic
    'buyer-persona-transactions'." }

POST /stream/disconnect
→ { "status": "disconnected" }
```

### Async Training
```
POST /train?source=neon
→ { "task_id": "uuid", "status": "queued",
    "message": "Training job queued successfully." }

GET /train/{task_id}
→ { "task_id": "uuid", "status": "SUCCESS",
    "message": "{'customers': 1000, 'clusters': 4}" }
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

---

## ✅ Testing

**22 unit tests** covering every stage:

```
src/preprocessing   → CleanData, HandleOutliers, ScaleFeatures, FeatureSelection
src/features        → BuildCustomerFeatures (RFM columns, behavioral, shapes)
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
| **ML & Data** | scikit-learn, pandas, numpy, umap-learn, SHAP |
| **Deep Learning** | IsolationForest (anomaly detection) |
| **Hyperparameter Tuning** | Optuna (TPE sampler) |
| **Time Series** | Prophet |
| **LLM & GenAI** | Groq API (llama-3.3-70b), httpx |
| **RAG & Vector DB** | LangChain, ChromaDB, sentence-transformers |
| **Streaming** | AIOKafka |
| **Framework** | FastAPI, Uvicorn, Pydantic v2 |
| **Database** | PostgreSQL (Neon), SQLAlchemy 2.0 (async), asyncpg, Alembic |
| **Caching & Queue** | Redis (async), Celery |
| **Dashboard** | Streamlit, matplotlib, seaborn |
| **MLOps** | MLflow (tracking + model registry) |
| **Infrastructure** | Docker, Docker Compose, Prometheus, Grafana |
| **CI/CD** | GitHub Actions (lint → typecheck → test → build → deploy) |
| **Testing** | pytest |

---

## 📄 License

MIT
