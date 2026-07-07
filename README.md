# Buyer Persona ML

<p align="center">
  <b>AI-Powered Customer Segmentation Engine</b><br>
  Unsupervised ML → SHAP Explainability → Churn Prediction → Anomaly Detection →<br>
  LLM Narratives → RAG Chatbot → Time-Series Forecast → Kafka Streaming
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

An end-to-end AI system that takes raw customer purchase data, automatically groups customers into meaningful segments (like VIP Loyal, Discount Hunters, Churn Risk, One-Time Buyers), and provides actionable insights through an API, interactive dashboard, and AI-powered tools. Built with 19+ features spanning machine learning, LLMs, real-time streaming, and cloud deployment.

> **Data:** 10K synthetic transactions, 1K customers for demonstration.

---

## 🔄 How It Works — End to End

```
                    ┌──────────────────────────────────────────────────────────┐
                    │                   INPUT LAYER                           │
                    │                                                          │
Input Data ────────▶│  Raw CSV or PostgreSQL  ──▶  Clean & Validate Data      │
(transactions.csv)  │                                                          │
                    └──────────────────────┬───────────────────────────────────┘
                                           ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │              FEATURE ENGINEERING                        │
                    │                                                          │
                    │  For each customer, calculate:                          │
                    │  • Recency (days since last purchase)                   │
                    │  • Frequency (how many orders)                          │
                    │  • Monetary (total spent)                               │
                    │  • Average basket size, discount usage, return rate     │
                    │  • Weekend/night shopping habits, product diversity     │
                    │                                                          │
                    └──────────────────────┬───────────────────────────────────┘
                                           ▼
                    ┌──────────────────────────────────────────────────────────┐
                    │         MACHINE LEARNING PIPELINE                       │
                    │                                                          │
                    │  1. PCA (compresses 14 features into 4-6 dimensions)    │
                    │  2. Compare 5 clustering algorithms                     │
                    │  3. Pick the best one automatically                     │
                    │  4. Assign persona labels to each customer              │
                    │                                                          │
                    └──────┬───────────────┬───────────────┬──────────────────┘
                           ▼               ▼               ▼
              ┌────────────────┐ ┌────────────────┐ ┌────────────────┐
              │  SHAP Explain  │ │  Churn Model   │ │ Anomaly Detect │
              │  Why was this  │ │ Will this cust │ │ Is this cust's │
              │  customer put  │ │ leave soon?    │ │ behavior       │
              │  in this group?│ │ (probability   │ │ unusual?       │
              │                │ │  + risk level) │ │ (suspicious?)  │
              └────────────────┘ └────────────────┘ └────────────────┘
                           ▼               ▼               ▼
              ┌──────────────────────────────────────────────────────┐
              │              OUTPUT & INSIGHTS                      │
              │                                                      │
              │  1. REST API  ── 21 endpoints for any app to use    │
              │  2. Dashboard ── Visual charts and reports           │
              │  3. Forecast  ── Predict next 3 months revenue      │
              │  4. LLM Story ── AI writes persona descriptions     │
              │  5. Chatbot   ── Ask questions in plain English     │
              │  6. Streaming ── Real-time via Kafka                │
              │                                                      │
              └──────────────────────────────────────────────────────┘
```

---

## 🎯 What It Does

### 1. Groups Customers Automatically
The system analyzes purchase patterns and divides customers into four segments:

| Group | Who They Are | What To Do |
|-------|-------------|------------|
| 🏆 VIP Loyal | Best customers — buy often, spend lots, shop recently | Give them exclusive rewards and VIP treatment |
| 💰 Discount Hunters | Only buy during sales, small baskets, use discounts | Send flash sales and bundle deals |
| ⚠️ Churn Risk | Haven't bought in a while, low spending | Win them back with special offers |
| 🆕 One-Time Buyers | Bought once and disappeared | Cross-sell and nurture them |

### 2. Explains Its Decisions (SHAP)
When asked "Why is this customer in this group?", the API returns a breakdown showing exactly which behaviors drove the decision — like "Recency was the biggest factor pushing them toward Churn Risk."

### 3. Predicts Who Will Leave (Churn)
For every customer, it predicts how likely they are to stop buying (0-100%) and shows the top 3 reasons why. Businesses can target at-risk customers before they leave.

### 4. Flags Suspicious Customers (Anomaly Detection)
Automatically detects customers whose behavior looks unusual — could be data errors, fraud, or one-time bulk buyers that need a second look.

### 5. Forecasts Future Revenue (Time-Series)
For each customer group, predicts how much they'll spend in the next 3 months with confidence ranges. Helps with budgeting and inventory planning.

### 6. Generates AI Persona Descriptions (Groq LLM)
Feeds behavioral profiles to an LLM (llama-3.3-70b via Groq) which writes detailed marketing descriptions — demographics, purchase drivers, channel preferences, lifetime value potential. Ready for stakeholder presentations.

### 7. Answers Questions in Plain English (RAG Chatbot)
Chat with the system using natural language:
- *"What marketing strategies work best for Churn Risk customers?"*
- *"Which persona has the highest return rate?"*
The chatbot searches through business documents and returns answers with sources.

### 8. Processes Data in Real-Time (Kafka Streaming)
Instead of batch processing once a day, the system can receive transactions through Kafka and update predictions in real-time — useful for e-commerce checkouts, mobile apps, and live dashboards.

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           HOW USERS INTERACT                                │
│                                                                             │
│   ┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐     │
│   │  Web Dashboard   │    │  API (curl/app)  │    │  Real-time Feed  │     │
│   │  (Streamlit)     │    │  (FastAPI)       │    │  (Kafka)         │     │
│   └────────┬─────────┘    └────────┬─────────┘    └────────┬─────────┘     │
└────────────┼────────────────────────┼──────────────────────┼───────────────┘
             │                        │                      │
             ▼                        ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CORE APPLICATION                                    │
│                                                                             │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                          FastAPI Server                              │  │
│   │                                                                      │  │
│   │  /predict  /explain  /churn  /anomalies  /forecast  /narrate  /chat │  │
│   │                                                                      │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
│                                      │                                      │
│   ┌──────────────────────────────────────────────────────────────────────┐  │
│   │                   Background Worker (Celery)                         │  │
│   │              Runs ML training, churn model, forecasts                │  │
│   └──────────────────────────────────────────────────────────────────────┘  │
└────────────────────────────┬───────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     DATA & MACHINE LEARNING                                 │
│                                                                             │
│   Clean Data → Build Features → PCA → Compare 5 Models → Pick Best        │
│                                                                             │
│   Additional models built on top:                                          │
│   ┌────────┐ ┌────────┐ ┌──────────┐ ┌────────┐ ┌───────┐ ┌──────────┐   │
│   │ SHAP   │ │ Churn  │ │ Anomaly  │ │Forecast│ │ LLM   │ │  RAG     │   │
│   │Explain │ │Predict │ │ Detect   │ │(Prophet)│ │Story  │ │ Chatbot  │   │
│   └────────┘ └────────┘ └──────────┘ └────────┘ └───────┘ └──────────┘   │
└────────────────────────────────┬───────────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            STORAGE LAYER                                    │
│                                                                             │
│   ┌────────────────┐  ┌────────────────┐  ┌──────────────┐  ┌──────────┐  │
│   │  PostgreSQL    │  │  Redis (Cache) │  │  MLflow      │  │ ChromaDB │  │
│   │  (Customer     │  │  (Fast access  │  │  (Training   │  │ (Chatbot │  │
│   │   data)        │  │   to results)  │  │   history)   │  │  memory) │  │
│   └────────────────┘  └────────────────┘  └──────────────┘  └──────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

```bash
# 1. Install everything
cd buyer-persona-ml
pip install -r requirements.txt

# 2. Run the full pipeline (CSV mode — no database needed)
python -m src.pipeline --csv

# 3. Launch the web dashboard
streamlit run dashboard/app.py

# 4. Start the API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# Open http://localhost:8000/docs to see all available endpoints

# 5. Predict customer personas from a file
python -m src.predict --input data/raw/transactions.csv --output predictions.csv

# 6. Get an AI-generated persona story (set LLM_API_KEY in .env first)
curl -X POST http://localhost:8000/persona/narrate \
  -H "Content-Type: application/json" \
  -d '{"persona": "VIP Loyal Customers"}'

# 7. Ask the chatbot
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "Which customers are most likely to leave?"}'

# 8. See revenue forecast
curl "http://localhost:8000/forecast?persona=VIP%20Loyal%20Customers"

# 9. Run the full stack with Docker
docker compose up --build -d

# 10. Run tests
python -m pytest tests/ -v
```

---

## 📋 API Endpoints (All 21)

Full interactive docs at `http://localhost:8000/docs` (Swagger UI).

| What It Does | How To Call It |
|-------------|---------------|
| Check if the system is healthy | `GET /health` |
| Predict customer groups from transactions | `POST /predict` |
| Explain why a customer was grouped that way | `POST /predict/explain` |
| Predict if a customer will stop buying | `POST /predict/churn` |
| Train the churn prediction model | `POST /train/churn` |
| Train the anomaly detection model | `POST /anomalies/train` |
| Find unusual customer behavior | `GET /anomalies` |
| Get revenue forecast for next 3 months | `GET /forecast` |
| Refresh all forecasts | `POST /forecast/refresh` |
| Get AI-written persona description | `POST /persona/narrate` |
| Get descriptions for all personas | `GET /persona/narrate/all` |
| Ask questions in plain English | `POST /chat` |
| Send transactions through Kafka | `POST /stream/predict` |
| Connect to Kafka | `POST /stream/connect` |
| Disconnect from Kafka | `POST /stream/disconnect` |
| Start training a new model (async) | `POST /train` |
| Check training progress | `GET /train/{task_id}` |
| See all model versions | `GET /models` |
| Deploy a specific model version | `POST /models/deploy` |
| Rollback to previous version | `POST /models/rollback` |
| Check if data has drifted from baseline | `GET /health/drift` |

---

## 📁 Project Structure

```
buyer-persona-ml/
├── api/                  # API server (handles all requests)
│   ├── main.py          # Server setup — connects everything
│   ├── schemas.py       # Data formats for requests/responses
│   ├── dependencies.py  # Shared tools (model loader)
│   ├── middleware.py    # Request logging
│   └── routes/          # Individual API endpoint files
│       ├── predict.py    # Persona prediction
│       ├── explain.py    # SHAP explanations
│       ├── churn.py      # Churn prediction
│       ├── anomalies.py  # Anomaly detection
│       ├── forecast.py   # Revenue forecasting
│       ├── persona.py    # LLM narrative generation
│       ├── chat.py       # RAG chatbot
│       ├── stream.py     # Kafka streaming
│       ├── training.py   # Model training
│       ├── models.py     # Model version management
│       └── drift.py      # Data drift monitoring
│
├── dashboard/            # Web dashboard (Streamlit)
│   ├── app.py           # 6-page interactive dashboard
│   └── cache.py         # Dashboard speed optimization
│
├── src/                  # Core logic
│   ├── config.py        # Settings and configuration
│   ├── database.py      # Database connection
│   ├── models.py        # Database table definitions
│   ├── preprocessing.py # Data cleaning
│   ├── features.py      # Customer feature calculation
│   ├── clustering.py    # Customer grouping algorithms
│   ├── tuning.py        # Automatic model optimization
│   ├── evaluation.py    # Model quality checks
│   ├── explainer.py     # SHAP explanations
│   ├── churn.py         # Churn prediction model
│   ├── anomaly_detector.py # Anomaly detection
│   ├── forecast.py      # Time-series forecasting
│   ├── llm.py           # AI narrative generation
│   ├── rag.py           # Chatbot system
│   ├── streaming.py     # Kafka streaming
│   ├── pipeline.py      # End-to-end pipeline runner
│   ├── predict.py       # Batch prediction from command line
│   ├── tracking.py      # Experiment logging
│   ├── model_registry.py # Model version control
│   ├── feature_store.py # Feature calculation service
│   ├── drift_detector.py # Data drift monitoring
│   ├── cache.py         # Caching service
│   ├── celery_app.py    # Background task setup
│   ├── tasks.py         # Background tasks
│   ├── data_generator.py # Generate sample data
│   └── visualization.py # Chart utilities
│
├── models/               # Saved ML models
├── data/                 # Input data files
├── notebooks/            # Jupyter notebooks (5 stages)
├── tests/                # Automated tests (22 tests)
│
├── docker-compose.yml    # Full system setup
├── Dockerfile            # Container build instructions
├── Makefile              # Shortcut commands
├── requirements.txt      # Required packages
└── README.md
```

---

## 🛠 Technologies Used

| Area | What's Used |
|------|-------------|
| **Language** | Python 3.12+ |
| **Web Framework** | FastAPI (modern, fast Python web framework) |
| **Dashboard** | Streamlit, matplotlib, seaborn |
| **Machine Learning** | scikit-learn, SHAP, Optuna |
| **LLM & AI** | Groq (llama-3.3-70b via API), LangChain |
| **Vector Database** | ChromaDB (for chatbot memory) |
| **Time-Series** | Prophet (Facebook) |
| **Streaming** | Apache Kafka (via aiokafka) |
| **Database** | PostgreSQL (Neon), SQLAlchemy |
| **Caching** | Redis |
| **Background Tasks** | Celery |
| **Containers** | Docker, Docker Compose |
| **Experiment Tracking** | MLflow |
| **Testing** | pytest (22 tests) |
| **CI/CD** | GitHub Actions |

---

## ✅ Testing

22 automated tests covering data cleaning, feature engineering, clustering, evaluation, and model registry.

```bash
python -m pytest tests/ -v
```

---

## 📄 License

MIT
