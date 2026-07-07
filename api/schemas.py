from pydantic import BaseModel
from typing import Optional


class TransactionInput(BaseModel):
    invoice_id: str
    customer_id: str
    invoice_date: str
    product_category: str
    product_id: str
    quantity: int
    unit_price: float
    discount_pct: float
    payment_method: str
    returned: bool


class PredictRequest(BaseModel):
    transactions: list[TransactionInput]


class PersonaPrediction(BaseModel):
    customer_id: str
    cluster: int
    persona: str


class PredictResponse(BaseModel):
    predictions: list[PersonaPrediction]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    database: str
    redis: str
    model: str
    model_version: Optional[str] = None


class TrainResponse(BaseModel):
    task_id: str
    status: str
    message: str


class ModelVersion(BaseModel):
    version: str
    silhouette: Optional[float] = None
    created_at: Optional[str] = None
    status: str


class FeatureAttribution(BaseModel):
    feature: str
    importance: float
    direction: str


class ExplainResponse(BaseModel):
    customer_id: str
    persona: str
    base_value: float
    contributions: list[FeatureAttribution]


class ChurnPrediction(BaseModel):
    customer_id: str
    churn_probability: float
    churn_risk: str
    top_factors: list[FeatureAttribution]


class ChurnPredictResponse(BaseModel):
    predictions: list[ChurnPrediction]
    model_version: str


class AnomalyResult(BaseModel):
    customer_id: str
    anomaly_score: float
    is_anomaly: bool
    reconstruction_error: float


class AnomalyResponse(BaseModel):
    total_checked: int
    anomalies_found: int
    results: list[AnomalyResult]


class ForecastPoint(BaseModel):
    date: str
    predicted_value: float
    lower_bound: float
    upper_bound: float


class ForecastResponse(BaseModel):
    persona: str
    metric: str
    forecast: list[ForecastPoint]


class NarrateRequest(BaseModel):
    persona: str
    profile: Optional[dict] = None


class NarrateResponse(BaseModel):
    persona: str
    narrative: str
    model_used: str


class ChatRequest(BaseModel):
    query: str
    history: Optional[list[dict]] = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[str]


class StreamPredictRequest(BaseModel):
    transactions: list[TransactionInput]


class StreamPredictResponse(BaseModel):
    status: str
    message: str
