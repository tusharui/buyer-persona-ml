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
