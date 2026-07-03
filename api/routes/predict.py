import hashlib
import json

import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import PredictRequest, PredictResponse, PersonaPrediction
from api.dependencies import model_loader
from src.features import build_customer_features
from src.cache import cache
from src.config import PERSONA_MAP

router = APIRouter(tags=["predict"])

COLUMN_MAP_IN = {
    "invoice_id": "InvoiceID",
    "customer_id": "CustomerID",
    "invoice_date": "InvoiceDate",
    "product_category": "ProductCategory",
    "product_id": "ProductID",
    "quantity": "Quantity",
    "unit_price": "UnitPrice",
    "discount_pct": "DiscountPct",
    "payment_method": "PaymentMethod",
    "returned": "Returned",
}


def _feature_hash(features: dict) -> str:
    raw = json.dumps(features, sort_keys=True)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded. Run pipeline first.")

    raw = [t.model_dump() for t in req.transactions]
    df = pd.DataFrame(raw)
    df.rename(columns=COLUMN_MAP_IN, inplace=True)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    if df["CustomerID"].nunique() == 0:
        raise HTTPException(status_code=400, detail="No customers found in transactions.")

    features_df = build_customer_features(df).fillna(0)
    predictions = []

    for _, row in features_df.iterrows():
        cid = row["CustomerID"]
        feat_vec = {col: float(row[col]) for col in model_loader.features}
        fhash = _feature_hash(feat_vec)
        cache_key = f"predict:{cid}:{fhash}"

        cached = await cache.get(cache_key)
        if cached is not None:
            result = json.loads(cached)
            predictions.append(PersonaPrediction(**result))
            continue

        scaled = model_loader.scaler.transform(pd.DataFrame([feat_vec]))
        reduced = model_loader.pca.transform(scaled) if model_loader.pca is not None else scaled
        cluster = int(model_loader.kmeans.predict(reduced)[0])
        persona = PERSONA_MAP.get(cluster, "Unknown")

        result = PersonaPrediction(customer_id=cid, cluster=cluster, persona=persona)
        await cache.set(cache_key, result.model_dump_json(), ttl=cache.prediction_cache_ttl)
        predictions.append(result)

    return PredictResponse(predictions=predictions, model_version=model_loader.version)
