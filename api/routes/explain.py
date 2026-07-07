
import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import ExplainResponse, FeatureAttribution
from api.dependencies import model_loader
from src.explainer import persona_explainer
from src.features import build_customer_features

router = APIRouter(tags=["explain"])

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


@router.post("/predict/explain", response_model=list[ExplainResponse])
async def explain_persona(req: list[dict]):
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded. Run pipeline first.")
    if not persona_explainer.is_loaded():
        try:
            persona_explainer.load()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Explainer failed to load: {e}")

    raw = req
    df = pd.DataFrame(raw)
    df.rename(columns=COLUMN_MAP_IN, inplace=True)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

    if df["CustomerID"].nunique() == 0:
        raise HTTPException(status_code=400, detail="No customers found in transactions.")

    features_df = build_customer_features(df).fillna(0)
    responses = []

    for _, row in features_df.iterrows():
        cid = row["CustomerID"]
        feat_vec = {col: float(row[col]) for col in model_loader.features}
        feat_vec["CustomerID"] = cid

        explanation = persona_explainer.explain(feat_vec)
        responses.append(ExplainResponse(
            customer_id=explanation["customer_id"],
            persona=explanation["persona"],
            base_value=explanation["base_value"],
            contributions=[
                FeatureAttribution(**c) for c in explanation["contributions"]
            ],
        ))

    return responses
