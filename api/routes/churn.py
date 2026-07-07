from fastapi import APIRouter, HTTPException

from api.schemas import ChurnPredictResponse, ChurnPrediction, FeatureAttribution
from api.dependencies import model_loader
from src.churn import train_churn_model, predict_churn
from src.config import PROCESSED_DIR
from src.features import build_customer_features, FEATURE_COLS
import pandas as pd

router = APIRouter(tags=["churn"])

CHURN_TRAINED = False


@router.post("/train/churn")
async def train_churn():
    global CHURN_TRAINED
    personas_path = PROCESSED_DIR / "customer_personas.csv"
    if not personas_path.exists():
        raise HTTPException(status_code=404, detail="No personas found. Run pipeline first.")

    df = pd.read_csv(personas_path)
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    if len(feat_cols) < 2:
        raise HTTPException(status_code=400, detail="Not enough feature columns for churn model.")

    result = train_churn_model(df, feat_cols)
    CHURN_TRAINED = True
    return {
        "status": "completed",
        "metrics": result["metrics"],
        "top_features": result["importance"].head(5).to_dict("records"),
    }


@router.post("/predict/churn", response_model=ChurnPredictResponse)
async def predict_churn_endpoint(req: list[dict]):
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded. Run pipeline first.")

    df = pd.DataFrame(req)
    df.rename(columns={
        "invoice_id": "InvoiceID", "customer_id": "CustomerID",
        "invoice_date": "InvoiceDate", "product_category": "ProductCategory",
        "product_id": "ProductID", "quantity": "Quantity",
        "unit_price": "UnitPrice", "discount_pct": "DiscountPct",
        "payment_method": "PaymentMethod", "returned": "Returned",
    }, inplace=True)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    features_df = build_customer_features(df).fillna(0)

    predictions = []
    for _, row in features_df.iterrows():
        feat_vec = {col: float(row[col]) for col in model_loader.features}
        feat_vec["CustomerID"] = row["CustomerID"]
        result = predict_churn(feat_vec)
        predictions.append(ChurnPrediction(
            customer_id=result["customer_id"],
            churn_probability=result["churn_probability"],
            churn_risk=result["churn_risk"],
            top_factors=[FeatureAttribution(**f) for f in result["top_factors"]],
        ))

    return ChurnPredictResponse(predictions=predictions, model_version=model_loader.version)
