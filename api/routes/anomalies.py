import pandas as pd
from fastapi import APIRouter, HTTPException

from api.schemas import AnomalyResponse, AnomalyResult
from api.dependencies import model_loader
from src.anomaly_detector import anomaly_detector
from src.features import build_customer_features

router = APIRouter(tags=["anomalies"])

COLUMN_MAP_IN = {
    "invoice_id": "InvoiceID", "customer_id": "CustomerID",
    "invoice_date": "InvoiceDate", "product_category": "ProductCategory",
    "product_id": "ProductID", "quantity": "Quantity",
    "unit_price": "UnitPrice", "discount_pct": "DiscountPct",
    "payment_method": "PaymentMethod", "returned": "Returned",
}


@router.get("/anomalies", response_model=AnomalyResponse)
async def detect_anomalies():
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded. Run pipeline first.")

    if not anomaly_detector.is_loaded():
        loaded = anomaly_detector.load()
        if not loaded.is_loaded():
            raise HTTPException(status_code=503, detail="Anomaly model not found. Train it first.")

    from src.config import PROCESSED_DIR
    personas_path = PROCESSED_DIR / "customer_personas.csv"
    if not personas_path.exists():
        raise HTTPException(status_code=404, detail="No persona data found.")

    df = pd.read_csv(personas_path)
    feat_cols = [c for c in model_loader.features if c in df.columns]
    results = anomaly_detector.detect_batch(df)

    return AnomalyResponse(
        total_checked=len(results),
        anomalies_found=sum(1 for r in results if r["is_anomaly"]),
        results=[AnomalyResult(**r) for r in results],
    )


@router.post("/anomalies/train")
async def train_anomaly_model():
    if not model_loader.is_loaded():
        raise HTTPException(status_code=503, detail="Model not loaded. Run pipeline first.")

    from src.config import PROCESSED_DIR
    personas_path = PROCESSED_DIR / "customer_personas.csv"
    if not personas_path.exists():
        raise HTTPException(status_code=404, detail="No persona data found.")

    df = pd.read_csv(personas_path)
    feat_cols = [c for c in model_loader.features if c in df.columns]
    result = anomaly_detector.train(df, feat_cols)

    return {
        "status": "completed",
        "total_checked": result["total_checked"],
        "anomalies_found": result["anomalies_found"],
        "anomaly_rate": result["anomaly_rate"],
    }
