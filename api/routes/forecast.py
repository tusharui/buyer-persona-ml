from fastapi import APIRouter, HTTPException

from api.schemas import ForecastResponse, ForecastPoint
from src.forecast import prepare_time_series, forecast_persona, save_forecast, get_forecast_for_persona
from src.config import PERSONA_MAP, PROCESSED_DIR
import pandas as pd

router = APIRouter(tags=["forecast"])


@router.get("/forecast", response_model=list[ForecastResponse])
async def get_forecast(persona: str = None, metric: str = "monetary"):
    personas_path = PROCESSED_DIR / "customer_personas.csv"
    if not personas_path.exists():
        raise HTTPException(status_code=404, detail="No persona data found.")

    df = pd.read_csv(personas_path)
    transactions_path = PROCESSED_DIR.parent / "raw" / "transactions.csv"
    tx_df = None
    if transactions_path.exists():
        tx_df = pd.read_csv(transactions_path, parse_dates=["InvoiceDate"])

    if tx_df is None:
        df_sales = df.copy()
        df_sales["InvoiceDate"] = pd.Timestamp.now() - pd.to_timedelta(
            df_sales["Recency"], unit="D"
        )
        tx_df = df_sales

    if persona:
        personas_to_forecast = [persona]
    else:
        personas_to_forecast = list(PERSONA_MAP.values())

    results = []
    for p in personas_to_forecast:
        cached = get_forecast_for_persona(p, metric)
        if cached:
            results.append(ForecastResponse(
                persona=p,
                metric=metric,
                forecast=[ForecastPoint(**pt) for pt in cached],
            ))
            continue

        mask = tx_df["Persona"] == p if "Persona" in tx_df.columns else tx_df.get("Persona", "").isin([p])
        if mask.sum() == 0:
            continue

        series = prepare_time_series(tx_df, persona_col="Persona")
        if p not in series:
            continue

        pts = forecast_persona(series[p])
        if pts:
            save_forecast(p, pts, metric)
            results.append(ForecastResponse(
                persona=p,
                metric=metric,
                forecast=[ForecastPoint(**pt) for pt in pts],
            ))

    return results


@router.post("/forecast/refresh")
async def refresh_forecasts():
    personas_path = PROCESSED_DIR / "customer_personas.csv"
    transactions_path = PROCESSED_DIR.parent / "raw" / "transactions.csv"

    tx_df = pd.read_csv(transactions_path, parse_dates=["InvoiceDate"]) if transactions_path.exists() else None
    if tx_df is None:
        df = pd.read_csv(personas_path)
        df["InvoiceDate"] = pd.Timestamp.now() - pd.to_timedelta(df["Recency"], unit="D")
        tx_df = df

    series = prepare_time_series(tx_df, persona_col="Persona")
    refreshed = []
    for persona, monthly_df in series.items():
        pts = forecast_persona(monthly_df)
        if pts:
            save_forecast(persona, pts)
            refreshed.append(persona)

    return {"status": "completed", "personas_refreshed": refreshed}
