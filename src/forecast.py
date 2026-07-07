import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
from pathlib import Path

from src.config import MODELS_DIR, FORECAST_DAYS


FORECAST_MODEL_DIR = MODELS_DIR / "forecast"


def _ensure_dir():
    FORECAST_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def prepare_time_series(df: pd.DataFrame, persona_col: str = "Persona") -> dict[str, pd.DataFrame]:
    if "InvoiceDate" not in df.columns and "invoice_date" not in df.columns:
        return {}
    date_col = "InvoiceDate" if "InvoiceDate" in df.columns else "invoice_date"
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["Month"] = df[date_col].dt.to_period("M").dt.to_timestamp()

    if "Monetary" not in df.columns:
        if all(c in df.columns for c in ("Quantity", "UnitPrice")):
            discount = df["DiscountPct"] / 100.0 if "DiscountPct" in df.columns else 0.0
            df["Monetary"] = df["Quantity"] * df["UnitPrice"] * (1.0 - discount)
        elif "TotalAmount" in df.columns:
            df["Monetary"] = df["TotalAmount"]
        elif "Revenue" in df.columns:
            df["Monetary"] = df["Revenue"]
        else:
            return {}

    series = {}
    for persona in df[persona_col].unique():
        mask = df[persona_col] == persona
        monthly = df[mask].set_index(date_col).resample("ME")["Monetary"].sum().reset_index()
        monthly.columns = ["ds", "y"]
        monthly = monthly.dropna()
        if len(monthly) > 3:
            series[persona] = monthly
    return series


def forecast_persona(monthly_df: pd.DataFrame, periods: int = None) -> list[dict]:
    try:
        from prophet import Prophet
    except ImportError:
        return _simple_forecast(monthly_df, periods)

    periods = periods or FORECAST_DAYS // 30
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
    )
    model.fit(monthly_df)
    future = model.make_future_dataframe(periods=periods, freq="ME")
    forecast = model.predict(future)

    results = []
    for _, row in forecast.tail(periods).iterrows():
        results.append({
            "date": row["ds"].strftime("%Y-%m-%d"),
            "predicted_value": round(max(0, row["yhat"]), 2),
            "lower_bound": round(max(0, row["yhat_lower"]), 2),
            "upper_bound": round(max(0, row["yhat_upper"]), 2),
        })
    return results


def _simple_forecast(monthly_df: pd.DataFrame, periods: int = None) -> list[dict]:
    periods = periods or FORECAST_DAYS // 30
    if len(monthly_df) < 2:
        return []
    from sklearn.linear_model import LinearRegression
    monthly_df = monthly_df.copy().reset_index(drop=True)
    X = np.arange(len(monthly_df)).reshape(-1, 1)
    y = monthly_df["y"].values
    model = LinearRegression()
    model.fit(X, y)
    last_idx = len(monthly_df)
    future_dates = pd.date_range(
        start=monthly_df["ds"].iloc[-1] + pd.DateOffset(months=1),
        periods=periods,
        freq="ME",
    )
    future_X = np.arange(last_idx, last_idx + periods).reshape(-1, 1)
    preds = model.predict(future_X)
    residuals = np.std(y - model.predict(X))
    results = []
    for i, date in enumerate(future_dates):
        results.append({
            "date": date.strftime("%Y-%m-%d"),
            "predicted_value": round(max(0, preds[i]), 2),
            "lower_bound": round(max(0, preds[i] - 1.96 * residuals), 2),
            "upper_bound": round(max(0, preds[i] + 1.96 * residuals), 2),
        })
    return results


def get_forecast_for_persona(persona: str, metric: str = "monetary") -> list[dict]:
    forecast_path = FORECAST_MODEL_DIR / f"{persona.lower().replace(' ', '_')}.json"
    if forecast_path.exists():
        import json
        with open(forecast_path) as f:
            data = json.load(f)
            return data.get(metric, [])
    return []


def save_forecast(persona: str, results: list[dict], metric: str = "monetary"):
    _ensure_dir()
    import json
    forecast_path = FORECAST_MODEL_DIR / f"{persona.lower().replace(' ', '_')}.json"
    data = {metric: results}
    if forecast_path.exists():
        with open(forecast_path) as f:
            existing = json.load(f)
            existing[metric] = results
            data = existing
    with open(forecast_path, "w") as f:
        json.dump(data, f, indent=2)
