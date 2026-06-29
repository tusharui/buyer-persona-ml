"""Data cleaning module."""

import pandas as pd
import numpy as np
from src.config import CORRELATION_THRESHOLD, VARIANCE_THRESHOLD, RAW_DATA_PATH
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
import joblib


def load_raw_data(path=None) -> pd.DataFrame:
    path = path or RAW_DATA_PATH
    return pd.read_csv(path, parse_dates=["InvoiceDate"])


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df[~df["InvoiceID"].astype(str).str.startswith("C", na=False)]
    df = df.dropna(subset=["CustomerID"])
    df["CustomerID"] = df["CustomerID"].astype(str).str.strip()
    df = df[df["CustomerID"] != ""]
    df = df[df["Quantity"] > 0]
    return df


def handle_outliers(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    for col in cols:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
        df = df[(df[col] >= lo) & (df[col] <= hi)]
    return df


def scale_features(df: pd.DataFrame, cols: list[str], method: str = "standard",
                   save_path=None):
    scaler = StandardScaler() if method == "standard" else RobustScaler()
    df[cols] = scaler.fit_transform(df[cols])
    if save_path:
        joblib.dump(scaler, save_path)
    return df, scaler


def select_features_by_correlation(df: pd.DataFrame, cols: list[str],
                                    threshold: float = None):
    """Automatically drop one feature from each pair with |r| > threshold."""
    threshold = threshold or CORRELATION_THRESHOLD
    corr = df[cols].corr()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = set()
    for col in upper.columns:
        for row in upper.index:
            if abs(upper.loc[row, col]) > threshold:
                # Drop the feature with lower variance
                var_col = df[col].var()
                var_row = df[row].var()
                to_drop.add(col if var_col < var_row else row)
    return [c for c in to_drop if c in df.columns]


def select_features_by_variance(df: pd.DataFrame, cols: list[str],
                                 threshold: float = None):
    threshold = threshold or VARIANCE_THRESHOLD
    selector = VarianceThreshold(threshold=threshold)
    selector.fit(df[cols])
    kept = [c for c, m in zip(cols, selector.get_support()) if m]
    dropped = [c for c, m in zip(cols, selector.get_support()) if not m]
    return kept, dropped
