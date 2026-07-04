import pandas as pd


def build_customer_features(df: pd.DataFrame, ref_date=None) -> pd.DataFrame:
    if ref_date is None:
        ref_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

    df["NetRevenue"] = df["Quantity"] * df["UnitPrice"] * (1 - df["DiscountPct"] / 100)
    df["IsWeekend"] = df["InvoiceDate"].dt.weekday.isin([5, 6]).astype(int)
    df["IsNight"] = (df["InvoiceDate"].dt.hour < 6).astype(int)

    cust = df.groupby("CustomerID").agg(
        Recency=("InvoiceDate", lambda x: (ref_date - x.max()).days),
        Frequency=("InvoiceID", "nunique"),
        Monetary=("NetRevenue", "sum"),
        AvgQuantity=("Quantity", "mean"),
        MaxOrderValue=("NetRevenue", "max"),
        TotalQty=("Quantity", "sum"),
        WeekendOrders=("IsWeekend", "sum"),
        NightOrders=("IsNight", "sum"),
        DiscountOrders=("DiscountPct", lambda x: (x > 0).sum()),
        ReturnCount=("Returned", "sum"),
        ProductDiversity=("ProductID", "nunique"),
        CategoryDiversity=("ProductCategory", "nunique"),
        TotalOrders=("InvoiceID", "nunique"),
        FirstPurchase=("InvoiceDate", "min"),
        LastPurchase=("InvoiceDate", "max"),
    ).reset_index()

    cust["AvgBasketSize"] = cust["Monetary"] / cust["TotalOrders"]
    cust["PurchaseInterval"] = (
        (cust["LastPurchase"] - cust["FirstPurchase"]).dt.days
        / cust["Frequency"].clip(lower=2).sub(1)
    ).fillna(0).clip(lower=0)
    cust["WeekendRatio"] = cust["WeekendOrders"] / cust["TotalOrders"]
    cust["NightRatio"] = cust["NightOrders"] / cust["TotalOrders"]
    cust["DiscountUsage"] = cust["DiscountOrders"] / cust["TotalOrders"]
    cust["ReturnRate"] = cust["ReturnCount"] / cust["TotalOrders"]

    return cust


FEATURE_COLS = [
    "Recency", "Frequency", "Monetary", "AvgBasketSize",
    "PurchaseInterval", "WeekendRatio", "NightRatio",
    "DiscountUsage", "ReturnRate", "ProductDiversity",
    "CategoryDiversity", "AvgQuantity", "MaxOrderValue", "TotalQty",
]
