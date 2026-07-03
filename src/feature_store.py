import sys, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.database import AsyncSessionLocal
from src.models import CustomerFeature
from src.features import build_customer_features, FEATURE_COLS
from src.cache import cache


COLUMN_MAP = {
    "Recency": "recency",
    "Frequency": "frequency",
    "Monetary": "monetary",
    "AvgBasketSize": "avg_basket_size",
    "PurchaseInterval": "purchase_interval",
    "WeekendRatio": "weekend_ratio",
    "NightRatio": "night_ratio",
    "DiscountUsage": "discount_usage",
    "ReturnRate": "return_rate",
    "ProductDiversity": "product_diversity",
    "CategoryDiversity": "category_diversity",
    "AvgQuantity": "avg_quantity",
    "MaxOrderValue": "max_order_value",
    "TotalQty": "total_qty",
}


async def load_transactions(session: AsyncSession) -> pd.DataFrame:
    query = text("""
        SELECT invoice_id, customer_id, invoice_date, product_category,
               product_id, quantity, unit_price, discount_pct,
               payment_method, returned
        FROM transactions
        ORDER BY invoice_date
    """)
    result = await session.execute(query)
    rows = result.fetchall()
    df = pd.DataFrame(rows, columns=[
        "InvoiceID", "CustomerID", "InvoiceDate", "ProductCategory",
        "ProductID", "Quantity", "UnitPrice", "DiscountPct",
        "PaymentMethod", "Returned",
    ])
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    return df


async def compute_and_store_features():
    await cache.connect()

    async with AsyncSessionLocal() as session:
        print("Loading transactions from Neon...")
        df = await load_transactions(session)
        print(f"  Loaded {len(df)} transactions for {df['CustomerID'].nunique()} customers.")

        print("Computing customer features...")
        features_df = build_customer_features(df)
        print(f"  Computed {len(features_df)} customer profiles, {len(FEATURE_COLS)} features each.")

        print("Storing features in Neon...")
        existing = await session.execute(text("SELECT customer_id FROM customer_features"))
        existing_ids = {row[0] for row in existing}

        new_count = 0
        update_count = 0
        for _, row in features_df.iterrows():
            cid = row["CustomerID"]
            kwargs = {"customer_id": cid}
            for df_col, db_col in COLUMN_MAP.items():
                kwargs[db_col] = float(row[df_col]) if pd.notna(row[df_col]) else 0.0

            if cid in existing_ids:
                await session.execute(
                    text(f"""
                        UPDATE customer_features SET
                            {', '.join(f'{col} = :{col}' for col in COLUMN_MAP.values())},
                            updated_at = NOW()
                        WHERE customer_id = :customer_id
                    """),
                    kwargs
                )
                update_count += 1
            else:
                session.add(CustomerFeature(**kwargs))
                new_count += 1

        await session.commit()
        print(f"  Inserted {new_count}, updated {update_count} feature records.")

    print("Caching features in Redis...")
    async with AsyncSessionLocal() as session:
        feats = await session.execute(text("SELECT * FROM customer_features"))
        for row in feats:
            key = f"features:{row.customer_id}"
            payload = {
                col: getattr(row, col)
                for col in COLUMN_MAP.values()
            }
            import json
            await cache.set(key, json.dumps(payload), ttl=cache.feature_cache_ttl)
    print(f"  Cached {new_count + update_count} feature vectors in Redis.")

    await cache.close()
    print("Done.")


if __name__ == "__main__":
    asyncio.run(compute_and_store_features())
