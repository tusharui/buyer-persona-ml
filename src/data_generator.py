import sys
import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text
from src.database import AsyncSessionLocal
from src.models import Customer, Transaction

PRODUCT_CATEGORIES = ["Grocery", "Books", "Sports", "Electronics", "Home", "Fashion"]
PAYMENT_METHODS = ["Cash", "NetBanking", "UPI", "Card", "Wallet"]

PRODUCT_POOL = {
    "Grocery": [f"P{i}" for i in range(900, 999)],
    "Books": [f"P{i}" for i in range(700, 799)],
    "Sports": [f"P{i}" for i in range(500, 599)],
    "Electronics": [f"P{i}" for i in range(300, 399)],
    "Home": [f"P{i}" for i in range(100, 199)],
    "Fashion": [f"P{i}" for i in range(1, 99)],
}


def _random_date(start: datetime, end: datetime) -> datetime:
    delta = end - start
    return start + timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def _generate_customers(n: int) -> list[Customer]:
    return [
        Customer(customer_id=f"C{1000 + i}")
        for i in range(1, n + 1)
    ]


def _generate_transactions(customer_ids: list[str], n: int) -> list[Transaction]:
    start = datetime(2024, 1, 1)
    end = datetime(2025, 12, 31)
    transactions = []
    invoice_base = 100000
    weights = [0.2, 0.25, 0.25, 0.15, 0.1, 0.05]
    for i in range(n):
        cust_id = random.choice(customer_ids)
        cat = random.choices(PRODUCT_CATEGORIES, weights=weights, k=1)[0]
        transactions.append(
            Transaction(
                invoice_id=f"INV{invoice_base + i}",
                customer_id=cust_id,
                invoice_date=_random_date(start, end),
                product_category=cat,
                product_id=random.choice(PRODUCT_POOL[cat]),
                quantity=random.randint(1, 5),
                unit_price=float(random.choice([199, 299, 499, 999, 1499, 2499, 3999, 4999])),
                discount_pct=float(random.choice([0, 5, 10, 15, 20, 25, 30])),
                payment_method=random.choice(PAYMENT_METHODS),
                returned=random.random() < 0.2,
            )
        )
    return transactions


async def generate_data(n_customers: int = 1000, n_transactions: int = 10000, clear: bool = False):
    if clear:
        async with AsyncSessionLocal() as session:
            await session.execute(text("TRUNCATE TABLE transactions, customer_features, customers CASCADE"))
            await session.commit()
        print("Cleared existing data.")

    customers = _generate_customers(n_customers)
    customer_ids = [c.customer_id for c in customers]
    transactions = _generate_transactions(customer_ids, n_transactions)

    batch_size = 500
    async with AsyncSessionLocal() as session:
        for i in range(0, len(customers), batch_size):
            batch = customers[i:i + batch_size]
            session.add_all(batch)
        await session.commit()
    print(f"Inserted {len(customers)} customers.")

    async with AsyncSessionLocal() as session:
        for i in range(0, len(transactions), batch_size):
            batch = transactions[i:i + batch_size]
            session.add_all(batch)
        await session.commit()
    print(f"Inserted {len(transactions)} transactions.")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic transaction data into Neon.")
    parser.add_argument("--customers", type=int, default=1000, help="Number of customers")
    parser.add_argument("--transactions", type=int, default=10000, help="Number of transactions")
    parser.add_argument("--clear", action="store_true", help="Truncate existing data before generating")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    import asyncio
    asyncio.run(generate_data(args.customers, args.transactions, args.clear))

    print("Done.")


if __name__ == "__main__":
    main()
