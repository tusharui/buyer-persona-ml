from datetime import datetime
from sqlalchemy import String, Integer, Float, Boolean, DateTime, ForeignKey, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from typing import Optional


class Base(DeclarativeBase):
    pass


class Customer(Base):
    __tablename__ = "customers"

    customer_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    transactions: Mapped[list["Transaction"]] = relationship(back_populates="customer", cascade="all, delete-orphan")
    feature: Mapped[Optional["CustomerFeature"]] = relationship(back_populates="customer", uselist=False, cascade="all, delete-orphan")


class Transaction(Base):
    __tablename__ = "transactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    invoice_id: Mapped[str] = mapped_column(String(20), index=True)
    customer_id: Mapped[str] = mapped_column(String(20), ForeignKey("customers.customer_id"), index=True)
    invoice_date: Mapped[datetime] = mapped_column(DateTime, index=True)
    product_category: Mapped[str] = mapped_column(String(50))
    product_id: Mapped[str] = mapped_column(String(20))
    quantity: Mapped[int] = mapped_column(Integer)
    unit_price: Mapped[float] = mapped_column(Float)
    discount_pct: Mapped[float] = mapped_column(Float, default=0.0)
    payment_method: Mapped[str] = mapped_column(String(20))
    returned: Mapped[bool] = mapped_column(Boolean, default=False)

    customer: Mapped["Customer"] = relationship(back_populates="transactions")


class CustomerFeature(Base):
    __tablename__ = "customer_features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    customer_id: Mapped[str] = mapped_column(String(20), ForeignKey("customers.customer_id"), unique=True, index=True)

    recency: Mapped[float] = mapped_column(Float)
    frequency: Mapped[float] = mapped_column(Float)
    monetary: Mapped[float] = mapped_column(Float)
    avg_basket_size: Mapped[float] = mapped_column(Float)
    purchase_interval: Mapped[float] = mapped_column(Float)
    weekend_ratio: Mapped[float] = mapped_column(Float)
    night_ratio: Mapped[float] = mapped_column(Float)
    discount_usage: Mapped[float] = mapped_column(Float)
    return_rate: Mapped[float] = mapped_column(Float)
    product_diversity: Mapped[float] = mapped_column(Float)
    category_diversity: Mapped[float] = mapped_column(Float)
    avg_quantity: Mapped[float] = mapped_column(Float)
    max_order_value: Mapped[float] = mapped_column(Float)
    total_qty: Mapped[float] = mapped_column(Float)

    cluster: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    persona: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    customer: Mapped["Customer"] = relationship(back_populates="feature")

    __table_args__ = (
        UniqueConstraint("customer_id", name="uq_customer_features_customer_id"),
    )
