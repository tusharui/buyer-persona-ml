from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = '613d0d2bf62a'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table('customers',
    sa.Column('customer_id', sa.String(length=20), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.PrimaryKeyConstraint('customer_id')
    )
    op.create_table('customer_features',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('customer_id', sa.String(length=20), nullable=False),
    sa.Column('recency', sa.Float(), nullable=False),
    sa.Column('frequency', sa.Float(), nullable=False),
    sa.Column('monetary', sa.Float(), nullable=False),
    sa.Column('avg_basket_size', sa.Float(), nullable=False),
    sa.Column('purchase_interval', sa.Float(), nullable=False),
    sa.Column('weekend_ratio', sa.Float(), nullable=False),
    sa.Column('night_ratio', sa.Float(), nullable=False),
    sa.Column('discount_usage', sa.Float(), nullable=False),
    sa.Column('return_rate', sa.Float(), nullable=False),
    sa.Column('product_diversity', sa.Float(), nullable=False),
    sa.Column('category_diversity', sa.Float(), nullable=False),
    sa.Column('avg_quantity', sa.Float(), nullable=False),
    sa.Column('max_order_value', sa.Float(), nullable=False),
    sa.Column('total_qty', sa.Float(), nullable=False),
    sa.Column('cluster', sa.Integer(), nullable=True),
    sa.Column('persona', sa.String(length=50), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.ForeignKeyConstraint(['customer_id'], ['customers.customer_id'], ),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('customer_id', name='uq_customer_features_customer_id')
    )
    op.create_index(op.f('ix_customer_features_customer_id'), 'customer_features', ['customer_id'], unique=True)
    op.create_table('transactions',
    sa.Column('id', sa.Integer(), autoincrement=True, nullable=False),
    sa.Column('invoice_id', sa.String(length=20), nullable=False),
    sa.Column('customer_id', sa.String(length=20), nullable=False),
    sa.Column('invoice_date', sa.DateTime(), nullable=False),
    sa.Column('product_category', sa.String(length=50), nullable=False),
    sa.Column('product_id', sa.String(length=20), nullable=False),
    sa.Column('quantity', sa.Integer(), nullable=False),
    sa.Column('unit_price', sa.Float(), nullable=False),
    sa.Column('discount_pct', sa.Float(), nullable=False),
    sa.Column('payment_method', sa.String(length=20), nullable=False),
    sa.Column('returned', sa.Boolean(), nullable=False),
    sa.ForeignKeyConstraint(['customer_id'], ['customers.customer_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_transactions_customer_id'), 'transactions', ['customer_id'], unique=False)
    op.create_index(op.f('ix_transactions_invoice_date'), 'transactions', ['invoice_date'], unique=False)
    op.create_index(op.f('ix_transactions_invoice_id'), 'transactions', ['invoice_id'], unique=False)


def downgrade() -> None:
    op.drop_index(op.f('ix_transactions_invoice_id'), table_name='transactions')
    op.drop_index(op.f('ix_transactions_invoice_date'), table_name='transactions')
    op.drop_index(op.f('ix_transactions_customer_id'), table_name='transactions')
    op.drop_table('transactions')
    op.drop_index(op.f('ix_customer_features_customer_id'), table_name='customer_features')
    op.drop_table('customer_features')
    op.drop_table('customers')
