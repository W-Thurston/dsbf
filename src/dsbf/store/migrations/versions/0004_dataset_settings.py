"""Per-dataset settings overlays.

Revision ID: 0004
Revises: 0003
"""

import sqlalchemy as sa
from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dataset_settings",
        sa.Column("dataset_key", sa.Text, primary_key=True),
        sa.Column("overlay_json", sa.Text, nullable=False),
        sa.Column("updated_at", sa.Text, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("dataset_settings")
