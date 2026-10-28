"""Artifacts and the data snapshot of each run.

Revision ID: 0002
Revises: 0001
"""

import sqlalchemy as sa
from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "artifacts",
        sa.Column(
            "run_id",
            sa.Text,
            sa.ForeignKey("runs.run_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("ref", sa.Text, primary_key=True),
        sa.Column("path", sa.Text, nullable=False),
        sa.Column("media_type", sa.Text, nullable=False),
        sa.Column("rows", sa.Integer),
        sa.Column("bytes", sa.Integer, nullable=False),
        sa.Column("sha256", sa.Text, nullable=False),
        sa.Column("sampling_json", sa.Text),
    )


def downgrade() -> None:
    op.drop_table("artifacts")
