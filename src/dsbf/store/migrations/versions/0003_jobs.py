"""Profiling jobs and their progress events.

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("job_id", sa.Text, primary_key=True),
        sa.Column("state", sa.Text, nullable=False),
        sa.Column("requested_at", sa.Text, nullable=False),
        sa.Column("started_at", sa.Text),
        sa.Column("finished_at", sa.Text),
        sa.Column("run_id", sa.Text),
        sa.Column("error", sa.Text),
        sa.Column("last_seq", sa.Integer, nullable=False),
        sa.Column("request_json", sa.Text, nullable=False),
    )
    op.create_index("ix_jobs_requested_at", "jobs", ["requested_at"])
    op.create_table(
        "job_events",
        sa.Column(
            "job_id",
            sa.Text,
            sa.ForeignKey("jobs.job_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("seq", sa.Integer, primary_key=True),
        sa.Column("event_json", sa.Text, nullable=False),
    )


def downgrade() -> None:
    op.drop_table("job_events")
    op.drop_index("ix_jobs_requested_at", table_name="jobs")
    op.drop_table("jobs")
