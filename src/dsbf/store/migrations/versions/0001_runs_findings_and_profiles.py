"""Runs, check results, findings and profiles.

Revision ID: 0001
Revises:
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("key", sa.Text, primary_key=True),
        sa.Column("name", sa.Text, nullable=False),
        sa.Column("created_at", sa.Text, nullable=False),
    )
    op.create_table(
        "runs",
        sa.Column("run_id", sa.Text, primary_key=True),
        sa.Column("dataset_key", sa.Text, sa.ForeignKey("datasets.key")),
        sa.Column("dataset_name", sa.Text, nullable=False),
        sa.Column("started_at", sa.Text, nullable=False),
        sa.Column("finished_at", sa.Text, nullable=False),
        sa.Column("n_rows", sa.Integer, nullable=False),
        sa.Column("n_columns", sa.Integer, nullable=False),
        sa.Column("n_findings", sa.Integer, nullable=False),
        sa.Column("max_severity_rank", sa.Integer),
        sa.Column("header_json", sa.Text, nullable=False),
    )
    op.create_index(
        "ix_runs_dataset_key_started_at", "runs", ["dataset_key", "started_at"]
    )
    op.create_index("ix_runs_started_at", "runs", ["started_at"])
    op.create_table(
        "profiles",
        sa.Column(
            "run_id",
            sa.Text,
            sa.ForeignKey("runs.run_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("profile_json", sa.Text, nullable=False),
    )
    op.create_table(
        "check_results",
        sa.Column(
            "run_id",
            sa.Text,
            sa.ForeignKey("runs.run_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("check_id", sa.Text, primary_key=True),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("status", sa.Text, nullable=False),
        sa.Column("result_json", sa.Text, nullable=False),
    )
    op.create_table(
        "findings",
        sa.Column(
            "run_id",
            sa.Text,
            sa.ForeignKey("runs.run_id", ondelete="CASCADE"),
            primary_key=True,
        ),
        sa.Column("id", sa.Text, primary_key=True),
        sa.Column("check_id", sa.Text, nullable=False),
        sa.Column("position", sa.Integer, nullable=False),
        sa.Column("code", sa.Text, nullable=False),
        sa.Column("dimension", sa.Text, nullable=False),
        sa.Column("severity_rank", sa.Integer, nullable=False),
        sa.Column("finding_json", sa.Text, nullable=False),
    )
    op.create_index(
        "ix_findings_run_id_severity_rank", "findings", ["run_id", "severity_rank"]
    )
    op.create_index("ix_findings_id", "findings", ["id"])
    op.create_index("ix_findings_code", "findings", ["code"])
    op.create_table(
        "finding_columns",
        sa.Column("run_id", sa.Text, primary_key=True),
        sa.Column("finding_id", sa.Text, primary_key=True),
        sa.Column("column", sa.Text, primary_key=True),
        sa.ForeignKeyConstraint(
            ["run_id", "finding_id"],
            ["findings.run_id", "findings.id"],
            ondelete="CASCADE",
        ),
    )
    op.create_index("ix_finding_columns_column", "finding_columns", ["column"])


def downgrade() -> None:
    for name in (
        "finding_columns",
        "findings",
        "check_results",
        "profiles",
        "runs",
        "datasets",
    ):
        op.drop_table(name)
