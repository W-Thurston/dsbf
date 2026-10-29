# dsbf/store/tables.py
"""The store's tables. Revision scripts under ``migrations/`` build them, so a
change here needs a new revision."""

import sqlalchemy as sa

metadata = sa.MetaData()

datasets = sa.Table(
    "datasets",
    metadata,
    sa.Column("key", sa.Text, primary_key=True),
    sa.Column("name", sa.Text, nullable=False),
    sa.Column("created_at", sa.Text, nullable=False),
)

runs = sa.Table(
    "runs",
    metadata,
    sa.Column("run_id", sa.Text, primary_key=True),
    sa.Column("dataset_key", sa.Text, sa.ForeignKey("datasets.key"), nullable=True),
    sa.Column("dataset_name", sa.Text, nullable=False),
    sa.Column("started_at", sa.Text, nullable=False),
    sa.Column("finished_at", sa.Text, nullable=False),
    sa.Column("n_rows", sa.Integer, nullable=False),
    sa.Column("n_columns", sa.Integer, nullable=False),
    sa.Column("n_findings", sa.Integer, nullable=False),
    sa.Column("max_severity_rank", sa.Integer, nullable=True),
    sa.Column("header_json", sa.Text, nullable=False),
    sa.Index("ix_runs_dataset_key_started_at", "dataset_key", "started_at"),
    sa.Index("ix_runs_started_at", "started_at"),
)

profiles = sa.Table(
    "profiles",
    metadata,
    sa.Column(
        "run_id",
        sa.Text,
        sa.ForeignKey("runs.run_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column("profile_json", sa.Text, nullable=False),
)

check_results = sa.Table(
    "check_results",
    metadata,
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

findings = sa.Table(
    "findings",
    metadata,
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
    sa.Index("ix_findings_run_id_severity_rank", "run_id", "severity_rank"),
    sa.Index("ix_findings_id", "id"),
    sa.Index("ix_findings_code", "code"),
)

finding_columns = sa.Table(
    "finding_columns",
    metadata,
    sa.Column("run_id", sa.Text, primary_key=True),
    sa.Column("finding_id", sa.Text, primary_key=True),
    sa.Column("column", sa.Text, primary_key=True),
    sa.ForeignKeyConstraint(
        ["run_id", "finding_id"],
        ["findings.run_id", "findings.id"],
        ondelete="CASCADE",
    ),
    sa.Index("ix_finding_columns_column", "column"),
)

artifacts = sa.Table(
    "artifacts",
    metadata,
    sa.Column(
        "run_id",
        sa.Text,
        sa.ForeignKey("runs.run_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column("ref", sa.Text, primary_key=True),
    sa.Column("path", sa.Text, nullable=False),
    sa.Column("media_type", sa.Text, nullable=False),
    sa.Column("rows", sa.Integer, nullable=True),
    sa.Column("bytes", sa.Integer, nullable=False),
    sa.Column("sha256", sa.Text, nullable=False),
    sa.Column("sampling_json", sa.Text, nullable=True),
)

jobs = sa.Table(
    "jobs",
    metadata,
    sa.Column("job_id", sa.Text, primary_key=True),
    sa.Column("state", sa.Text, nullable=False),
    sa.Column("requested_at", sa.Text, nullable=False),
    sa.Column("started_at", sa.Text, nullable=True),
    sa.Column("finished_at", sa.Text, nullable=True),
    sa.Column("run_id", sa.Text, nullable=True),
    sa.Column("error", sa.Text, nullable=True),
    sa.Column("last_seq", sa.Integer, nullable=False),
    sa.Column("request_json", sa.Text, nullable=False),
    sa.Index("ix_jobs_requested_at", "requested_at"),
)

job_events = sa.Table(
    "job_events",
    metadata,
    sa.Column(
        "job_id",
        sa.Text,
        sa.ForeignKey("jobs.job_id", ondelete="CASCADE"),
        primary_key=True,
    ),
    sa.Column("seq", sa.Integer, primary_key=True),
    sa.Column("event_json", sa.Text, nullable=False),
)

dataset_settings = sa.Table(
    "dataset_settings",
    metadata,
    sa.Column("dataset_key", sa.Text, primary_key=True),
    sa.Column("overlay_json", sa.Text, nullable=False),
    sa.Column("updated_at", sa.Text, nullable=False),
)
