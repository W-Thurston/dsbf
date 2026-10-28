# dsbf/store/sqlite.py
"""The run repository on SQLite, through SQLAlchemy Core."""

import json
import shutil
import threading
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import sqlalchemy as sa
from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory

from dsbf.engine.artifacts import encode
from dsbf.model import CheckResult, Dimension, Finding, Run, Severity
from dsbf.paths import dsbf_home
from dsbf.sinks.files import artifact_path
from dsbf.store import tables as t
from dsbf.store.repository import (
    FindingOccurrence,
    Page,
    RunExistsError,
    RunNotFoundError,
    RunSummary,
    StoreError,
)

DATABASE_FILE = "dsbf.db"
ARTIFACTS_DIR = "artifacts"
_MIGRATIONS = Path(__file__).parent / "migrations"
_BUSY_TIMEOUT_MS = 30_000
_SEVERITIES = list(Severity)


def database_path(home: Path | None = None) -> Path:
    """``<DSBF_HOME>/dsbf.db``."""
    return (home or dsbf_home()) / DATABASE_FILE


def open_repository(home: Path | None = None) -> "SqliteRunRepository":
    """The repository under ``home`` (default ``DSBF_HOME``), created if absent."""
    home = home or dsbf_home()
    return SqliteRunRepository(database_path(home), home / ARTIFACTS_DIR)


def _text(moment: datetime) -> str:
    """UTC ISO text, which sorts as time does."""
    return moment.astimezone(UTC).isoformat()


def _rank(severity: Severity) -> int:
    return _SEVERITIES.index(severity)


def _dump(model: Any, **options: Any) -> str:
    return json.dumps(model.model_dump(mode="json", **options), ensure_ascii=False)


class SqliteRunRepository:
    """Runs in a SQLite file in WAL mode, with one writer at a time.

    Opening applies the migrations. A database written by a newer DSBF is refused
    rather than read wrongly. Artifact payloads are kept as files under
    ``artifacts_dir/<run_id>``.
    """

    def __init__(self, database: Path, artifacts_dir: Path) -> None:
        database.parent.mkdir(parents=True, exist_ok=True)
        self._artifacts_dir = artifacts_dir
        self._write_lock = threading.Lock()
        self._engine = sa.create_engine(f"sqlite:///{database}")
        sa.event.listen(self._engine, "connect", _configure_connection)
        self._migrate()

    def close(self) -> None:
        """Release the database connections."""
        self._engine.dispose()

    def _migrate(self) -> None:
        config = Config()
        config.set_main_option("script_location", str(_MIGRATIONS))
        script = ScriptDirectory.from_config(config)
        with self._engine.begin() as connection:
            current = MigrationContext.configure(connection).get_current_revision()
            known = {revision.revision for revision in script.walk_revisions()}
            if current is not None and current not in known:
                raise StoreError(
                    f"the database is at revision {current}, which this DSBF does not "
                    f"know (it has {sorted(known)}); it was written by a newer DSBF"
                )
            config.attributes["connection"] = connection
            command.upgrade(config, "head")

    # -- writing ---------------------------------------------------------------------

    def save(self, run: Run) -> None:
        run_id = str(run.run_id)
        key = run.dataset.key or None
        directory = self._artifacts_dir / run_id
        with self._write_lock:
            with self._engine.connect() as connection:
                if connection.execute(
                    sa.select(t.runs.c.run_id).where(t.runs.c.run_id == run_id)
                ).first():
                    raise RunExistsError(f"run {run_id} is already stored")
            try:
                self._write_artifacts(run, directory)
                with self._engine.begin() as connection:
                    self._insert(connection, run, run_id, key)
            except BaseException:
                shutil.rmtree(directory, ignore_errors=True)
                raise

    @staticmethod
    def _write_artifacts(run: Run, directory: Path) -> None:
        for artifact in run.artifacts:
            target = artifact_path(directory, artifact)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(encode(run.artifact(artifact.ref))[0])

    @staticmethod
    def _insert(
        connection: sa.Connection, run: Run, run_id: str, key: str | None
    ) -> None:
        findings = list(run.findings)
        if key is not None:
            connection.execute(
                sa.insert(t.datasets)
                .prefix_with("OR IGNORE")
                .values(
                    key=key,
                    name=run.dataset.name,
                    created_at=_text(run.started_at),
                )
            )
            connection.execute(
                sa.update(t.datasets)
                .where(t.datasets.c.key == key)
                .values(name=run.dataset.name)
            )
        connection.execute(
            sa.insert(t.runs).values(
                run_id=run_id,
                dataset_key=key,
                dataset_name=run.dataset.name,
                started_at=_text(run.started_at),
                finished_at=_text(run.finished_at),
                n_rows=run.dataset.n_rows,
                n_columns=run.dataset.n_columns,
                n_findings=len(findings),
                max_severity_rank=(
                    max(_rank(f.severity) for f in findings) if findings else None
                ),
                header_json=_dump(run, exclude={"results", "profile"}),
            )
        )
        connection.execute(
            sa.insert(t.profiles).values(run_id=run_id, profile_json=_dump(run.profile))
        )
        result_rows, finding_rows, column_rows = [], [], []
        for position, result in enumerate(run.results.values()):
            result_rows.append(
                {
                    "run_id": run_id,
                    "check_id": result.check_id,
                    "position": position,
                    "status": result.status.value,
                    "result_json": _dump(result, exclude={"findings"}),
                }
            )
            for index, finding in enumerate(result.findings):
                finding_rows.append(
                    {
                        "run_id": run_id,
                        "id": finding.id,
                        "check_id": finding.check_id,
                        "position": index,
                        "code": finding.code,
                        "dimension": finding.dimension.value,
                        "severity_rank": _rank(finding.severity),
                        "finding_json": _dump(finding),
                    }
                )
                column_rows.extend(
                    {"run_id": run_id, "finding_id": finding.id, "column": column}
                    for column in dict.fromkeys(finding.subject.columns)
                )
        connection.execute(sa.insert(t.check_results), result_rows)
        if finding_rows:
            connection.execute(sa.insert(t.findings), finding_rows)
        if column_rows:
            connection.execute(sa.insert(t.finding_columns), column_rows)

    def delete(self, run_id: UUID) -> None:
        with self._write_lock, self._engine.begin() as connection:
            deleted = connection.execute(
                sa.delete(t.runs).where(t.runs.c.run_id == str(run_id))
            )
            if deleted.rowcount == 0:
                raise RunNotFoundError(f"no stored run {run_id}")
        shutil.rmtree(self._artifacts_dir / str(run_id), ignore_errors=True)

    # -- reading ---------------------------------------------------------------------

    def get(self, run_id: UUID) -> Run:
        key = str(run_id)
        with self._engine.connect() as connection:
            header = connection.execute(
                sa.select(t.runs.c.header_json).where(t.runs.c.run_id == key)
            ).scalar_one_or_none()
            if header is None:
                raise RunNotFoundError(f"no stored run {run_id}")
            profile = connection.execute(
                sa.select(t.profiles.c.profile_json).where(t.profiles.c.run_id == key)
            ).scalar_one()
            results = connection.execute(
                sa.select(t.check_results.c.check_id, t.check_results.c.result_json)
                .where(t.check_results.c.run_id == key)
                .order_by(t.check_results.c.position)
            ).all()
            found = defaultdict(list)
            for check_id, text in connection.execute(
                sa.select(t.findings.c.check_id, t.findings.c.finding_json)
                .where(t.findings.c.run_id == key)
                .order_by(t.findings.c.check_id, t.findings.c.position)
            ):
                found[check_id].append(json.loads(text))
        fields = json.loads(header)
        fields["profile"] = json.loads(profile)
        fields["results"] = {
            check_id: {**json.loads(text), "findings": found[check_id]}
            for check_id, text in results
        }
        run = Run.model_validate(fields)
        directory = self._artifacts_dir / key
        if directory.is_dir():
            run._artifact_dir = directory  # noqa: SLF001 (the run's own lookup)
        return run

    def get_result(self, run_id: UUID, check_id: str) -> CheckResult:
        key = str(run_id)
        with self._engine.connect() as connection:
            text = connection.execute(
                sa.select(t.check_results.c.result_json).where(
                    t.check_results.c.run_id == key,
                    t.check_results.c.check_id == check_id,
                )
            ).scalar_one_or_none()
            if text is None:
                raise RunNotFoundError(f"run {run_id} has no result for {check_id}")
            rows = connection.execute(
                sa.select(t.findings.c.finding_json)
                .where(t.findings.c.run_id == key, t.findings.c.check_id == check_id)
                .order_by(t.findings.c.position)
            ).scalars()
            return CheckResult.model_validate(
                {**json.loads(text), "findings": [json.loads(r) for r in rows]}
            )

    def get_finding(self, run_id: UUID, finding_id: str) -> Finding:
        with self._engine.connect() as connection:
            text = connection.execute(
                sa.select(t.findings.c.finding_json).where(
                    t.findings.c.run_id == str(run_id), t.findings.c.id == finding_id
                )
            ).scalar_one_or_none()
        if text is None:
            raise RunNotFoundError(f"run {run_id} has no finding {finding_id}")
        return Finding.model_validate_json(text)

    def list_runs(
        self, *, dataset_key: str | None = None, limit: int = 50, offset: int = 0
    ) -> Page[RunSummary]:
        where = [t.runs.c.dataset_key == dataset_key] if dataset_key is not None else []
        with self._engine.connect() as connection:
            total = connection.execute(
                sa.select(sa.func.count()).select_from(t.runs).where(*where)
            ).scalar_one()
            rows = connection.execute(
                sa.select(t.runs)
                .where(*where)
                .order_by(t.runs.c.started_at.desc(), t.runs.c.run_id)
                .limit(limit)
                .offset(offset)
            ).mappings()
            items = tuple(
                RunSummary(
                    run_id=UUID(row["run_id"]),
                    dataset_key=row["dataset_key"],
                    dataset_name=row["dataset_name"],
                    started_at=datetime.fromisoformat(row["started_at"]),
                    finished_at=datetime.fromisoformat(row["finished_at"]),
                    n_rows=row["n_rows"],
                    n_columns=row["n_columns"],
                    n_findings=row["n_findings"],
                    max_severity=(
                        None
                        if row["max_severity_rank"] is None
                        else _SEVERITIES[row["max_severity_rank"]]
                    ),
                )
                for row in rows
            )
        return Page[RunSummary](items=items, total=total, limit=limit, offset=offset)

    def find_findings(
        self,
        run_id: UUID,
        *,
        dimension: Dimension | None = None,
        min_severity: Severity | None = None,
        code: str | None = None,
        column: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Page[Finding]:
        key = str(run_id)
        where = [t.findings.c.run_id == key]
        if dimension is not None:
            where.append(t.findings.c.dimension == dimension.value)
        if min_severity is not None:
            where.append(t.findings.c.severity_rank >= _rank(min_severity))
        if code is not None:
            where.append(t.findings.c.code == code)
        if column is not None:
            where.append(
                t.findings.c.id.in_(
                    sa.select(t.finding_columns.c.finding_id).where(
                        t.finding_columns.c.run_id == key,
                        t.finding_columns.c.column == column,
                    )
                )
            )
        with self._engine.connect() as connection:
            if not connection.execute(
                sa.select(t.runs.c.run_id).where(t.runs.c.run_id == key)
            ).first():
                raise RunNotFoundError(f"no stored run {run_id}")
            total = connection.execute(
                sa.select(sa.func.count()).select_from(t.findings).where(*where)
            ).scalar_one()
            texts = connection.execute(
                sa.select(t.findings.c.finding_json)
                .where(*where)
                .order_by(
                    t.findings.c.severity_rank.desc(),
                    t.findings.c.check_id,
                    t.findings.c.id,
                )
                .limit(limit)
                .offset(offset)
            ).scalars()
            items = tuple(Finding.model_validate_json(text) for text in texts)
        return Page[Finding](items=items, total=total, limit=limit, offset=offset)

    def finding_history(
        self,
        dataset_key: str,
        *,
        finding_id: str | None = None,
        code: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> Page[FindingOccurrence]:
        if finding_id is None and code is None:
            raise ValueError("give a finding id or a code")
        where = [t.runs.c.dataset_key == dataset_key]
        if finding_id is not None:
            where.append(t.findings.c.id == finding_id)
        if code is not None:
            where.append(t.findings.c.code == code)
        joined = t.findings.join(t.runs, t.findings.c.run_id == t.runs.c.run_id)
        with self._engine.connect() as connection:
            total = connection.execute(
                sa.select(sa.func.count()).select_from(joined).where(*where)
            ).scalar_one()
            rows = connection.execute(
                sa.select(
                    t.runs.c.run_id, t.runs.c.started_at, t.findings.c.finding_json
                )
                .select_from(joined)
                .where(*where)
                .order_by(
                    t.runs.c.started_at.desc(),
                    t.runs.c.run_id,
                    t.findings.c.severity_rank.desc(),
                    t.findings.c.check_id,
                    t.findings.c.id,
                )
                .limit(limit)
                .offset(offset)
            ).all()
        items = tuple(
            FindingOccurrence(
                run_id=UUID(run_key),
                started_at=datetime.fromisoformat(started),
                finding=Finding.model_validate_json(text),
            )
            for run_key, started, text in rows
        )
        return Page[FindingOccurrence](
            items=items, total=total, limit=limit, offset=offset
        )


def _configure_connection(dbapi_connection: Any, _record: Any) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    cursor.close()
