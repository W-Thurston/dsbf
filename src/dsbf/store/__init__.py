# dsbf/store/__init__.py
"""The run repository: stored runs, their findings and their history.

Needs the ``server`` extra. The core never imports this package.
"""

try:
    import alembic  # noqa: F401
    import sqlalchemy  # noqa: F401
except ImportError as error:
    raise ImportError(
        "dsbf.store needs the `server` extra. Install it with "
        "`pip install 'dsbf[server]'`."
    ) from error

from dsbf.store.artifacts import (
    ArtifactNotFoundError,
    ArtifactStore,
    FsspecArtifactStore,
)
from dsbf.store.jobs import Job, JobEvent, JobRequest, JobState
from dsbf.store.repository import (
    SNAPSHOT_REF,
    DatasetSummary,
    FindingCount,
    FindingOccurrence,
    FindingSort,
    GcResult,
    JobNotFoundError,
    Page,
    RunExistsError,
    RunHeader,
    RunNotFoundError,
    RunRepository,
    RunSummary,
    Snapshot,
    StoredArtifact,
    StoreError,
)
from dsbf.store.save import save
from dsbf.store.sqlite import SqliteRunRepository, database_path, open_repository

__all__ = [
    "SNAPSHOT_REF",
    "ArtifactNotFoundError",
    "ArtifactStore",
    "DatasetSummary",
    "FindingCount",
    "FindingOccurrence",
    "FindingSort",
    "FsspecArtifactStore",
    "GcResult",
    "Job",
    "JobEvent",
    "JobNotFoundError",
    "JobRequest",
    "JobState",
    "Snapshot",
    "StoredArtifact",
    "Page",
    "RunExistsError",
    "RunHeader",
    "RunNotFoundError",
    "RunRepository",
    "RunSummary",
    "SqliteRunRepository",
    "StoreError",
    "database_path",
    "open_repository",
    "save",
]
