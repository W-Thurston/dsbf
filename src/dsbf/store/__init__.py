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
from dsbf.store.repository import (
    SNAPSHOT_REF,
    GcResult,
    Page,
    RunExistsError,
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
    "FsspecArtifactStore",
    "GcResult",
    "Snapshot",
    "StoredArtifact",
    "Page",
    "RunExistsError",
    "RunNotFoundError",
    "RunRepository",
    "RunSummary",
    "SqliteRunRepository",
    "StoreError",
    "database_path",
    "open_repository",
    "save",
]
