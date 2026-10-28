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

from dsbf.store.repository import (
    Page,
    RunExistsError,
    RunNotFoundError,
    RunRepository,
    RunSummary,
    StoreError,
)
from dsbf.store.sqlite import SqliteRunRepository, database_path, open_repository

__all__ = [
    "Page",
    "RunExistsError",
    "RunNotFoundError",
    "RunRepository",
    "RunSummary",
    "SqliteRunRepository",
    "StoreError",
    "database_path",
    "open_repository",
]
