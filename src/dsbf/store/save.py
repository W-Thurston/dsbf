# dsbf/store/save.py
"""Store a run in the repository under ``DSBF_HOME``."""

from pathlib import Path

from dsbf.model import Run
from dsbf.store.sqlite import DEFAULT_SNAPSHOT_ROWS, open_repository


def save(
    run: Run, *, home: Path | None = None, snapshot_rows: int = DEFAULT_SNAPSHOT_ROWS
) -> Path:
    """Keep ``run``, its artifacts and a snapshot of its data; return the database.

    ``home`` is the directory that holds ``dsbf.db`` (default ``DSBF_HOME``).
    ``snapshot_rows`` is the ``store.snapshot_rows`` setting: the most rows of the
    analysed data to keep, and 0 keeps none.

    Raises:
        RunExistsError: a run with this id is stored.
    """
    repository = open_repository(home, snapshot_rows=snapshot_rows)
    try:
        repository.save(run)
        return repository.database
    finally:
        repository.close()
