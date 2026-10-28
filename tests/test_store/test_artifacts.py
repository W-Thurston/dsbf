"""Artifacts and the data snapshot are stored beside a run and read back."""

import hashlib
from pathlib import Path

import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import dsbf
from dsbf.engine.artifacts import ROW_GROUP_ROWS, encode
from dsbf.store import (
    SNAPSHOT_REF,
    ArtifactNotFoundError,
    FsspecArtifactStore,
    RunNotFoundError,
    SqliteRunRepository,
)
from tests.golden.datasets import KITCHEN5K

pytestmark = pytest.mark.unit

TITANIC = Path(__file__).parents[1] / "golden" / "data" / "titanic.csv"


def frame(rows: int) -> pl.DataFrame:
    return pl.DataFrame({"id": range(rows), "x": [i * 0.5 for i in range(rows)]})


def stored_snapshot(repo, run):
    snapshot = repo.snapshot(run.run_id)
    assert snapshot is not None
    return snapshot


def profile(rows: int):
    return dsbf.profile(frame(rows), depth="quick", dataset="snap")


def test_a_snapshot_that_fits_the_cap_holds_every_row_and_no_sampling(repo):
    run = profile(300)
    repo.save(run)

    snapshot = stored_snapshot(repo, run)

    assert snapshot.sampling is None
    assert snapshot.table.equals(frame(300).to_arrow())
    stored = {a.ref: a for a in repo.artifacts(run.run_id)}[SNAPSHOT_REF]
    assert stored.rows == 300 and stored.sampling is None


def test_a_snapshot_above_the_cap_is_a_seeded_sample_that_says_so(tmp_path):
    repo = SqliteRunRepository(
        tmp_path / "dsbf.db", tmp_path / "artifacts", snapshot_rows=100
    )
    try:
        run = profile(1_000)
        repo.save(run)
        snapshot = stored_snapshot(repo, run)

        sampling = snapshot.sampling
        assert sampling is not None
        assert snapshot.table.num_rows == 100
        assert sampling.strategy == "random"
        assert (sampling.seed, sampling.original_rows) == (0, 1_000)
        assert sampling.sampled_rows == 100
        ids = snapshot.table["id"].to_pylist()
        assert ids == sorted(set(ids)) and set(ids) <= set(range(1_000))
        again = profile(1_000)
        repo.save(again)
        assert stored_snapshot(repo, again).table.equals(snapshot.table)
    finally:
        repo.close()


def test_a_cap_of_zero_keeps_no_snapshot(tmp_path):
    repo = SqliteRunRepository(
        tmp_path / "dsbf.db", tmp_path / "artifacts", snapshot_rows=0
    )
    try:
        run = profile(50)
        repo.save(run)

        assert repo.snapshot(run.run_id) is None
        assert not (
            tmp_path / "artifacts" / str(run.run_id) / "snapshot.parquet"
        ).exists()
    finally:
        repo.close()


def test_a_run_read_back_has_no_frame_so_it_cannot_be_snapshotted_again(repo, tmp_path):
    run = profile(50)
    repo.save(run)
    back = repo.get(run.run_id)

    assert run.analysed_frame is not None and back.analysed_frame is None


def test_the_stored_description_matches_the_file(repo, tmp_path):
    run = profile(50)
    repo.save(run)

    stored = {a.ref: a for a in repo.artifacts(run.run_id)}[SNAPSHOT_REF]
    path = tmp_path / "artifacts" / str(run.run_id) / "snapshot.parquet"
    assert stored.bytes == path.stat().st_size
    assert stored.sha256 == hashlib.sha256(path.read_bytes()).hexdigest()


def test_titanic_artifacts_and_snapshot_read_back_equal(repo):
    run = dsbf.profile(TITANIC)
    repo.save(run)

    stored = {a.ref for a in repo.artifacts(run.run_id)}
    assert stored == {a.ref for a in run.artifacts} | {SNAPSHOT_REF}
    for artifact in run.artifacts:
        left, right = (
            run.artifact(artifact.ref),
            repo.read_artifact(run.run_id, artifact.ref),
        )
        assert left.equals(right) if hasattr(left, "equals") else left == right
    assert stored_snapshot(repo, run).table.num_rows == run.dataset.n_rows


@pytest.mark.slow
def test_kitchen_artifacts_read_whole_and_by_row_range(repo, tmp_path):
    run = dsbf.profile(
        KITCHEN5K.write_csv(tmp_path), depth="deep", time_index="normal_date"
    )
    repo.save(run)
    tables = [a for a in run.artifacts if a.rows is not None]
    assert tables

    for artifact in tables:
        whole = run.artifact(artifact.ref)
        assert artifact.rows is not None
        stop = min(artifact.rows, 7)
        part = repo.read_artifact_rows(run.run_id, artifact.ref, 2, stop)
        assert part.equals(whole.slice(2, max(stop - 2, 0)))
        assert repo.read_artifact(run.run_id, artifact.ref).equals(whole)


def test_a_range_read_returns_the_rows_asked_for_from_few_row_groups(
    tmp_path, monkeypatch
):
    rows = ROW_GROUP_ROWS * 3 + 500
    table = pa.table({"n": list(range(rows))})
    store = FsspecArtifactStore(str(tmp_path))
    store.put("r", "t.parquet", encode(table)[0])
    read = []
    original = pq.ParquetFile.read_row_group
    monkeypatch.setattr(
        pq.ParquetFile,
        "read_row_group",
        lambda self, i, *a, **k: read.append(i) or original(self, i, *a, **k),
    )

    start = ROW_GROUP_ROWS * 2 - 3
    part = store.read_rows("r", "t.parquet", start, start + 10)

    assert part["n"].to_pylist() == list(range(start, start + 10))
    assert read == [1, 2]
    assert store.read_rows("r", "t.parquet", rows + 5, rows + 9).num_rows == 0


def test_a_missing_file_is_reported_as_missing(repo, tmp_path):
    run = profile(50)
    repo.save(run)
    (tmp_path / "artifacts" / str(run.run_id) / "snapshot.parquet").unlink()

    with pytest.raises(ArtifactNotFoundError, match="snapshot.parquet"):
        repo.snapshot(run.run_id)
    with pytest.raises(ArtifactNotFoundError):
        repo.read_artifact_rows(run.run_id, SNAPSHOT_REF, 0, 5)
    assert not FsspecArtifactStore(str(tmp_path / "artifacts")).exists(
        str(run.run_id), "snapshot.parquet"
    )


def test_an_unknown_run_or_artifact_is_not_found(repo):
    run = profile(50)
    repo.save(run)

    with pytest.raises(RunNotFoundError):
        repo.read_artifact(run.run_id, "no/such")
    with pytest.raises(RunNotFoundError):
        repo.snapshot(profile(5).run_id)
    with pytest.raises(RunNotFoundError):
        repo.artifacts(profile(5).run_id)


def test_deleting_a_run_leaves_no_files_and_no_rows(repo, tmp_path):
    run = profile(50)
    repo.save(run)
    directory = tmp_path / "artifacts" / str(run.run_id)
    assert directory.is_dir()

    repo.delete(run.run_id)

    assert not directory.exists()
    with pytest.raises(RunNotFoundError):
        repo.artifacts(run.run_id)
    with repo._engine.connect() as connection:  # noqa: SLF001
        from sqlalchemy import func, select

        from dsbf.store import tables

        assert (
            connection.execute(
                select(func.count()).select_from(tables.artifacts)
            ).scalar()
            == 0
        )


def test_a_failed_save_leaves_no_files(repo, tmp_path, monkeypatch):
    run = profile(50)

    def boom(*_a, **_k):
        raise RuntimeError("disk")

    monkeypatch.setattr(repo, "_insert", boom)
    with pytest.raises(RuntimeError):
        repo.save(run)

    assert not (tmp_path / "artifacts" / str(run.run_id)).exists()
