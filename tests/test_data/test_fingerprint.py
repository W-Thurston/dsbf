# tests/test_data/test_fingerprint.py
"""A dataset's fingerprint names its content: same data, same fingerprint."""

import datetime as dt
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from dsbf.data import Dataset, load
from dsbf.data.fingerprint import SCHEME
from dsbf.model.manifest import SourceRef

pytestmark = pytest.mark.unit

TABLE = pl.DataFrame(
    {
        "id": [1, 2, 3, 4],
        "x": [1.5, None, float("nan"), 4.0],
        "s": ["a", None, "", "d"],
        "flag": [True, False, None, True],
        "when": [dt.datetime(2024, 1, d) for d in (1, 2, 3, 4)],
        "tags": [[1, 2], None, [], [3]],
        "pair": [{"a": 1}, {"a": 2}, None, {"a": 4}],
    }
)


def _fingerprint(frame: pl.DataFrame) -> str:
    return load(frame).fingerprint


def _set(frame: pl.DataFrame, column: str, row: int, value: Any) -> pl.DataFrame:
    return frame.with_columns(frame[column].scatter(row, value))


def test_the_fingerprint_is_pinned():
    """A Polars upgrade that changes `Series.hash` fails here: bump the scheme tag."""
    frame = pl.DataFrame({"a": [1, 2, 3], "b": ["x", None, "z"], "c": [0.5, 1.5, None]})

    assert _fingerprint(frame) == "polars-hash-v2:372970ba3d617b3657f6b8e4305374cb"


def test_the_fingerprint_names_its_scheme():
    assert _fingerprint(TABLE).startswith(f"{SCHEME}:")


def test_it_is_stable_across_loads_and_calls():
    dataset = load(TABLE)

    assert dataset.fingerprint == dataset.fingerprint == load(TABLE.clone()).fingerprint


def test_it_is_stable_across_processes():
    code = (
        "import polars as pl\n"
        "from dsbf.data import load\n"
        "print(load(pl.DataFrame({'a': [1, None], 's': ['x', 'y']})).fingerprint)"
    )
    runs = {
        subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        ).stdout
        for _ in range(2)
    }

    assert len(runs) == 1
    assert runs.pop().startswith(SCHEME)


def test_it_ignores_how_the_table_is_chunked():
    chunked = pl.concat([TABLE[:1], TABLE[1:3], TABLE[3:]], rechunk=False)

    assert chunked.n_chunks() > 1
    assert _fingerprint(chunked) == _fingerprint(TABLE)


def test_the_same_data_from_pandas_polars_and_arrow_agrees():
    frame = pl.DataFrame({"a": [1, 2, 3], "b": [0.5, None, 2.5], "c": ["x", "y", None]})

    fingerprints = {
        load(frame).fingerprint,
        load(frame.to_pandas()).fingerprint,
        load(frame.to_arrow()).fingerprint,
        load(frame.lazy()).fingerprint,
    }

    assert len(fingerprints) == 1


def test_a_file_round_trip_keeps_the_fingerprint(tmp_path):
    path = tmp_path / "t.parquet"
    TABLE.select("id", "x", "s", "flag", "when").write_parquet(path)

    assert (
        load(path).fingerprint
        == load(TABLE.select("id", "x", "s", "flag", "when")).fingerprint
    )


def test_changing_one_cell_changes_the_fingerprint():
    base = _fingerprint(TABLE)
    for column, row, value in [
        ("id", 0, 9),
        ("x", 0, 1.6),
        ("s", 2, "e"),
        ("flag", 0, False),
        ("when", 3, dt.datetime(2025, 1, 4)),
    ]:
        edited = _set(TABLE, column, row, value)

        assert _fingerprint(edited) != base, column


@pytest.mark.parametrize(
    "edit",
    [
        pytest.param(lambda t: t.rename({"id": "key"}), id="rename"),
        pytest.param(lambda t: t.with_columns(pl.col("id").cast(pl.Int32)), id="dtype"),
        pytest.param(lambda t: t.select(t.columns[::-1]), id="column-order"),
        pytest.param(lambda t: t[[1, 0, 2, 3]], id="row-order"),
        pytest.param(lambda t: t.head(3), id="fewer-rows"),
        pytest.param(lambda t: t.drop("flag"), id="fewer-columns"),
        pytest.param(lambda t: _set(t, "s", 0, None), id="value-to-null"),
        pytest.param(
            lambda t: _set(t, "x", 1, float("nan")),
            id="null-to-nan",
        ),
    ],
)
def test_structure_changes_the_fingerprint(edit):
    assert _fingerprint(edit(TABLE)) != _fingerprint(TABLE)


def test_column_names_cannot_run_into_each_other():
    first = pl.DataFrame({"ab": [1], "c": [1]})
    second = pl.DataFrame({"a": [1], "bc": [1]})

    assert _fingerprint(first) != _fingerprint(second)


def test_an_empty_table_has_a_fingerprint():
    empty = pl.DataFrame({"a": pl.Series([], dtype=pl.Int64)})

    assert _fingerprint(empty) != _fingerprint(empty.rename({"a": "b"}))


def test_a_one_cell_edit_in_a_large_pandas_frame_is_noticed():
    rng = np.random.default_rng(0)
    values = rng.normal(size=100_000)
    nudged = values.copy()
    nudged[73_000] += 1e-9
    ints = rng.integers(0, 9, 100_000)
    frame = pd.DataFrame({"a": values, "b": ints})
    edited = pd.DataFrame({"a": nudged, "b": ints})

    assert load(frame).fingerprint != load(edited).fingerprint
    assert load(frame).fingerprint == load(frame.copy()).fingerprint


def test_it_is_computed_once_and_kept(monkeypatch):
    calls = []
    import dsbf.data.dataset as dataset_module

    real = dataset_module.fingerprint_frame
    monkeypatch.setattr(
        dataset_module,
        "fingerprint_frame",
        lambda frame: calls.append(1) or real(frame),
    )
    dataset = load(TABLE)

    assert calls == []
    dataset.fingerprint
    dataset.fingerprint

    assert calls == [1]


def test_a_dataset_built_directly_has_one_too():
    dataset = Dataset(frame=TABLE, name="t", source=SourceRef(kind="dataframe"))

    assert dataset.fingerprint == _fingerprint(TABLE)


def test_arrow_input_with_chunks_agrees():
    chunked = pa.concat_tables([pa.table({"a": [1, 2]}), pa.table({"a": [3]})])

    assert load(chunked).fingerprint == _fingerprint(pl.DataFrame({"a": [1, 2, 3]}))
