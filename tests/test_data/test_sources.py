# tests/test_data/test_sources.py
"""Every tier-1 source type loads into a Polars-backed Dataset."""

import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from dsbf.data import Dataset, load
from dsbf.model.manifest import SourceRef
from tests.fixtures.adversarial import build_adversarial_frame

pytestmark = pytest.mark.unit

WHEN = [dt.datetime(2024, 1, 1, 9, 30), dt.datetime(2024, 1, 2, 10, 45), None]
TABLE = pl.DataFrame(
    {
        "id": [1, 2, 3],
        "x": [1.5, None, 3.5],
        "s": ["a", None, "c"],
        "when": WHEN,
    }
)
COLUMNS = ["id", "x", "s", "when"]


def _time_zone(dataset: Dataset, column: str) -> str | None:
    dtype = dataset.schema[column]
    assert isinstance(dtype, pl.Datetime)
    return dtype.time_zone


def _assert_values(dataset: Dataset) -> None:
    frame = dataset.frame
    assert frame["id"].to_list() == [1, 2, 3]
    assert frame["x"].to_list() == [1.5, None, 3.5]
    assert frame["s"].to_list() == ["a", None, "c"]


IN_MEMORY = {
    "pandas": lambda: TABLE.to_pandas(),
    "polars": lambda: TABLE,
    "polars-lazy": lambda: TABLE.lazy(),
    "arrow-table": lambda: TABLE.to_arrow(),
    "arrow-batch": lambda: pa.RecordBatch.from_pylist(TABLE.to_dicts()),
}


@pytest.mark.parametrize("build", IN_MEMORY.values(), ids=IN_MEMORY.keys())
def test_in_memory_tables_load(build):
    dataset = load(build())

    assert isinstance(dataset.frame, pl.DataFrame)
    assert dataset.columns == COLUMNS
    assert (dataset.n_rows, dataset.n_columns) == (3, 4)
    assert dataset.name == "dataframe"
    assert dataset.source == SourceRef(kind="dataframe")
    assert dataset.coerced == {}
    assert dataset.schema["when"] == pl.Datetime
    _assert_values(dataset)


def _write_json(path: Path) -> None:
    records = [{"id": 1, "x": 1.5, "s": "a"}, {"id": 2, "x": None, "s": None}]
    records.append({"id": 3, "x": 3.5, "s": "c"})
    path.write_text(json.dumps(records))


def _write_ndjson(path: Path) -> None:
    rows = [{"id": 1, "x": 1.5, "s": "a"}, {"id": 2, "x": None, "s": None}]
    rows.append({"id": 3, "x": 3.5, "s": "c"})
    path.write_text("\n".join(json.dumps(row) for row in rows))


FILES = {
    "csv": ("data.csv", "csv", lambda p: TABLE.write_csv(p)),
    "tsv": ("data.tsv", "tsv", lambda p: TABLE.write_csv(p, separator="\t")),
    "parquet": ("data.parquet", "parquet", lambda p: TABLE.write_parquet(p)),
    "pq": ("data.pq", "parquet", lambda p: TABLE.write_parquet(p)),
    "excel": ("data.xlsx", "excel", lambda p: TABLE.write_excel(p)),
    "json": ("data.json", "json", _write_json),
    "ndjson": ("data.ndjson", "ndjson", _write_ndjson),
    "jsonl": ("data.jsonl", "ndjson", _write_ndjson),
}


@pytest.mark.parametrize("filename,format,write", FILES.values(), ids=FILES.keys())
def test_files_load(tmp_path, filename, format, write):
    path = tmp_path / filename
    write(path)

    dataset = load(path)

    assert dataset.source == SourceRef(
        kind="file", location=str(path.resolve()), format=format
    )
    assert dataset.name == path.stem
    assert dataset.n_rows == 3
    assert dataset.columns[:3] == ["id", "x", "s"]
    _assert_values(dataset)


def test_a_path_string_loads_like_a_path(tmp_path):
    TABLE.write_csv(tmp_path / "t.csv")

    assert load(str(tmp_path / "t.csv")).frame.equals(load(tmp_path / "t.csv").frame)


def test_format_overrides_the_extension(tmp_path):
    TABLE.write_csv(tmp_path / "data.txt")

    dataset = load(tmp_path / "data.txt", format="csv")

    assert dataset.source.format == "csv"
    assert dataset.n_rows == 3


def test_name_overrides_the_default(tmp_path):
    TABLE.write_csv(tmp_path / "t.csv")

    assert load(tmp_path / "t.csv", name="orders").name == "orders"
    assert load(TABLE, name="orders").name == "orders"


def test_a_dataset_passes_through():
    dataset = load(TABLE)

    assert load(dataset) is dataset
    assert load(dataset, name="renamed").name == "renamed"


@pytest.mark.parametrize("format", ["parquet", "excel"])
def test_typed_files_keep_datetimes(tmp_path, format):
    path = tmp_path / ("t.parquet" if format == "parquet" else "t.xlsx")
    if format == "parquet":
        TABLE.write_parquet(path)
    else:
        TABLE.write_excel(path)

    dataset = load(path)

    assert dataset.schema["when"] == pl.Datetime
    assert dataset.frame["when"].to_list() == WHEN


# --- pandas edge cases ---------------------------------------------------------------


def test_adversarial_frame_keeps_names_values_and_timezone():
    frame = build_adversarial_frame()

    dataset = load(frame)

    assert dataset.columns == list(frame.columns)  # including the empty name
    assert dataset.n_rows == len(frame)
    nulls = dataset.frame.null_count().row(0, named=True)
    assert nulls["all_null"] == len(frame)
    assert nulls["all_nan_float"] == len(frame)  # pandas NaN is missing
    assert np.isinf(dataset.frame["inf_col"].to_numpy()).sum() > 0
    assert _time_zone(dataset, "ts_tz") == "US/Eastern"  # unit is pandas' choice
    assert dataset.frame["ts_tz"].dt.replace_time_zone(None).to_list() == list(
        frame["ts_tz"].dt.tz_localize(None)
    )


def test_pandas_nan_is_null_but_polars_nan_is_a_value():
    from_pandas = load(pd.DataFrame({"v": [1.0, np.nan]}))
    from_polars = load(pl.DataFrame({"v": [1.0, float("nan")]}))

    assert from_pandas.frame["v"].null_count() == 1
    assert from_polars.frame["v"].null_count() == 0
    assert from_polars.frame["v"].is_nan().sum() == 1


def test_pandas_extension_dtypes_load():
    frame = pd.DataFrame(
        {
            "text": pd.array(["a", None], dtype="string"),
            "arrow_text": pd.array(["a", None], dtype="string[pyarrow]"),
            "category": pd.Categorical(["u", "v"]),
            "count": pd.array([1, None], dtype="Int64"),
            "flag": pd.array([True, None], dtype="boolean"),
            "plain": ["a", "b"],
            "moment": pd.DatetimeIndex(["2024-01-01", None]),
        }
    )

    dataset = load(frame)

    schema = dataset.schema
    assert schema["text"] == schema["arrow_text"] == schema["plain"] == pl.String
    assert schema["category"] == pl.Categorical
    assert schema["count"] == pl.Int64
    assert schema["flag"] == pl.Boolean
    assert schema["moment"] == pl.Datetime
    assert dataset.coerced == {}


def test_mixed_type_columns_become_text_and_are_reported():
    frame = pd.DataFrame({"mixed": [1, "a", None, [1, 2]], "n": [1, 2, 3, 4]})

    dataset = load(frame)

    assert dataset.frame["mixed"].to_list() == ["1", "a", None, "[1, 2]"]
    assert dataset.schema["n"] == pl.Int64
    assert list(dataset.coerced) == ["mixed"]


def test_non_string_column_names_are_stringified():
    dataset = load(pd.DataFrame([[1, 2]], columns=[0, 1]))

    assert dataset.columns == ["0", "1"]


def test_float_columns_are_not_copied():
    frame = pd.DataFrame({"a": np.arange(1000.0)})

    dataset = load(frame)

    assert np.shares_memory(frame["a"].to_numpy(), dataset.frame["a"].to_numpy())


def test_pandas_leg_runs_the_requested_major():
    expected = os.environ.get("DSBF_TEST_PANDAS_MAJOR")
    if expected is None:
        pytest.skip("only the pandas-3 CI leg sets DSBF_TEST_PANDAS_MAJOR")
    assert pd.__version__.split(".")[0] == expected


# --- CSV semantics -------------------------------------------------------------------


def test_csv_missing_tokens_match_pandas(tmp_path):
    tokens = ["NA", "N/A", "NaN", "nan", "null", "NULL", "None", "n/a", "<NA>", "#N/A"]
    rows = ["num,txt"] + [f"{i},{token}" for i, token in enumerate(tokens)]
    rows += ["10,keep", "NA,", ",x"]
    path = tmp_path / "tokens.csv"
    path.write_text("\n".join(rows) + "\n")

    dataset = load(path)
    expected = pd.read_csv(path)

    assert dataset.frame.null_count().row(0) == tuple(expected.isna().sum())
    assert dataset.schema["num"] == pl.Int64


def test_csv_text_far_below_the_top_does_not_fail_an_integer_column(tmp_path):
    path = tmp_path / "late.csv"
    path.write_text("v\n" + "1\n" * 5000 + "oops\n")

    dataset = load(path)

    assert dataset.schema["v"] == pl.String
    assert dataset.n_rows == 5001


def test_csv_date_like_text_stays_text(tmp_path):
    path = tmp_path / "dates.csv"
    path.write_text("d\n2024-01-01\n2024-01-02\n")

    assert load(path).schema["d"] == pl.String


# --- parquet written by pandas ---------------------------------------------------


def test_pandas_parquet_with_datetime64_and_timezone_loads(tmp_path):
    frame = pd.DataFrame(
        {
            "x": [1.0, 2.0, np.nan],
            "when": pd.date_range("2024-01-01", periods=3),
            "when_tz": pd.date_range("2024-01-01", periods=3, tz="Europe/Paris"),
        }
    )
    path = tmp_path / "pandas.parquet"
    frame.to_parquet(path)

    dataset = load(path)

    assert dataset.schema["when"] == pl.Datetime
    assert _time_zone(dataset, "when_tz") == "Europe/Paris"
    assert dataset.frame["when"].to_list() == list(frame["when"])
    assert dataset.frame["x"].null_count() == 1  # parquet stores the NaN as null


# --- isolation -----------------------------------------------------------------------


def test_a_polars_source_does_not_import_pandas():
    code = (
        "import sys, polars as pl\n"
        "from dsbf.data import load\n"
        "load(pl.DataFrame({'a': [1]}))\n"
        "load(pl.DataFrame({'a': [1]}).to_arrow())\n"
        "assert 'pandas' not in sys.modules, 'pandas was imported'\n"
    )

    result = subprocess.run(
        [sys.executable, "-I", "-c", code], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
