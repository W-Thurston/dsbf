# tests/test_data/test_errors.py
"""Input DSBF cannot load raises a typed error that says what to try."""

import pandas as pd
import polars as pl
import pyarrow as pa
import pytest

from dsbf.data import (
    SourceDependencyError,
    SourceError,
    SourceReadError,
    UnsupportedSourceError,
    load,
)

pytestmark = pytest.mark.unit


def _write(tmp_path, name, content, mode="w"):
    path = tmp_path / name
    path.write_text(content) if mode == "w" else path.write_bytes(content)
    return path


# (id, builds the source from tmp_path, load kwargs, error, text the hint must carry)
CASES = [
    ("missing", lambda p: p / "nope.csv", {}, SourceReadError, "the path"),
    ("empty csv", lambda p: _write(p, "e.csv", ""), {}, SourceReadError, "UTF-8"),
    (
        "ragged csv",
        lambda p: _write(p, "r.csv", "a,b\n1,2,3\n4,5\n"),
        {},
        SourceReadError,
        "same number of fields",
    ),
    (
        "corrupt parquet",
        lambda p: _write(p, "c.parquet", b"not parquet", "wb"),
        {},
        SourceReadError,
        "Parquet",
    ),
    (
        "corrupt excel",
        lambda p: _write(p, "c.xlsx", b"not excel", "wb"),
        {},
        SourceReadError,
        "valid workbook",
    ),
    (
        "json that is not an array",
        lambda p: _write(p, "o.json", '{"a": 1}\n{"a": 2}\n'),
        {},
        SourceReadError,
        "ndjson",
    ),
    (
        "malformed ndjson",
        lambda p: _write(p, "m.ndjson", '{"a": 1}\n{oops\n'),
        {},
        SourceReadError,
        "one complete JSON object",
    ),
    (
        "unknown extension",
        lambda p: _write(p, "x.weird", "a\n1\n"),
        {},
        UnsupportedSourceError,
        ".parquet",
    ),
    (
        "unknown format",
        lambda p: _write(p, "x.csv", "a\n1\n"),
        {"format": "xml"},
        UnsupportedSourceError,
        "csv",
    ),
    ("directory", lambda p: p, {}, UnsupportedSourceError, "one file"),
    ("url", lambda p: "s3://bucket/data.parquet", {}, UnsupportedSourceError, "local"),
    ("number", lambda p: 3, {}, UnsupportedSourceError, "pandas, Polars or Arrow"),
    ("list", lambda p: [{"a": 1}], {}, UnsupportedSourceError, "pandas, Polars"),
    ("arrow array", lambda p: pa.array([1, 2]), {}, UnsupportedSourceError, "Arrow"),
    (
        "format on a table",
        lambda p: pl.DataFrame({"a": [1]}),
        {"format": "csv"},
        UnsupportedSourceError,
        "Drop format",
    ),
    (
        "duplicate pandas columns",
        lambda p: pd.DataFrame([[1, 2]], columns=["a", "a"]),
        {},
        SourceReadError,
        "unique",
    ),
    (
        "pattern column names",
        lambda p: pl.DataFrame({"^a$": [1], "*": [2], "ok": [3]}),
        {},
        SourceReadError,
        "Rename",
    ),
    ("no columns", lambda p: pl.DataFrame(), {}, SourceReadError, "holds data"),
]


@pytest.mark.parametrize(
    "build,kwargs,error,hint",
    [case[1:] for case in CASES],
    ids=[case[0] for case in CASES],
)
def test_bad_input_raises_a_typed_error_with_a_hint(
    tmp_path, build, kwargs, error, hint
):
    source = build(tmp_path)

    with pytest.raises(error) as caught:
        load(source, **kwargs)

    assert hint in (caught.value.hint or "")
    assert str(caught.value).endswith(f"Hint: {caught.value.hint}")
    assert caught.value.source


def test_read_errors_chain_the_underlying_error(tmp_path):
    path = _write(tmp_path, "c.parquet", b"not parquet", "wb")

    with pytest.raises(SourceReadError) as caught:
        load(path)

    assert caught.value.__cause__ is not None
    assert caught.value.message.startswith("Could not read 'c.parquet' as parquet")


def test_source_errors_are_not_value_errors():
    assert not issubclass(SourceError, ValueError)
    for error in (SourceReadError, UnsupportedSourceError, SourceDependencyError):
        assert issubclass(error, SourceError)


def test_zero_rows_with_columns_still_loads(tmp_path):
    path = _write(tmp_path, "header.csv", "a,b\n")

    dataset = load(path)

    assert dataset.n_rows == 0
    assert dataset.columns == ["a", "b"]


def test_excel_without_fastexcel_names_the_extra(tmp_path, monkeypatch):
    path = tmp_path / "t.xlsx"
    pl.DataFrame({"a": [1]}).write_excel(path)
    monkeypatch.setattr("dsbf.data.files.find_spec", lambda name: None)

    with pytest.raises(SourceDependencyError) as caught:
        load(path)

    assert caught.value.hint and "dsbf[excel]" in caught.value.hint


def test_pattern_column_error_lists_only_the_unsafe_names():
    with pytest.raises(SourceReadError) as caught:
        load(pl.DataFrame({"^a$": [1], "*": [2], ".*": [3], "a.b": [4]}))

    assert "'^a$', '*'" in caught.value.message
    assert ".*'" not in caught.value.message


def test_pattern_column_names_are_refused_from_files(tmp_path):
    path = _write(tmp_path, "p.csv", "^a$,b\n1,2\n")

    with pytest.raises(SourceReadError, match=r"\^a\$"):
        load(path)


def test_names_that_only_resemble_patterns_still_profile():
    import dsbf

    frame = pl.DataFrame({name: [1, 2, 3] for name in (".*", "a.b", "^a", "a$")})

    run = dsbf.profile(frame)

    assert run.dataset.n_columns == 4
