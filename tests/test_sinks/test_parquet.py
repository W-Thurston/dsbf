"""The Parquet sinks: fixed schemas, the published listing and the consumer guide."""

import json
import re
import warnings
from dataclasses import replace
from pathlib import Path

import duckdb
import pyarrow.parquet as pq
import pytest

import dsbf
from dsbf.interfaces import schema_export
from dsbf.model import Run
from dsbf.model.schema import schema_text
from dsbf.sinks import get_sinks, known_sinks
from dsbf.sinks.tables import TABLES, TABLES_SCHEMA_VERSION, tables_schema

ROOT = Path(__file__).parents[2]
TITANIC = ROOT / "tests" / "golden" / "data" / "titanic.csv"
GUIDE = ROOT / "docs" / "CONSUMING_PARQUET.md"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module", autouse=True)
def quiet_libraries():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield


@pytest.fixture(scope="module")
def titanic() -> Run:
    return dsbf.profile(TITANIC)


@pytest.fixture(scope="module")
def written(titanic, tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("parquet")
    titanic.write(directory, sinks=["parquet"])
    return directory


def test_parquet_is_an_alias_for_the_three_table_sinks():
    names = [s.name for s in get_sinks(("parquet",))]
    assert names == ["findings_parquet", "profile_parquet", "results_parquet"]
    assert "parquet" in known_sinks()
    assert [s.name for s in get_sinks(("parquet", "findings_parquet"))] == names


def test_the_alias_writes_three_files_and_names_each_sink(titanic, tmp_path):
    files = titanic.write(tmp_path, sinks=["parquet"])
    assert {k: v.name for k, v in files.items()} == {
        "findings_parquet": "findings.parquet",
        "profile_parquet": "profile.parquet",
        "results_parquet": "results.parquet",
    }


def test_each_file_has_the_declared_schema_and_version(written):
    for table in TABLES.values():
        schema = pq.read_schema(written / table.filename)
        assert [f.name for f in schema] == [c.name for c in table.columns]
        assert [f.type for f in schema] == [c.type for c in table.columns]
        assert [f.nullable for f in schema] == [c.nullable for c in table.columns]
        assert schema.metadata[b"schema_version"] == TABLES_SCHEMA_VERSION.encode()


def test_row_counts_match_the_run(titanic, written):
    assert pq.read_table(written / "findings.parquet").num_rows == len(titanic.findings)
    assert pq.read_table(written / "profile.parquet").num_rows == len(
        titanic.profile.columns
    )
    assert pq.read_table(written / "results.parquet").num_rows == len(titanic.results)


def test_findings_agree_with_the_findings_view(titanic, written):
    table = pq.read_table(written / "findings.parquet").to_pylist()
    assert [r["id"] for r in table] == [f.id for f in titanic.findings]
    for row, finding in zip(table, titanic.findings, strict=True):
        assert json.loads(row["metrics"]) == json.loads(
            json.dumps(finding.evidence.metrics)
        )
        assert row["columns"] == list(finding.subject.columns)
        assert row["run_id"] == str(titanic.run_id)


def test_an_empty_table_keeps_its_columns(titanic):
    empty = replace(TABLES["findings"], rows=lambda run: [])
    table = empty.build(titanic)
    assert table.num_rows == 0
    assert table.schema.names == [c.name for c in TABLES["findings"].columns]


def test_the_checked_in_listing_is_current():
    expected = schema_text(tables_schema())
    assert (ROOT / "schema" / "tables.schema.json").read_text("utf-8") == expected


def _listing(*columns, version="0.1.0", table="t"):
    return {
        "schema_version": version,
        "tables": {table: {"file": "t.parquet", "columns": list(columns)}},
    }


def _col(name, type="string", nullable=True):
    return {"name": name, "type": type, "nullable": nullable, "description": ""}


@pytest.mark.parametrize(
    ("old", "new", "breaking"),
    [
        (_col("a"), None, True),
        (_col("a"), _col("a", "int64"), True),
        (_col("a"), _col("a", nullable=False), True),
        (_col("a", nullable=False), _col("a"), True),
    ],
)
def test_breaking_table_changes(old, new, breaking):
    new_listing = _listing(*([new] if new else []))
    changes = schema_export.compare_table_schemas(_listing(old), new_listing)
    assert [c.breaking for c in changes] == [breaking]


def test_additive_table_changes():
    old = _listing(_col("a"))
    assert [
        c.breaking
        for c in schema_export.compare_table_schemas(
            old, _listing(_col("a"), _col("b"))
        )
    ] == [False]
    added = schema_export.compare_table_schemas(
        old, _listing(_col("a"), _col("b", nullable=False))
    )
    assert [c.breaking for c in added] == [True]
    renamed = _listing(_col("a"), table="u")
    assert {c.breaking for c in schema_export.compare_table_schemas(old, renamed)} == {
        True,
        False,
    }
    assert schema_export.compare_table_schemas(old, old) == []


def test_guide_queries_run_through_duckdb(written, monkeypatch):
    blocks = re.findall(r"```sql\n(.*?)```", GUIDE.read_text("utf-8"), re.S)
    assert len(blocks) >= 6
    monkeypatch.chdir(written)
    connection = duckdb.connect()
    for sql in blocks:
        rows = connection.sql(sql).fetchall()
        assert rows, f"no rows from:\n{sql}"


def test_the_null_rate_query_reads_the_metric_back(written):
    rows = duckdb.execute(
        "SELECT columns[1], json_extract(metrics, '$.null_share')::DOUBLE "
        f"FROM '{written}/findings.parquet' WHERE check_id = 'null_rate' "
        "ORDER BY 2 DESC"
    ).fetchall()
    assert rows[0][0] == "deck"
    assert rows[0][1] == pytest.approx(688 / 891)
