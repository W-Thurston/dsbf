# Consume DSBF findings in your own dashboard

DSBF writes its result in forms a dashboard, a warehouse or another program can read without
importing DSBF. This guide shows each, from the most structured to the most convenient. The examples
use the customer table `data.csv` from [Use DSBF in your pipeline](pipeline.md).

```python
import dsbf

run = dsbf.profile("data.csv", depth="quick")
paths = run.write("out", sinks=["json", "ndjson", "parquet"])
print(sorted(path.name for path in paths.values()))
```

## Which file to read

| You want | Read | Contract |
| --- | --- | --- |
| A table of problems to filter and chart | `findings.parquet` | [`schema/tables.schema.json`](../../schema/tables.schema.json) |
| Column statistics, histograms and top values | `profile.parquet` | the same file |
| Which checks ran, were skipped or failed | `results.parquet` | the same file |
| A stream of findings for a log or a queue | `findings.ndjson`, one finding per line | `schema/checks/` and `schema/run.schema.json` |
| Everything, including each check's own output | `run.json` | [`schema/run.schema.json`](../../schema/run.schema.json) |
| A page to send to a person | `report.html`, self-contained and offline | none |

The Parquet tables all start with `run_id`, so the files of many runs append into one table.
[Consume DSBF results in your own warehouse](../CONSUMING_PARQUET.md) lists every column and gives
SQL queries for DuckDB, BigQuery and Snowflake.

## Findings as a table

```python
import polars as pl

findings = pl.read_parquet("out/findings.parquet")
by_severity = (
    findings.group_by("severity", "dimension").len().sort("severity", "dimension")
)
print(by_severity)
print(
    findings.select("run_id", "id", "severity", "check_id", "columns", "title").head(5)
)
```

A finding's `id` is stable across runs for the same check, subject and code, so joining two runs on
`id` shows what is new, what went away and what changed in `severity`. A finding's `columns` is a
list: empty for the dataset, one name for a column and two for a pair. `metrics`, the numbers behind
a finding, is JSON text because its keys differ by check:

```python
import json

row = findings.filter(pl.col("code") == "heavy_tails").row(0, named=True)
print(row["columns"], json.loads(row["metrics"])["kurtosis"])
```

## Findings as a stream

`findings.ndjson` holds one JSON object per finding, in the run's order, so a log shipper can send it
unchanged:

```python
import json
from pathlib import Path

lines = Path("out/findings.ndjson").read_text(encoding="utf-8").splitlines()
first = json.loads(lines[0])
print(len(lines), first["check_id"], first["code"], first["severity"])
```

## The whole run

`run.json` validates against `schema/run.schema.json`, and `dsbf schema export --types run.d.ts`
writes TypeScript types for it, so a web dashboard can type what it reads. In Python, `Run.read`
returns the same `Run` that `dsbf.profile` returned, with its findings, its profile and each check's
output:

```python
from dsbf import Run

again = Run.read("out")
columns = {column.name: column.semantic_type for column in again.profile.columns}
print(columns)
print(again.results["null_rate"].status)
```

The schema's version follows semantic versioning, and a reader refuses a run from another major
version (`RunReadError`) rather than misreading it. Cross-check a check's output shape against
`schema/checks/<check id>.schema.json`.

## In a notebook

A `Run` renders itself as HTML in a notebook cell. `run.findings.to_polars()` and `.to_pandas()` give
the findings as a table without writing a file.

## What a dashboard should not do

- Do not average severities or sum findings into a score. DSBF reports five dimensions separately
  (`completeness`, `validity`, `usability`, `redundancy`, `leakage`, read from
  `run.results["quality_dimensions"].output`) because one number hides which one matters.
- Do not hide a check that did not complete. `results.parquet` lists each check's `status`; a
  `not_applicable`, `skipped`, `error` or `timeout` is a reason a finding you expected is absent.
- Show `confidence` beside `severity`, and the finding's `message`, which carries the caveats.
