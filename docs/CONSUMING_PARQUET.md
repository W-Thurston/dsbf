# Consume DSBF results in your own warehouse or dashboard

`run.write(dir, sinks=["parquet"])` (or `dsbf profile --sink parquet`) writes three flat tables next to
`run.json`. Every table starts with `run_id`, so the files of many runs can be appended into one
warehouse table and joined on that key.

| File | One row per | Use it to |
|---|---|---|
| `findings.parquet` | finding | list problems by severity, dimension, column or check |
| `profile.parquet` | column | chart distributions, null rates and types |
| `results.parquet` | planned check | see which checks ran, were skipped or failed, and how long they took |

The column listing, with types, nullability and a description of each column, is checked in as
[`schema/tables.schema.json`](../schema/tables.schema.json). Each file also carries `schema_version` in
its Parquet key-value metadata. The version follows the run schema's rules: a removed or retyped column
is a major bump (a minor one while the major is 0), and a new nullable column is a minor bump.
`dsbf schema check` fails when the listing changes without the bump.

## Conventions

- Enumerations (`severity`, `dimension`, `status`, `semantic_type`) are plain strings. Severity runs
  `info`, `notice`, `warn`, `error`; sort with the `CASE` below, not alphabetically.
- `columns` (the subject of a finding) is a list: empty for the dataset, one name for a column, two for
  a pair.
- `metrics` (a finding's evidence), `params` and `output` (a result) are JSON text, because their
  values differ by check. Read one with `json_extract` (DuckDB, SQLite) or `JSON_VALUE` (BigQuery,
  Snowflake, Postgres `->>`).
- A statistic that applies to a type of column only (`mean` for numeric columns, `true_count` for
  booleans) is null for the others. A NaN or infinite value is null, and a finding lists the metric in
  `non_finite_metrics`.
- `profile` stores `top_values` as a list of `(value, count)`, with the value as text, and a numeric
  column's histogram as `histogram_edges` (one more than `histogram_counts`).

## Example queries

The queries are DuckDB SQL and run against a directory holding the three files. The test suite runs
every `sql` block below.

The worst findings of the run:

```sql
SELECT severity, check_id, columns, title
FROM 'findings.parquet'
ORDER BY CASE severity WHEN 'error' THEN 0 WHEN 'warn' THEN 1 WHEN 'notice' THEN 2 ELSE 3 END,
         check_id
LIMIT 10;
```

Findings per quality dimension and severity:

```sql
SELECT dimension, severity, count(*) AS findings
FROM 'findings.parquet'
GROUP BY ALL
ORDER BY dimension, severity;
```

The columns with the most nulls, from the profile:

```sql
SELECT name, semantic_type, nulls, round(nulls::DOUBLE / rows, 3) AS null_share
FROM 'profile.parquet'
WHERE nulls > 0
ORDER BY null_share DESC;
```

A metric out of a finding's evidence (here, the share of nulls the `null_rate` check measured):

```sql
SELECT columns[1] AS column_name,
       json_extract(metrics, '$.null_share')::DOUBLE AS null_share
FROM 'findings.parquet'
WHERE check_id = 'null_rate'
ORDER BY null_share DESC;
```

The five most common values of a categorical column:

```sql
SELECT unnest(top_values, recursive := true)
FROM 'profile.parquet'
WHERE name = 'sex';
```

How the checks ended (the `note_code` and `note_message` columns say why a check was skipped,
unavailable or failed):

```sql
SELECT status, count(*) AS checks, max(wall_seconds) AS slowest_seconds
FROM 'results.parquet'
GROUP BY status;
```

Findings with the status of the check that raised them:

```sql
SELECT r.status, count(*) AS findings
FROM 'findings.parquet' f
JOIN 'results.parquet' r USING (run_id, check_id)
GROUP BY r.status;
```

## Loading into a warehouse

Append each run's files to one table per file, keyed by `run_id`. Most warehouses load Parquet
directly: `COPY INTO` (Snowflake), `bq load --source_format=PARQUET` (BigQuery),
`COPY ... FROM ... (FORMAT parquet)` through DuckDB or `pg_parquet` (Postgres). Before loading, compare
the file's `schema_version` with the one you built the table for.
