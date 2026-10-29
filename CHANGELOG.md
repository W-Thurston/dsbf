# Changelog

All notable changes to DSBF are recorded here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions before `1.0.0a1`
were development builds that were never published; their history is in git.

## [Unreleased]

Work toward `1.0.0`, which waits for the dashboard redesign. The run and table schemas are
already at `1.0.0` and follow the rules in
[`docs/guides/compatibility.md`](docs/guides/compatibility.md): additive within 1.x, a
deprecation lasting one minor release.

### Added

- A run store (SQLite, with Alembic migrations) and an artifact store, kept with
  `--sink store`: runs, findings and their history, the bulky evidence a check attaches
  (artifacts), and a Parquet snapshot of the data as profiled.
- `dsbf runs` (`list`, `show`, `diff`, `delete`, `gc`, `import`) to browse, compare and clean
  the store.
- A local API (`/v2`, OpenAPI at `schema/openapi.json`) over the store with a job runner that
  profiles a local file, and `dsbf dashboard`, which serves it on `127.0.0.1`, `localhost` or
  `::1` only.
- The dashboard, rebuilt in TypeScript (Vue 3) on a client generated from the OpenAPI
  document and bundled in the wheel. It is a first version, mostly tables and cards, and is
  redesigned before `1.0.0`. Pages: datasets with lineage and finding history, a run's
  overview, findings, columns, relationships, quality and ML readiness, time series,
  a comparison of two runs, and a form for a dataset's saved settings with a rerun.
- `Run.plan.settings` records the settings a run used; a rerun starts from them.
- `dsbf schema check` also gates the `RunDiff` schema.
- A guide to running DSBF on your own machine.

### Changed

- The `Run` and Parquet table schemas are `1.0.0` (the run was `0.10.0`); the policy schema
  is version 1.
- `fuzzy_duplicates` no longer reports pairs whose one-character deletions only collide in
  code space.
- The `server` extra now holds the store and the service (FastAPI, SQLAlchemy, Alembic,
  fsspec); the `sql` extra is still a placeholder.

### Notes

- `RunDiff` reports count-scaled statistics as changed when two runs differ in sample
  size; see the known limitations in the compatibility guide.

## [1.0.0a1] - 2026-10-27

The first release, for embedded use: profile a table from Python or a pipeline and
get one validated result, the `Run`.

### Added

- `dsbf.profile()` profiles a Polars, pandas, CSV, Parquet or Excel source and returns a
  `Run`: the data profile, 47 checks' findings, a health summary and a plan that records
  how it was made.
- Findings query API (`run.findings`), notebook display and a policy evaluator
  (`policy.yaml` rules over findings and check health, with `previous_run` comparison).
- Sinks for a run: `run.json`, NDJSON findings, Parquet tables and a self-contained HTML
  report, all through `Run.write`; `Run.read` loads a `run.json` back.
- The `dsbf` command: `profile`, `check` (a policy-aware exit code for pipeline steps),
  `plan`, `schema`, `docs` and `cache`, with layered settings (defaults, user file,
  project file, environment, flags).
- Per-check timeouts, process isolation, a result cache with eviction, and optional
  extras (`ml`, `timeseries`, `text`, `excel`, `pandas`) whose checks report
  `unavailable` when the extra is missing.
- Published JSON Schemas for the run, the policy, the Parquet tables and every check's
  parameters and output, guarded against unannounced breaking changes by
  `dsbf schema check`.
- Guides (pipeline, policy, dashboard, writing a check), a generated check reference
  and an example GitHub Actions workflow in `docs/` and `examples/`.

### Notes

- The `sql` and `server` extras are placeholders; nothing imports them yet.
- Schema versions are pre-1.0 (`Run` 0.10.0); the compatibility guarantee starts with
  the stable `1.0.0`.

[1.0.0a1]: https://github.com/W-Thurston/dsbf/releases/tag/v1.0.0a1
