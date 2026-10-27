# Changelog

All notable changes to DSBF are recorded here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versions before `1.0.0a1`
were development builds that were never published; their history is in git.

## [1.0.0a1] - Unreleased

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
