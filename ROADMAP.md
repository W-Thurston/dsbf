# DSBF Roadmap

This is the backlog of upcoming work. Each **phase** is a program: its capability, motivation,
boundaries, sequence and success criteria are locked at the top of the phase. Work inside a phase
is broken into **units**. A unit is one branch, one PR, one commit and one session. The workflow
for running a unit is in [`CLAUDE.md`](CLAUDE.md), and the active unit (at most one) lives in
[`PLAN.md`](PLAN.md).

- **Design:** [`docs/REFACTOR_PLAN.md`](docs/REFACTOR_PLAN.md)
- **Findings cited as F-xx / N-xx, with their status:** [`docs/REVIEW_FINDINGS.md`](docs/REVIEW_FINDINGS.md)
- **Finished programs, with every unit and PR:** [`docs/HISTORY.md`](docs/HISTORY.md)
- **How a legacy run works, with measurements:** [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md)

**Status values:**
- `todo`: ready once its dependencies are `done`
- `active`: currently in `PLAN.md`
- `done`: merged; the entry carries notes and the PR number. When a phase's last unit closes, its
  unit entries become a program summary in `docs/HISTORY.md` and a row under "Finished programs"
  (see `.claude/commands/closeout.md`).
- `outline`: a named unit with its goal, dependencies and acceptance. Its phase's planning unit
  confirms it, adds assumptions and steps, and sets it `todo`.
- `placeholder`: a later phase's work, broken into units by that phase's planning unit

**Next unit:** the first `todo` unit, in order, whose dependencies are all `done`.

Phases 6–9 have a locked phase block and an ordered list of placeholders; each starts with a
`Ux-00` planning unit that turns the list into full units, using what earlier phases taught.

## Dependency upgrades

Upgrades ride with the unit that owns the code they affect, so a behaviour change is reviewed
once, next to the code that changes it (proposed 2026-10-16, after U0-16). When a unit starts,
list the open Dependabot PRs and run `uv lock --upgrade --dry-run`. Take the upgrades in the area
the unit touches, verified like any other change: the fast gate, the golden suite when engine
output can move, and a real run. An upgrade that changes results is not taken silently: it either
becomes its own reviewed step (golden snapshots regenerated, with the differences explained in the
PR) or stays behind a cap, recorded in the table below with the unit that revisits it. When a
unit takes an upgrade, it raises the cap and deletes the row.

| Held back (`pyproject.toml` cap) | Why | Revisit in |
|---|---|---|
| `vue-router <5` (PR #69, ignored from 2026-10-17) | 5 is a major bump of the router the current dashboard is built on; checking it needs the dashboard viewed in a browser, and the dashboard is rebuilt in Phase 6 | U6-00 |
| `plotly.js-dist-min <4` (PR #70, ignored from 2026-10-17) | 4 is a major bump of the library behind the legacy charts; checking it needs the dashboard viewed in a browser, and the charts are rebuilt in Phase 6 | U6-00 |

`.github/dependabot.yml` carries an `ignore` entry for each row, so Dependabot doesn't re-raise the held-back version. Removing a cap removes its entry.

## Finished programs

Each program's summary (capability, measured success criteria, key decisions, deviations, PRs) is
in [`docs/HISTORY.md`](docs/HISTORY.md), and how each review finding stands is recorded in
[`docs/REVIEW_FINDINGS.md`](docs/REVIEW_FINDINGS.md).

| Phase | Done | Units (PRs) | Delivered |
|---|---|---|---|
| [0 · Stabilise and baseline](docs/HISTORY.md#phase-0-stabilise-and-baseline) | 2026-10-16 | 17 (#29–#64) | CI green on uv, ruff from the lock, pyrefly with a baseline and test tier markers; hermetic tests; golden snapshots for 10 datasets; the perf harness and CI baseline; ADRs D1–D11 |
| [1 · Contracts](docs/HISTORY.md#phase-1-contracts) | 2026-10-17 | 11 (#66–#79) | `src/` layout, pnpm frontend toolchain, `DSBF_HOME`, extras, the versioned `dsbf.model` contracts, the layered config loader, `dsbf schema export` with a breaking-change gate |
| [2 · Data and profile](docs/HISTORY.md#phase-2-data-and-profile) | 2026-10-18 | 8 (#80–#87) | `dsbf.data.load` to a fingerprinted, sampled `Dataset`; one profile pass (0.44 s at 52.5k × 60); parity with the legacy summaries documented |
| [3 · Engine and check API](docs/HISTORY.md#phase-3-engine-and-check-api) | 2026-10-19 | 12 (#88–#99) | `@check` API, registry and plugins, planner and `dsbf plan`, an isolated, parallel and cached executor, library-friendly logging, all 65 legacy tasks adapted with findings, `dsbf.profile()` |
| [4 · Migrate checks](docs/HISTORY.md#phase-4-migrate-checks) | 2026-10-25 | 20 (#100–#130) | 47 typed `core` checks with calibrated rates and tips, a stopgap `dsbf profile`, the legacy engine, tasks, store and API v1 deleted, a base install of 10 dependencies (533 MB, from 750 MB) with the `ml`, `timeseries` and `text` checks reporting `unavailable` without their extras |
| [5 · Policy, sinks, CLI v2](docs/HISTORY.md#phase-5-policy-sinks-cli-v2-first-release-for-embedded-use) | 2026-10-26 | 12 (#132–#143) | User policies with exit codes, JSON, NDJSON, Parquet and HTML sinks, a findings query API, `dsbf check` with layered settings and a generated reference, guides, calibration fixes, and the release workflow; `1.0.0a1` is on PyPI (2026-10-27) |

---

## Phase 6: Store, service and dashboard v2. Standalone local (placeholder)

- **Capability:** A run store (SQLite with Alembic) and an artifact store (fsspec), `dsbf runs ...`, API v2 with a local job runner, a generated TypeScript client, and the dashboard rebuilt on API v2 and served by `dsbf dashboard` from the wheel; release `1.0.0`.
- **Motivation:** Mode 2 (standalone, local): DSBF is the pipeline, with history and a bundled UI (REFACTOR_PLAN §8.2–§8.4).
- **Boundaries:** Local and single-user only: no Postgres, auth or queue (Phase 8). API v1 and the old store were deleted in U4-18, so this phase builds new and migrates nothing.
- **Sequence and dependencies:** Depends on Phase 5. Store, then API, then client, then the dashboard tab by tab, then packaging.
- **Success criteria:** `pip install "dsbf[server]" && dsbf check data.csv --sink store && dsbf dashboard` works on a clean machine. Opening a run transfers under 200 KB.

- **U6-00 · Plan Phase 6** (`todo`, depends on the Phase 5 units).  **Note from U5-08:** a `codes` field on the check spec would make the check reference's finding codes exact (scanned from source today). **Note from U5-06:** the HTML report omits `Run.policy` (the CLI prints the verdict); show it in the dashboard or report. **Note from U5-09:** (1) promote the perf baseline from the 2026-10-27 scheduled Performance run on `main` (the committed one predates U5-11's calibration changes); (2) re-check that `dsbf.interfaces.reference` still duck-types Typer commands on a Typer bump (0.27 bundles click as `typer._click`); (3) the `sql` and `server` extras still need an importer and an install-matrix leg together; (4) a coverage floor is undecided, the first number is in the `full` job's step summary.
  - **Carry in from U4-00:** the dashboard does not run from U4-18 until this phase's API v2 and shell land (decided 2026-10-19). F-15 (dataset identity, the snapshot of the data as profiled) and F-25 (pagination, error mapping, migrations, garbage collection, encoded path parameters) are design inputs for the store and API, not fixes to old code.
  - **Carry in from U4-18:** `tests/validation/DASHBOARD_CHECKLIST.md` still describes the old dashboard's browser checks; rewrite it for the rebuilt one. `.github/ISSUE_TEMPLATE/dashboard_ui.yml` and the `frontend/` source were kept.
  - **Carry in from U1-11:** (1) the dashboard is JavaScript, so it can use `frontend/src/generated/run.d.ts` only through JSDoc `import()` types or after a move to TypeScript; (2) a TypeScript consumer needs its own rank table for `Severity` and `Confidence`, which the schema cannot carry; (3) revisit the `vue-router <5` and `plotly.js-dist-min <4` caps in the held-back table.
  - **Carry in from U4-07 and U4-11:** `associations` returns the tested pairs and no correlation matrix, so the column view computes a matrix on demand; `seasonal_decomposition` keeps strengths, not the component series, so the store keeps those arrays.
  - **Carry in from U4-19:** the `sql` and `server` extras have no install-matrix leg or `unavailable` test, since no code imports them; add both with the first code that does. `--perf-cprofile` went with the legacy perf scenarios; add one to `tests/perf/measure_engine.py` only if a check-level profile is wanted.
  - **Carry in from U1-03 and U0-16:** clear the 20 frontend lint warnings and raise `no-unused-vars`, `vue/no-side-effects-in-computed-properties` and `vue/require-default-prop` to `error` (16 dead variables in 12 files; `acfSelectedColData` in `TimeSeriesTab.vue` assigns inside a `computed`; route props in `RunDetailView.vue` and `RunListView.vue`); revisit `vue/no-v-html` if another source appears. The light theme draws chart titles and card headings such as "Value Counts" almost white on a light background (same on ECharts 5).
- Placeholders, in order:
  1. Run repository on SQLite with Alembic (datasets with a stable identity, runs, check results, findings, profiles).
  2. Artifact store on fsspec: bulky evidence behind `details_ref` (fuzzy pairs, full matrices) and tracebacks. `fuzzy_duplicates` keeps only its 20 closest pairs and the counts in its output (U4-10); store the full flagged-pair list behind `details_ref`.
  3. A per-run Parquet snapshot of the data as profiled (F-15).
  4. `--sink store`, `dsbf.store.save(run)` and `dsbf runs list | show | diff | delete | gc`.
  5. API v2 read endpoints, paginated.
  6. Job API and local runner with progress events.
  7. Generated TypeScript client, and the JavaScript-or-TypeScript decision.
  8. Dashboard shell and Run health panel (F-20).
  9. Findings explorer (N-A5).
  10. Column view with the profile's bins (F-12) and Q-Q plots (N-A3), reusing the `wip/echarts-charts` components.
  11. Relationships.
  12. Quality and ML readiness, on the outputs of `quality_dimensions` and `ml_readiness` (U4-15): levels, gate, per-input states and coverage. A dimension that is `not assessed` has no level and needs its own state in the tab, not green. Coverage is per input check; showing it per column needs the columns a check examined on `CheckResult`, a Run schema change to decide here.
  13. Time series (N-A4). The `seasonal_decomposition` check keeps only the trend and seasonal strengths (U4-11): store the observed, trend, seasonal and remainder series as an artifact, and the ACF and PACF values and bands, which the check already keeps in its output, so the tab can draw them.
  14. Run compare and history.
  15. Settings, then rerun.
  16. Frontend component and end-to-end tests (F-29, Playwright), with the lint warnings cleared.
  17. Code-split the bundle (over 500 kB; Plotly and ECharts load eagerly) and remove Plotly (N-A1).
  18. Bundle the dashboard into the wheel with `dsbf dashboard`, and release `1.0.0`.

## Phase 7: Scale and sources (placeholder)

- **Capability:** Larger-than-memory files and object storage (T2), warehouse sources with pushdown through Ibis (T3), recurring datasets with history and profile-based drift (T4), and scheduling recipes.
- **Motivation:** R3: from a laptop to cloud pipelines without redesigning the core (REFACTOR_PLAN §5.1).
- **Boundaries:** No team features (Phase 8) and no new modalities (Phase 9). Checks never see a path or a connection, only a `Dataset` and its profile.
- **Sequence and dependencies:** Depends on Phase 6 (history needs the store). T2, then T3 (DuckDB and Postgres first), then T4, then recipes.
- **Success criteria:** 1M × 50 within budget; a 100M-row Parquet glob profiled with a sample taken at read time and its provenance shown; one warehouse source verified end to end.

- **U7-00 · Plan Phase 7**.
  - **Carry in from U4-11:** the time-series checks analyse one series; `group_by_column` (a series per group, as in panel data) was reserved in the legacy config and never supported. Decide whether it is wanted.
  - **Carry in from U4-09:** `missingness_mechanism` reports per-column evidence only; a real Little's MCAR test (expectation-maximisation estimates of the mean and covariance of the numeric columns) is a candidate if a dataset-level verdict is wanted.
  - **Carry in from U0-15:** re-check ADR 0004, that the profile interface can be expressed as an Ibis aggregation (with the approximate-statistics caveats), before committing to the pushdown design; amend or supersede the ADR if it can't.
  - **Carry in from U2-01:** `dsbf.data.load` reads one local file, the first Excel sheet only, and rejects directories, globs and URLs (`UnsupportedSourceError`). Sheet choice, globs and `s3://`-style paths belong to T2.
  - **Carry in from U2-08:** on 1M × 50 floats the profile pass takes 9.5 s on the 4-CPU runner (type inference 1.5 s, `profile_dataset` 8.0 s). If the standard plan at that size shows the profile as a cost, the options are arithmetic histogram binning with an edge fix-up, a sketched distinct count above a row threshold, and sharing `n_unique` and null counts between type inference and the count pass. `Dataset.frame.estimated_size()` undercounts a CSV-loaded frame, so a memory figure should say what it measures.
  - **Carry in from U4-12:** drift approximated from stored profiles is the basis of history-based drift.
- Placeholders, in order: lazy and streaming scans with read-time sampling, globs and object storage (T2, N-B6) · Ibis sources on DuckDB and Postgres with profile pushdown (T3) · Snowflake and BigQuery · recurring datasets, history views and drift from stored profiles (T4) · `dsbf watch` and scheduling recipes (cron, orchestrators)

## Phase 8: Team and cloud (placeholder)

- **Capability:** A team deployment: the Postgres repository, object-storage artifacts, pluggable auth, a job queue, and the integrations that wrap the library.
- **Motivation:** R4: local first, team later, as new implementations behind the Phase 6 interfaces, not a rewrite.
- **Boundaries:** No new check logic; integrations are thin wrappers over the library and CLI.
- **Sequence and dependencies:** Depends on Phase 6, and on Phase 7 for warehouse sources in the integrations.
- **Success criteria:** A documented team deployment; each adapter has an example and a CI smoke test.

- **U8-00 · Plan Phase 8**.
- Placeholders, in order: Postgres repository · object-storage artifacts (S3, GCS, Azure) · auth (token, then OIDC) · database-backed job queue with an adapter interface · Docker image and Helm chart (REVIEW_FINDINGS part 2 E) · GitHub Action and pre-commit hook · Airflow, Prefect and Dagster adapters · dbt recipe · notifiers (webhook, Slack, email) and OpenLineage facets · MCP server (REVIEW_FINDINGS part 2 G)

## Phase 9: Modalities and workflow phases (placeholder)

- **Capability:** Non-tabular modalities (text corpora first), domain packs, modelling-prep exports and modelling diagnostics.
- **Motivation:** The README vision: from EDA to the next workflow phases, on more kinds of data (REFACTOR_PLAN §5.1 T5).
- **Boundaries:** Each modality is a new profiler and check pack behind the same contracts.
- **Sequence and dependencies:** Depends on Phase 4's checks and Phase 5's contracts; independent of Phases 7 and 8.
- **Success criteria:** Set by U9-00.

- **U9-00 · Plan Phase 9**.
- Placeholders, in order: text-corpus modality · domain packs (REVIEW_FINDINGS part 2 G) · modelling-prep exports (the ML guidance actions from U4-13 as sklearn pipelines) · modelling diagnostics
