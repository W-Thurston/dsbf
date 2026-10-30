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

Phase 6b (the dashboard design and the release) and Phases 7–9 have a locked phase block and an ordered list of
placeholders; each starts with a `Ux-00` planning unit that turns the list into full units, using
what earlier phases taught.

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
| None at present | | |

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
| [6 · Store, service and dashboard v2](docs/HISTORY.md#phase-6-store-service-and-dashboard-v2-standalone-local) | 2026-10-29 | 20 (#147–#170, and U6-19) | A SQLite run store and fsspec artifact store with a data snapshot, `dsbf runs`, a local `/v2` API with a job runner, a first TypeScript dashboard with a page per area, bundled in the wheel and served by `dsbf dashboard`, and stable 1.0.0 run and table schemas. The review found the dashboard unusable as a visual exploration, so the `1.0.0` release moved to Phase 6b |

---

## Phase 6b: Dashboard design and the 1.0.0 release

- **Capability:** A dashboard designed around the questions a person asks of their data: a visual, chart-led exploration of a run (what is wrong, ranked; what each column looks like; how columns relate; what changed since the last run), with text as annotation, not as the content. Then release `1.0.0`.
- **Motivation:** Phase 6 rebuilt the dashboard page by page against the API, and each page's acceptance was that it showed the right data with honest empty states. Reviewing the result on 2026-10-29, the user found it unusable as a visual exploration of data: mostly tables, counts and cards, closer to a knowledge-base page than a dashboard, and less than the old dashboard it replaced (traffic-light quality indicators, per-column distributions, explanatory text). The phase block for Phase 6 had no visual acceptance, and passing Playwright tests did not measure it.
- **Boundaries:** Frontend design and build. The API, the store and the schemas change only where a page needs a field, and then additively (the run and table schemas are at 1.0.0). No new checks, no new sources (Phase 7), no team features (Phase 8). The bundling in the wheel, `dsbf dashboard` and the size budgets stay.
- **Sequence and dependencies:** Depends on Phase 6. Read the old dashboard first and use it as the design baseline, define what the dashboard must answer, then prototype before building, then a visual language, then the pages in the order a person uses them, then the release.
- **Success criteria:**
  - The user reviews a screenshot of every page, on real runs (titanic and the fixture store's archetypes), and signs each one off as readable and chart-led.
  - A person can answer the phase's questions (decided in U6b-00) from the dashboard without reading the JSON.
  - Opening a run still transfers under 200 KB, and the bundle stays within its budget.
  - The perf, golden and calibration nightlies are green on the release commit, and `1.0.0` is on PyPI.
- **Decided on 2026-10-29, when the phase was added:** merge the Phase 6 work without tagging; the release moves to the end of this phase; the version stays `1.0.0a1` and the classifier `Alpha` until then. (Asked the user.)

### U6b-00 · Plan Phase 6b
- **Status:** todo · **Size:** M · **Depends on:** Phase 6
- **Goal:** Turn the placeholders into units. First read the old dashboard in git history in full (see **Design baseline** below) and write down, per tab, what it showed, which chart or indicator carried each message, and what to keep, change or drop. Then interview the user (the grilling skill) for the jobs the dashboard must do and the questions each page must answer, starting from that inventory; prototype the overview, the column page and the quality page as throwaway visuals the user reacts to (the prototype skill); decide the visual language (tokens, chart types, density) and whether each page needs an API field it lacks.
- **Carry in from U6-19:** the dashboard's checklist lines for the `severe_multicollinearity`, `wide`, `all_categorical` and `single_column` archetypes and the other archetypes' quality lines were never built as Playwright tests (they need fixture stores in `frontend/e2e/fixture_store.py`); the dashboard is served uncompressed (add gzip if a page pushes the 200 KB transfer budget; titanic is 151 KB); run-health and findings views that show counts could chart them instead.
- **Carry in from U6-19:** `test_a_killed_worker_has_no_cpu_figure` (`tests/test_engine/test_process_workers.py`) failed once on the Python 3.13 full-suite job of #171 (`assert 0.5992 >= 0.6`: the wall time of a worker killed at a 0.6 s timeout, measured on a different clock from the timeout) and passed on rerun; the PR changed no engine code. Probable cause, not reproduced: clock jitter. Give the assertion a small tolerance (for example `>= 0.55`) before the release gate relies on the full suite; treat a second failure as a real defect. The wheel smoke script installs `[server]` by `file://` URL because Git Bash on Windows does not convert a path with `[server]` attached; keep that when the script changes.
- **Carry in from U6-12:** `DsbfChart` is one lazy ECharts chunk (about 690 KB); keep it lazy, and register only the chart types the design uses.
- **Design baseline: the old dashboard in git history.** Every design decision in this phase starts from the old dashboard, and a deviation from it is stated and justified. The old dashboard is the reference for what a person got from DSBF's dashboard, and the user's review judged the Phase 6 rebuild against it. Where to read it:
  - Source: `git ls-tree -r 15a6ac1^ -- frontend/src` (the last commit before the TypeScript rebuild). Read it with `git show 15a6ac1^:<path>`. The tabs are `views/tabs/{Overview,Quality,Distributions,Relationships,MlReadiness,TimeSeries}Tab.vue`; the parts are `components/DataHealthBar`, `overview/AlertsPanel`, `ColumnMetadataTable`, `DataSampleTable`, `DataLineageCard` and `components/charts/{BoxPlot,DistributionHistogram,ValueCounts}Chart` over `DsbfChart`.
  - Running build: tag `v0.20.0` (`c880348`, 2026-05-06), the last release where the old frontend ran, against the legacy API. Check it out in a worktree outside the repo and run it on titanic, cities and one archetype table (see its README: `poetry install`, `uvicorn dsbf.api.main:app`, `npm run dev` in the frontend directory). The legacy engine and API v1 were deleted in Phase 4, so `main` cannot run it.
  - Screenshots: `docs/assets/Overview_Tab.png`, plus new ones the user takes from the `v0.20.0` build for the Quality, Distributions and Relationships tabs (they go in `docs/assets/` or the unit's notes as the baseline for the sign-off screenshots).
  - Description: the `v0.20.0` README lists the six tabs: Overview (five traffic lights and a trust banner), Distributions (outliers, normality, skewness, transformation previews), Relationships (association table, correlation heatmap, fuzzy duplicates), Quality (severity-ranked rows with remediation), ML Readiness (five dimensions and a gate banner) and Time Series (ACF and PACF, stationarity, decomposition).
  - What is not carried over: the old code (it was built on API v1 and the legacy result shape) and anything the `/v2` API cannot supply. The inventory marks each old element as `have` (the API has the data), `need` (an additive field) or `drop`.
- **Acceptance:** `ROADMAP.md` holds the phase's units in full format, each with a visual acceptance (a screenshot the user signs off), the user has agreed the questions the dashboard answers, and the old-dashboard inventory (per tab: keep, change, drop, and `have`/`need` against the API) is recorded in the unit's notes and cited by each page unit.

- Placeholders, in order: visual language and chart components · overview and Run health, chart-led · findings explorer · column view and distributions · relationships · quality and ML readiness (traffic lights and gates) · time series · compare and history · settings and rerun · a screenshot-based review of every page on the fixture store, with the archetype stores · **release `1.0.0`**.
- **Release unit (last), carried from U6-19:** set the version to `1.0.0` and the classifier to `Development Status :: 5 - Production/Stable`; rename the changelog's `[Unreleased]` section to `[1.0.0]` with its date and link; restore the README's installation text and the example workflow's pin for a stable release (`pip install dsbf`, `dsbf>=1.0,<2`) and remove the preview notes from the README and the standalone-local guide; reword `CLAUDE.md` practice 11 for the stable 1.x guarantee; update the README screenshot; the user pushes the `v1.0.0` tag after merge (the release workflow builds the dashboard into the wheel, publishes to TestPyPI, installs it back and then publishes to PyPI).

## Phase 7: Scale and sources (placeholder)

- **Capability:** Larger-than-memory files and object storage (T2), warehouse sources with pushdown through Ibis (T3), recurring datasets with history and profile-based drift (T4), and scheduling recipes.
- **Motivation:** R3: from a laptop to cloud pipelines without redesigning the core (REFACTOR_PLAN §5.1).
- **Boundaries:** No team features (Phase 8) and no new modalities (Phase 9). Checks never see a path or a connection, only a `Dataset` and its profile.
- **Sequence and dependencies:** Depends on Phase 6 (history needs the store). T2, then T3 (DuckDB and Postgres first), then T4, then recipes.
- **Success criteria:** 1M × 50 within budget; a 100M-row Parquet glob profiled with a sample taken at read time and its provenance shown; one warehouse source verified end to end.

- **U7-00 · Plan Phase 7**.
  - **Status:** todo
  - **Carry in from U6-19:** `gc` leaves a dataset's settings overlay behind when it deletes the dataset's last run.
  - **Carry in from U4-19, U5-09 and U6-00:** the `sql` extra has no importer, install-matrix leg or `unavailable` test; add all three with the first Ibis source. (The `server` extra gets its pair in U6-04.)
  - **Carry in from U4-11:** the time-series checks analyse one series; `group_by_column` (a series per group, as in panel data) was reserved in the legacy config and never supported. Decide whether it is wanted.
  - **Carry in from U4-09:** `missingness_mechanism` reports per-column evidence only; a real Little's MCAR test (expectation-maximisation estimates of the mean and covariance of the numeric columns) is a candidate if a dataset-level verdict is wanted.
  - **Carry in from U0-15:** re-check ADR 0004, that the profile interface can be expressed as an Ibis aggregation (with the approximate-statistics caveats), before committing to the pushdown design; amend or supersede the ADR if it can't.
  - **Carry in from U2-01:** `dsbf.data.load` reads one local file, the first Excel sheet only, and rejects directories, globs and URLs (`UnsupportedSourceError`). Sheet choice, globs and `s3://`-style paths belong to T2.
  - **Carry in from U2-08:** on 1M × 50 floats the profile pass takes 9.5 s on the 4-CPU runner (type inference 1.5 s, `profile_dataset` 8.0 s). If the standard plan at that size shows the profile as a cost, the options are arithmetic histogram binning with an edge fix-up, a sketched distinct count above a row threshold, and sharing `n_unique` and null counts between type inference and the count pass. `Dataset.frame.estimated_size()` undercounts a CSV-loaded frame, so a memory figure should say what it measures.
  - **Carry in from U4-12:** drift approximated from stored profiles is the basis of history-based drift.
  - **Carry in from U6-18:** `RunDiff` reports count-scaled statistics as changed when two runs differ in sample size (titanic's full run against its 300-row sample: 101 changed findings). Investigate a per-metric tolerance or a list of metrics that scale with n; either changes what `RunDiff` reports, not its shape, so it fits within 1.x (`docs/guides/compatibility.md`, known limitations).
- Placeholders, in order: lazy and streaming scans with read-time sampling, globs and object storage (T2, N-B6) · Ibis sources on DuckDB and Postgres with profile pushdown (T3) · Snowflake and BigQuery · recurring datasets, history views and drift from stored profiles (T4) · `dsbf watch` and scheduling recipes (cron, orchestrators)

## Phase 8: Team and cloud (placeholder)

- **Capability:** A team deployment: the Postgres repository, object-storage artifacts, pluggable auth, a job queue, and the integrations that wrap the library.
- **Motivation:** R4: local first, team later, as new implementations behind the Phase 6 interfaces, not a rewrite.
- **Boundaries:** No new check logic; integrations are thin wrappers over the library and CLI.
- **Sequence and dependencies:** Depends on Phase 6, and on Phase 7 for warehouse sources in the integrations.
- **Success criteria:** A documented team deployment; each adapter has an example and a CI smoke test.

- **U8-00 · Plan Phase 8**.
- **Carry in from U6-05:** `Run.artifact` reads from a local directory only; object-storage artifacts need a loader hook on `Run` so a run read from a non-local store resolves its payloads.
- Placeholders, in order: Postgres repository · object-storage artifacts (S3, GCS, Azure) · auth (token, then OIDC) · database-backed job queue with an adapter interface · Docker image and Helm chart (REVIEW_FINDINGS part 2 E) · GitHub Action and pre-commit hook · Airflow, Prefect and Dagster adapters · dbt recipe · notifiers (webhook, Slack, email) and OpenLineage facets · MCP server (REVIEW_FINDINGS part 2 G)

## Phase 9: Modalities and workflow phases (placeholder)

- **Capability:** Non-tabular modalities (text corpora first), domain packs, modelling-prep exports and modelling diagnostics.
- **Motivation:** The README vision: from EDA to the next workflow phases, on more kinds of data (REFACTOR_PLAN §5.1 T5).
- **Boundaries:** Each modality is a new profiler and check pack behind the same contracts.
- **Sequence and dependencies:** Depends on Phase 4's checks and Phase 5's contracts; independent of Phases 7 and 8.
- **Success criteria:** Set by U9-00.

- **U9-00 · Plan Phase 9**.
- Placeholders, in order: text-corpus modality · domain packs (REVIEW_FINDINGS part 2 G) · modelling-prep exports (the ML guidance actions from U4-13 as sklearn pipelines) · modelling diagnostics
