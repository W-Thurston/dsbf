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

Phases 7–9 have a locked phase block and an ordered list of
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
| [6 · Store, service and dashboard v2](docs/HISTORY.md#phase-6-store-service-and-dashboard-v2-standalone-local) | 2026-10-29 | 20 (#147–#171) | A SQLite run store and fsspec artifact store with a data snapshot, `dsbf runs`, a local `/v2` API with a job runner, a first TypeScript dashboard with a page per area, bundled in the wheel and served by `dsbf dashboard`, and stable 1.0.0 run and table schemas. The review found the dashboard unusable as a visual exploration, so the `1.0.0` release moved to Phase 6b |

---

## Phase 6b: Dashboard design and the 1.0.0 release

- **Capability:** A dashboard designed around the questions a person asks of their data: a visual, chart-led exploration of a run (what is wrong, ranked; what each column looks like; how columns relate; what changed since the last run), with text as annotation, not as the content. Then release `1.0.0`.
- **Motivation:** Phase 6 rebuilt the dashboard page by page against the API, and each page's acceptance was that it showed the right data with honest empty states. Reviewing the result on 2026-10-29, the user found it unusable as a visual exploration of data: mostly tables, counts and cards, closer to a knowledge-base page than a dashboard, and less than the old dashboard it replaced (traffic-light quality indicators, per-column distributions, explanatory text). The phase block for Phase 6 had no visual acceptance, and passing Playwright tests did not measure it.
- **Boundaries:** Frontend design and build. The API, the store and the schemas change only where a page needs a field, and then additively (the run and table schemas are at 1.0.0). No new checks, no new sources (Phase 7), no team features (Phase 8). The bundling in the wheel, `dsbf dashboard` and the size budgets stay.
- **Sequence and dependencies:** Depends on Phase 6. Read the old dashboard first and use it as the design baseline, define what the dashboard must answer, then prototype before building, then a visual language, then the pages in the order a person uses them, then the release.
- **Success criteria:**
  - The user reviews a screenshot of every page, on real runs (titanic and the fixture store's archetypes), and signs each one off as readable and chart-led.
  - A person can answer the phase's questions from the dashboard without reading the JSON: can I trust this data at a glance; what is wrong, worst first; what does each column look like; how do the columns relate; is it ready to model and what preparation does it need; how does it behave over time; what changed since the last run (agreed in U6b-00).
  - Opening a run still transfers under 200 KB, and the bundle stays within its budget.
  - The perf, golden and calibration nightlies are green on the release commit, and `1.0.0` is on PyPI.
- **Decided on 2026-10-29, when the phase was added:** merge the Phase 6 work without tagging; the release moves to the end of this phase; the version stays `1.0.0a1` and the classifier `Alpha` until then. (Asked the user.)

### U6b-00 · Plan Phase 6b
- **Status:** done (2026-10-29, #173 `ba53f0a`) · **Size:** M · **Depends on:** Phase 6
- **Notes:** Read the old dashboard (`15a6ac1^`) in full and recorded a per-tab inventory (keep, change or drop, and `have`/`need` against `/v2`) in `docs/DASHBOARD_DESIGN.md`, which is now a planning document. Interviewed the user: seven questions the dashboard answers, the six old tabs as routes under a persistent health bar, the old look as the starting point, two additive API fields. Deviations: the old build was not run (the user chose to read the source); the visual language moved to U6b-01 and the prototypes to the page units, as the old dashboard already is the design.
- **Acceptance:** `ROADMAP.md` holds the phase's units in full format, each with a visual acceptance; the user agreed the questions; the inventory is recorded and cited by each page unit.

**Applies to every page unit below (decided in U6b-00):** the old dashboard is the baseline (`docs/DASHBOARD_DESIGN.md`, the unit's inventory section); a deviation from it is stated and justified in `PLAN.md` and in the inventory's decision column. A deviation whose readability on real data is uncertain is prototyped first (throwaway, shown to the user) before it is built. The page's checklist lines become Playwright tests, and its acceptance is a screenshot the user signs off, on titanic and on the archetype named in the unit, in both themes. Opening a run stays under 200 KB transferred, and the bundle stays within `frontend/scripts/check-bundle.mjs` (a unit that needs more raises the budget and says why).

### U6b-01 · Run shell, visual language and chart components
- **Status:** done (2026-10-30, #174 `f40b332`) · **Size:** L · **Depends on:** U6b-00
- **Notes:** Every run page now sits in one shell (header, five-dimension health bar with a grey unassessed state, six tab routes); `/columns` redirects to `/distributions`. Visual language and the nine chart components are recorded in `docs/DASHBOARD_DESIGN.md`; the `DsbfChart` chunk shrank to 670,670 bytes and now registers the visual map. Deviation: the ML readiness cards moved to their own tab, unchanged, so all six tabs exist.
- **Carry in from U6-12:** `DsbfChart` is one lazy ECharts chunk (690,875 bytes) registering bar, line, heatmap, scatter, boxplot and custom charts; keep it lazy and register only the chart types the design uses.
- **Goal:** The frame every run page shares, in the old dashboard's look: one persistent header (dataset › run, with run date, depth and sampling), the five-dimension health bar above the tabs on every tab (traffic-light dot, "All clear" or "N cols", a hover with the definition and "N of M columns", dimming of the dimensions irrelevant to the tab, and a grey state for an unassessed dimension), the six tabs as routes (Overview, Quality, Distributions, Relationships, ML Readiness, Time Series), and the theme toggle. Decide and record the visual language in `docs/DASHBOARD_DESIGN.md`: tokens (from the old slate palette, AA-checked by `tokens.test.ts`), the categorical and severity palettes, the chart types and their components, density and card layout. Build the shared chart components the pages use: histogram with reference lines, box plot, top-values bar, inline sparkline, heatmap, stacked strength bar, traffic-light dot and card, and level banner.
- **Assumptions to confirm:** the `quality_dimensions` check output gives every field the health bar needs (`level`, `assessed`, `coverage`, affected columns) from `/runs/{id}/checks/quality_dimensions` within the transfer budget; the router can nest the six tabs under `/runs/:runId` while keeping the Phase 6 URLs working or redirecting; the old palette passes AA in light mode.
- **Steps:** the visual-language section of the design document; tokens and the ECharts theme; the shell layout, header and health bar; the tab routes, replacing the line of links on the run page; the chart components, each with a unit test; the chart chunk's registrations trimmed to what the components use.
- **Acceptance:** The user signs off a screenshot of the shell (header, health bar, tabs) on titanic and on `high_missingness`, in both themes, and the visual-language section of the design document; the health bar shows a grey dimension on a run without a target.

### U6b-02 · Column sketches and dataset dimension levels in the API
- **Status:** done (2026-10-30, #175 `a56f93b`) · **Size:** M · **Depends on:** U6b-00
- **Notes:** `/columns` carries a `sketch` per column and `DatasetSummary` carries `dimension_levels`; titanic's `/columns` is 5.2 KB and `wide`'s 34 KB. A numeric column has a null `mode` (the profile stores none). `API_VERSION` rose to 1.9.0, as the breaking-change gate requires for additive fields.
- **Goal:** The two additive fields the inventory needs (`docs/DASHBOARD_DESIGN.md`, "API needs"). (1) `/runs/{id}/columns` gains a compact sketch per column: mode and mode share, skewness, min and max, and a small distribution (a coarse histogram for numeric columns, the top few values for categorical ones), so the Overview's column table and its inline charts need one request. (2) `DatasetSummary` gains the latest run's five quality-dimension levels, for the dots on the dataset list.
- **Assumptions to confirm:** the sketch can be derived from the stored profile without the snapshot; titanic's `/columns` response stays near 12 KB and an archetype with many columns (`wide`) stays within the 200 KB budget for opening a run; the dimension levels can be read from the stored `quality_dimensions` result when the dataset list is built (or must be stored with the run summary).
- **Steps:** the response models and their docstrings; the repository and service code; the generated OpenAPI types (`frontend/src/generated/api.d.ts`) through their owning command; tests at the service boundary; payload sizes measured on titanic and `wide`.
- **Acceptance:** Both fields are served and typed, measured payloads are recorded, and the run and table schemas are unchanged (the fields are API views, not run-schema fields). No screenshot: the fields are signed off through U6b-03 and U6b-10.

### U6b-03 · Overview tab, chart-led
- **Status:** done (2026-10-30, #176 `ba74641`) · **Size:** L · **Depends on:** U6b-01, U6b-02
- **Notes:** The Overview is a metric strip, findings by dimension and severity, column types, a sortable and filterable column table with an inline chart per row, missing values with a null co-occurrence matrix, the sample, run health and dtype savings; lineage moved into the header. Deviations: a new `GET /v2/runs/{id}/dataset` (memory and null pairs were not served; API 1.10.0); the page's charts are CSS and SVG, because ECharts pushed titanic to 329 KB, and the service now gzips (titanic transfers 102 KB); bundle budget 900 KB; top findings show 5.
- **Carry in from U6-19:** run-health and findings views that show counts could chart them instead; the dashboard is served uncompressed (add gzip if a page pushes the 200 KB transfer budget; titanic is 151 KB).
- **Carry in from U6b-01:** the run page still shows the Phase 6 cards (policy, run health, finding counts, top findings) under a mostly empty "Run" card with the source path, a Findings link and the compare link; redesign them chart-led, and keep the compare link's "previous run" text (it appears once the previous-run lookup finishes).
- **Goal:** The Overview answers "can I trust this data at a glance, and what is wrong, worst first?" (inventory: Overview). The metric strip (rows, columns, memory, missing %, duplicate rows, sample adequacy coloured by verdict, depth, sampling); the dataset sample from the snapshot, with an empty state without one; the column table with the old coloured cells (missing % at 5% and 20%, mode share at 70% and 90%, skew beyond ±1), type badges, sorting and filtering, and an inline chart per row from the column sketch; the column-types bar; missingness as a per-column bar plus a null co-occurrence heatmap; the ranked data signals (findings) with their counts charted by severity and dimension, linking to the findings explorer; run health and policy with their counts charted; dtype optimisations. Lineage folds into the header (location, format, and whether the content changed since the previous run).
- **Assumptions to confirm:** the null co-occurrence pairs (`missingness.pairs`) are dense enough on titanic and `high_missingness` to make a readable heatmap (prototype it first if not); the inline row charts stay readable at the table's row height.
- **Steps:** the page and its parts; the charts; the links to the findings explorer and the column pages; Playwright tests for the page's checklist lines; transfer and bundle sizes measured.
- **Acceptance:** The user signs off a screenshot of the Overview on titanic and `high_missingness`, in both themes, as readable and chart-led; opening titanic's run transfers under 200 KB.

### U6b-04 · Quality tab
- **Status:** done (2026-10-30) · **Size:** M · **Depends on:** U6b-01
- **Notes:** Quality is a trust banner, five traffic-light cards that open their sections, collapsible sections with sortable findings tables whose rows fetch their evidence and guidance on expand, clean columns, the missingness mechanism with predictor bars, fuzzy duplicates and the coverage; the Run health card folds its not-applicable checks. Deviations: sample values show only where evidence carries them; Cramér's V bars are unsigned; bundle budget 920 KB. Follow-up: hide the near-duplicate card on runs without that check (U6b-12).
- **Goal:** Quality answers "what is wrong with this data, by dimension?" (inventory: Quality). The trust banner (overall level, its label and sentence, columns checked, fully clean, dimensions flagged of those assessed); five traffic-light dimension cards that scroll to their sections; a collapsible section per dimension with a sortable findings table whose expanded rows show the finding's evidence, sample values and guidance; clean columns; the missingness mechanism with its predictor bars; fuzzy duplicates with similarity bars; and the check coverage, beside the dimension coverage. The "mark as reviewed" star is dropped (decided in U6b-00: unsaved, it was lost on reload).
- **Assumptions to confirm:** findings' `evidence` carries the per-issue metrics the old detail column showed (missing share, violation count, zero share, mode share, unique count, VIF, correlation); `missingness_mechanism` predictors give a signed effect for the bars.
- **Steps:** the page and its parts; Playwright tests for the quality checklist lines; the `clean` and `high_missingness` fixtures.
- **Acceptance:** The user signs off a screenshot of Quality on titanic and `high_missingness`, in both themes.

### U6b-05 · Findings explorer
- **Status:** todo · **Size:** S · **Depends on:** U6b-01
- **Goal:** The findings explorer (a Phase 6 page with no old counterpart; closest to the old data signals and Quality sections) in the new shell and visual language: severity and dimension counts charted above the filters, every finding count elsewhere linking into it with its filter set, and the detail showing guidance as the Quality tab does.
- **Assumptions to confirm:** the explorer's filter state is already in the URL, so counts on other pages can link to a filtered view.
- **Steps:** the page restyled; the count chart; links from the Overview, Quality and ML Readiness counts; Playwright tests.
- **Acceptance:** The user signs off a screenshot of the explorer on titanic, filtered and unfiltered, in both themes.

### U6b-06 · Distributions tab
- **Status:** todo · **Size:** L · **Depends on:** U6b-01
- **Goal:** Distributions answers "what does each column look like?" (inventory: Distributions). The column browser grouped by type, with a warning dot and a missing % bar per column; the column heading; the stats strip by type; the distribution chart from the profile's real histogram, with mean and median lines and a box-plot toggle for numeric columns and a top-values bar for categorical ones; the column's guidance beside the chart; the bimodal callout; percentiles or value counts and text lengths; outlier analysis (IQR, z-score, MAD and their consensus); normality with the Q-Q plot built from `QQData` (the old dashboard's placeholder); the top correlations, each selecting that column. The column page from Phase 6 becomes this tab's detail.
- **Assumptions to confirm:** every numeric profile has a histogram (or say when it does not); whether the bimodal callout should overlay the two fitted components on the histogram (prototype it on a bimodal archetype; ask the user).
- **Steps:** the tab and its parts; the Q-Q chart; Playwright tests for the column checklist lines.
- **Acceptance:** The user signs off a screenshot of a numeric and a categorical column on titanic, and of a skewed or bimodal column on an archetype, in both themes.

### U6b-07 · Relationships tab
- **Status:** todo · **Size:** M · **Depends on:** U6b-01
- **Carry in from U6-19:** the relationships checklist lines for the `severe_multicollinearity`, `wide`, `all_categorical` and `single_column` archetypes were never built as Playwright tests; they need fixture stores in `frontend/e2e/fixture_store.py`.
- **Carry in from U6b-03:** the Overview draws bars and a null matrix in CSS (`BarList`, `NullMatrix`); use them where a chart here needs no zoom or export, to keep the chart chunk off the page.
- **Carry in from U6b-01:** the matrix heatmap in `relationships.ts` used a visual map that `DsbfChart` never registered, so its colour scale was not applied until U6b-01 registered it; check its colours in the screenshot and move it onto `HeatmapChart`.
- **Goal:** Relationships answers "how do the columns relate?" (inventory: Relationships). The summary with the stacked strength bar and the collinearity and leakage chips that filter the browser; the correlation heatmap (the old dashboard's placeholder) from the relationship matrix; the column browser and the associations table with type and strength filters; the pair detail (stats, warnings, pair plot). The `multivariate_outliers` result is shown here or on Quality; the unit decides and states why.
- **Assumptions to confirm:** the heatmap stays readable on `wide` (cap, cluster or scroll; prototype if not obvious); the four archetypes can be generated for the fixture store.
- **Steps:** the tab and its parts; the heatmap; the archetype fixture stores; Playwright tests for the carried-in checklist lines.
- **Acceptance:** The user signs off a screenshot of Relationships on titanic and `severe_multicollinearity`, in both themes.

### U6b-08 · ML Readiness tab
- **Status:** todo · **Size:** M · **Depends on:** U6b-01
- **Goal:** ML Readiness answers "is it ready to model, and what preparation does it need?" (inventory: ML Readiness). The gate banner (ready, needs work, not ready, with its caveat when a dimension is unassessed); the five dimension cards; a section per dimension whose expanded rows show the finding, its metrics and its suggested actions, each with *what it does*, *trade-off*, *after this* and a reference link, ported from the old dashboard's 38-entry action dictionary (`15a6ac1^:frontend/src/utils/actionMeta.js`) as a frontend lookup by method (decided in U6b-00); the transformation preview as before and after skewness bars per candidate; ready features.
- **Assumptions to confirm:** the old dictionary's method keys match the `method` values `/v2` guidance emits (list the mismatches and the methods with no entry); the transformation candidates' names match the dictionary's transform entries.
- **Steps:** the dictionary, with a test that every emitted method has an entry or is listed as uncovered; the tab and its parts; Playwright tests for the ML readiness checklist lines.
- **Acceptance:** The user signs off a screenshot of ML Readiness on titanic and on an archetype that needs work, in both themes.

### U6b-09 · Time Series tab
- **Status:** todo · **Size:** S · **Depends on:** U6b-01
- **Goal:** Time Series answers "how does it behave over time?" (inventory: Time Series): the disabled and no-index states with the config to add; the header strip (index column, frequency, value columns); ACF and PACF with confidence bands and significant lags; stationarity with its reading note; STL decomposition, in the new shell and visual language.
- **Assumptions to confirm:** the fixture store has a run with a time index (or one is added).
- **Steps:** the tab restyled from the Phase 6 page; Playwright tests.
- **Acceptance:** The user signs off a screenshot of Time Series on a run with a time index and on titanic (the empty state), in both themes.

### U6b-10 · Dataset list, runs, compare and history
- **Status:** todo · **Size:** M · **Depends on:** U6b-01, U6b-02
- **Goal:** The dataset-level pages answer "what changed since the last run?" and get a person to a run. The dataset list as the old card grid with search, with the latest run's five dimension dots in place of the old score; a dataset's runs; compare (any two runs) with its changes charted; finding history and lineage, in the new shell and visual language.
- **Assumptions to confirm:** the compare view's diff sections can be summarised as counts per kind of change for a chart.
- **Steps:** the pages restyled; the dimension dots; Playwright tests.
- **Acceptance:** The user signs off screenshots of the dataset list (fixture store), a run list, and titanic's full run compared with its sample, in both themes.

### U6b-11 · Settings and rerun
- **Status:** todo · **Size:** S · **Depends on:** U6b-01
- **Goal:** The dataset settings form and rerun progress in the new shell and visual language.
- **Assumptions to confirm:** the generated settings form needs no change beyond styling.
- **Steps:** the page restyled; Playwright tests still green.
- **Acceptance:** The user signs off a screenshot of the settings page and a rerun in progress, in both themes.

### U6b-12 · Screenshot review on the fixture store
- **Status:** todo · **Size:** M · **Depends on:** U6b-03 to U6b-11
- **Carry in from U6b-04:** the Quality tab's Near-duplicate rows card says "did not run" on every run without that check; hide an absent optional check's card.
- **Carry in from U6-19:** the quality lines for the other archetypes still need fixture stores.
- **Carry in from U6b-01:** `e2e/run-shell.spec.ts` saves the shell screenshots when `SCREENSHOT_DIR` is set; extend that pattern to every page.
- **Goal:** A review of every page on the fixture store, with the archetype stores, against the phase's questions: a screenshot set the user signs off, the remaining archetype checklist lines as Playwright tests, and the fixes the review finds (each small, or a follow-up unit).
- **Assumptions to confirm:** every archetype generator in `tests/` can be stored in the fixture store within the e2e job's time.
- **Steps:** the remaining archetype fixture stores; a script that captures every page in both themes; the review with the user; the fixes.
- **Acceptance:** The user signs off the full screenshot set, and confirms each of the phase's seven questions can be answered from the dashboard without reading the JSON.

### U6b-13 · Release 1.0.0
- **Status:** todo · **Size:** M · **Depends on:** U6b-12
- **Carry in from U6-19:** `test_a_killed_worker_has_no_cpu_figure` (`tests/test_engine/test_process_workers.py:357`) failed once on the Python 3.13 full-suite job of #171 (`assert 0.5992 >= 0.6`: the wall time of a worker killed at a 0.6 s timeout, measured on a different clock from the timeout) and passed on rerun. Probable cause, not reproduced: clock jitter. Give the assertion a small tolerance (for example `>= 0.55`) before the release gate relies on the full suite; treat a second failure as a real defect. The wheel smoke script installs `[server]` by `file://` URL because Git Bash on Windows does not convert a path with `[server]` attached; keep that when the script changes.
- **Goal:** Release `1.0.0`. Set the version to `1.0.0` and the classifier to `Development Status :: 5 - Production/Stable`; rename the changelog's `[Unreleased]` section to `[1.0.0]` with its date and link; restore the README's installation text and the example workflow's pin for a stable release (`pip install dsbf`, `dsbf>=1.0,<2`) and remove the preview notes from the README and the standalone-local guide; reword `CLAUDE.md` practice 11 for the stable 1.x guarantee; replace the README screenshot with the signed-off Overview. The user pushes the `v1.0.0` tag after merge (the release workflow builds the dashboard into the wheel, publishes to TestPyPI, installs it back and then publishes to PyPI).
- **Assumptions to confirm:** the release workflow still matches what U6-19 left; the perf, golden and calibration nightlies are green on `main`.
- **Steps:** the flaky assertion's tolerance; the version, classifier, changelog, README, guide and `CLAUDE.md` changes; the nightlies run on the release commit.
- **Acceptance:** The phase's success criteria hold, each measured in the PR or the release run, and `1.0.0` is on PyPI.

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
- **Carry in from U6b-00:** the old dashboard let a person star a finding as reviewed, unsaved; Phase 6b dropped it. Saving reviewed findings per user belongs with auth.
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
