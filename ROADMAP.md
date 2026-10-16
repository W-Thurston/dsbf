# DSBF Roadmap

This is the backlog of upcoming work. Each **phase** is a program: its capability, motivation,
boundaries, sequence and success criteria are locked at the top of the phase. Work inside a phase
is broken into **units**. A unit is one branch, one PR, one commit and one session. The workflow for running a unit is in [`CLAUDE.md`](CLAUDE.md), and the active unit (at
most one) lives in [`PLAN.md`](PLAN.md).

- **Design:** [`docs/REFACTOR_PLAN.md`](docs/REFACTOR_PLAN.md)
- **Findings cited as F-xx / N-xx:** [`docs/REVIEW_FINDINGS.md`](docs/REVIEW_FINDINGS.md)
- **How a run works today, with measurements:** [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md)

**Status values:**
- `todo`: ready once its dependencies are `done`
- `active`: currently in `PLAN.md`
- `done`: merged; the entry carries notes and the PR number. When a phase's last unit closes,
  its unit entries are condensed into a short program summary (see `.claude/commands/closeout.md`).
- `placeholder`: a later phase, detailed by that phase's planning unit

**Next unit:** the first `todo` unit, in order, whose dependencies are all `done`.

Phases 0–3 are broken down in full. Phases 4–9 are placeholders. Each starts with a `Ux-00`
planning unit that turns the placeholders into full units, using what earlier phases taught.

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
| `polars <2` (PR #46, closed 2026-10-16) | 2.0 changes dataframe-layer behaviour; ADR 0003 makes Polars the data layer in Phase 2 | U2-01 |
| `pandas <3` (PR #47, closed 2026-10-16) | 3.0 fails the 3.12 and 3.13 test jobs (string dtype, copy-on-write); the legacy tasks run on pandas until they migrate | U2-01 (pandas 3 as an input type); U4-E (the base cap) |
| `statsmodels <0.15` | 0.15's `variance_inflation_factor` includes the intercept: VIFs drop and four columns on the `clean` archetype stop being flagged | U4-B (VIF redesign, F-32) |
| `scikit-learn <1.8` | 1.8 changes `GaussianMixture` log-likelihood on discrete columns (titanic `pclass` BIC 1016 → 1763), so `detect_bimodal_distribution` reports other BIC values and finding titles | U4-B (bimodality is an analysis check) |

`.github/dependabot.yml` carries an `ignore` entry for each row, so Dependabot doesn't re-raise the held-back version. Removing a cap removes its entry.

---

## Phase 0: Stabilise and baseline

**Done (2026-10-16), 16 units.** Detailed unit entries were condensed here at close-out; the per-unit notes that later units depend on were carried into those units as "Note from U0-xx" lines.

- **Capability delivered:** a trustworthy, measured baseline on the target toolchain. CI runs lint (ruff from `uv.lock`), type check (pyrefly with a baseline), the fast gate and the full suite on Python 3.12 and 3.13, CodeQL, and the `golden` and `perf` workflows (nightly, on demand and on labelled PRs). Dependencies are managed with uv. Tests run hermetically from a temporary directory, every registered task completes on a hostile frame, and ADRs D1–D11 are accepted.
- **Success criteria, measured:**
  - Every Phase 0 unit is `done`: 16 of 16.
  - Fast gate (`unit and not slow`) 29 s; full suite 5 min 14 s (858 tests, U0-06). The `titanic` quickstart from a temporary directory went from 4 failed tasks to 0 (U0-07); `kitchen5k` from 3 failed tasks to 0, with 5 skipped by design (U0-09).
  - Golden baselines: ten snapshots (the 8 archetypes, `titanic`, `kitchen5k`), 60 successful and 5 skipped tasks each, 0 errors; two full runs matched byte for byte (U0-13). Re-checked on 2026-10-16 against the upgraded dependency set: 10 of 10 pass.
  - Performance baseline, taken on `ubuntu-latest` (4 CPUs): `titanic` 26 s, adversarial frame 16 s, `kitchen5k` 147 s, `kitchen50k` 357 s with a 2.08 GB peak, 0 failed tasks (U0-14).
  - pyrefly baseline: 895 existing errors across `dsbf/`, `tests/` and `docs/practice_run/` (U0-12); 0 new.
  - ADRs D1–D11 are `Accepted`: [0001](docs/adr/0001-library-first-layering.md), [0002](docs/adr/0002-findings-as-primary-output.md), [0003](docs/adr/0003-polars-arrow-data-representation.md), [0004](docs/adr/0004-ibis-warehouse-pushdown.md), [0005](docs/adr/0005-pydantic-contracts.md), [0006](docs/adr/0006-report-only-failure-semantics.md), [0007](docs/adr/0007-entry-point-plugins.md), [0008](docs/adr/0008-sqlalchemy-alembic-fsspec-storage.md), [0009](docs/adr/0009-single-distribution-with-extras.md), [0010](docs/adr/0010-check-naming.md), [0011](docs/adr/0011-toolchain.md).
  - Security alerts: 0 open Dependabot alerts (2026-10-16). Code-scanning alerts 3–10 are fixed in the U0-16 PR and close when CodeQL re-scans `main` after it merges; the next `/start-unit` re-lists them and records the result here.
- **Key decisions:** uv with the `uv_build` backend and `uv.lock` (ADR 0011); pyrefly as a pinned dev dependency, `default` preset, committed baseline that only shrinks; Python 3.14 out of the CI matrix (nine locked packages had no cp314 wheel); golden runs pin `PYTHONHASHSEED=0` and rank value counts with a stable sort (`ranked_value_counts`); `titanic` is vendored for the golden suite (seaborn-data has no license); perf numbers are compared only against the CI-hardware baseline; U0-16 closed the code-scanning alerts by returning generic API error text and logging the exception, and by giving the workflows `permissions: contents: read`.
- **Deviations from the phase block:**
  - The phase promised "no new features" and cheap safety fixes only. U0-09 also fixed `RichHandler(markup=True)`, which made seven tasks crash on column names such as `[/x]` (the largest crash cause); U0-13 fixed unstable tie ordering in three tasks; U0-06 added API tests.
  - "Dependabot backlog merged" took two units: U0-04 and U0-16. U0-16 was widened on 2026-10-16 from the security alerts to the whole backlog, so Phase 1 starts with no open PR. It took #34, #38, #42, #43 and #45, and closed #46 (polars 2) and #47 (pandas 3) as deferred.
  - The golden suite showed that two upgrades in #43 change results, so they stay behind caps: `statsmodels <0.15` (VIF includes the intercept) and `scikit-learn <1.8` (`GaussianMixture` log-likelihood on discrete columns). With `pandas <3` and `polars <2` they are listed under "Dependency upgrades" at the top of this file, each with the unit that revisits it, and mirrored by `ignore` entries in `.github/dependabot.yml`.
  - The README's examples were all rebuilt (U0-11) because most no longer ran.
- **PRs:** #29, #31, #32, #40, #41, #48, #50, #51, #52, #53, #54, #55, #56, #57, #58, and the U0-16 PR (number and merge SHA are filled in by the next `/start-unit`). Dependency PRs merged along the way: #23, #26, #24, #28, #27, #20, #22 and #36 (U0-04); #38, #42, #45, #34 and #43 (U0-16).

---

## Phase 1: Contracts

- **Capability:** The `src/` layout, a top-level frontend on pnpm, runtime paths under `DSBF_HOME`, optional extras, and the versioned result contracts (`dsbf.model`) with JSON Schema and TypeScript export.
- **Motivation:** All three usage modes depend on stable, published contracts (REFACTOR_PLAN §4). Embedded users consume the schema without the rest of DSBF.
- **Boundaries:** Packaging, paths and models only. No profiling or engine code. The legacy engine keeps working throughout.
- **Sequence and dependencies:** Depends on Phase 0. Layout and toolchain first, then contracts from the leaves (evidence) to the root (`Run`), then export.
- **Success criteria:** A hand-built `Run` validates, round-trips, and exports as JSON Schema and TypeScript, and the breaking-change gate is active in CI.

### U1-01 · Move the Python package to a `src/` layout
- **Status:** todo · **Size:** M · **Depends on:** U0-15 · **Resolves:** groundwork (REFACTOR_PLAN §3.1)
- **Goal:** `dsbf/` lives at `src/dsbf/` with no behaviour change.
- **Assumptions to confirm:**
  - Code that relies on the repo layout: `Path(__file__).parents[2]` in `storage/schema.py`, `api/main.py` and `api/db.py`, plus test paths.
  - The build backend's package discovery config.
  - CI caching keys.
- **Note from U0-05:** `pyproject.toml` has `[tool.uv.build-backend] module-root = ""` plus `source-exclude` and `wheel-exclude` entries for `dsbf/outputs/**`, `dsbf/dsbf.db` and the frontend directories (`uv_build` ignores `.gitignore`). Drop `module-root` (the `src/` default applies) and re-point the excludes. Rebuild with `uv build` and compare the wheel's file list with `git ls-files src`.
- **Scope:** In: the move, imports, packaging config, path fixes, tests. Out: the frontend (U1-02) and runtime paths (U1-04).
- **Acceptance:** All tests and golden runs pass unchanged. `uv build` produces a wheel that installs in a clean venv, and `dsbf version` works from a temp dir.
- **Commit:** `refactor: move package to src layout`

### U1-02 · Move the frontend to a top-level `frontend/`
- **Status:** todo · **Size:** S · **Depends on:** U1-01
- **Goal:** The dashboard source no longer sits inside the Python package.
- **Assumptions to confirm:** The paths that reference `dsbf/dashboard/frontend`: Dependabot directories, `.nvmrc`, docs, and `.gitignore`.
- **Note from U0-05:** the wheel and sdist still ship the frontend source (`dsbf/dashboard/frontend`), as they did under Poetry, with only `node_modules/` and `dist/` excluded in `pyproject.toml`. Once the frontend leaves the package, remove those two exclude entries and check with `uv build` that the wheel holds no frontend files.
- **Scope:** In: the move and path updates; `dsbf/dashboard/__init__.py` cleanup. Out: bundling the built frontend into the wheel (Phase 6).
- **Acceptance:** `npm ci && npm run build` works in `frontend/`. The dev proxy still reaches the API. Dependabot config points at the new path.
- **Commit:** `refactor(dashboard): move frontend to top-level directory`

### U1-03 · Frontend toolchain: pnpm, lint and unit tests
- **Status:** todo · **Size:** M · **Depends on:** U1-02 · **Resolves:** F-29 (tooling part)
- **Goal:** `frontend/` uses pnpm with `lint`, `build` and `test:run` scripts. ESLint (with `eslint-plugin-vue`) and Vitest are configured, and CI runs all three.
- **Context:** The frontend has only `dev`, `build` and `preview` scripts, plus a Vite 8 beta pin with `overrides` (F-29). The escaped search highlighter is a natural first unit test.
- **Assumptions to confirm:**
  - The pnpm version through the `packageManager` field and corepack.
  - A stable Vite release that can replace the beta pin.
  - **Ask the user:** include Prettier, or rely on ESLint stylistic rules?
- **Scope:** In: `pnpm-lock.yaml` (delete `package-lock.json`), scripts, ESLint and Vitest config with a few utility tests, a CI job, and the Dependabot ecosystem update. Out: refactoring components beyond lint autofixes. Remaining violations are suppressed per rule and recorded as follow-ups.
- **Acceptance:** `pnpm lint && pnpm build && pnpm test:run` pass locally and in CI.
- **Commit:** `build(dashboard): move frontend to pnpm with lint and unit tests`

### U1-04 · Runtime paths under `DSBF_HOME`
- **Status:** todo · **Size:** M · **Depends on:** U1-01 · **Resolves:** F-16 and F-17 for the legacy engine
- **Goal:** Outputs, the database, and run records go to one configurable home, so nothing depends on the CWD or writes into the package.
- **Context:**
  - `base_engine.py:93` and `:99` (outputs, `dsbf_run.json`);
  - `storage/schema.py:26` (DB inside the package);
  - `task_registry.py:276` (metadata export on every run);
  - API path resolution relative to the repo root.
- **Assumptions to confirm:**
  - The default home: `platformdirs.user_data_dir("dsbf")`, or `./.dsbf` per project. **Ask the user.**
  - Absolute paths are stored in the DB from now on. Old rows stay as they are.
- **Note from U0-06:** `run_column_correlations` and `read_run_sample` in `dsbf/api/main.py` call the db helpers without the module's `_DB_PATH` override, so they ignore it. Route every handler through one resolved path. `tests/test_api/test_endpoints.py` sets both `_DB_PATH` and `DSBF_DB_PATH` to work around it; drop the environment variable there once the handlers agree.
- **Note from U0-07:** the test session now forces `DSBF_DB_PATH` and runs from a temporary CWD, so the output and record locations you move here stay hermetic. Four tests in `tests/eda/test_engine/test_profile_engine.py` still set the dead `engine.output_path` key, and `test_metadata_structure_and_keys` rebuilds the default `dsbf/outputs/<timestamp>` path by hand. Use `engine.output_dir` there once the location moves.
- **Note from U0-16:** the dashboard gets a 404 for figure files (`/api/figures/<id>/file`) of a run profiled outside the repo root, because the database holds the relative `dsbf/outputs/...` path and the API resolves it against the repo root. Check the acceptance with a run started from a temporary directory.
- **Scope:** In: a `dsbf.paths` module; outputs, DB and record paths; metadata export becomes opt-in (`dsbf tasks export`); `record_run` removed or capped; API path resolution. Out: a new storage design (Phase 6).
- **Acceptance:** Running from any directory writes only under the home. The dashboard shows figures for a run started outside the repo. Tests pass.
- **Commit:** `feat: resolve runtime paths from DSBF_HOME`

### U1-05 · Declare optional-dependency extras
- **Status:** todo · **Size:** S · **Depends on:** U1-01
- **Goal:** `pyproject.toml` defines `ml`, `timeseries`, `text`, `pandas`, `sql`, `server` and `all` as in REFACTOR_PLAN §3.1. The base install doesn't change yet; legacy tasks still need everything.
- **Note from U0-04:** no DSBF code imports `pyarrow`; pandas uses it as its parquet engine (`pd.read_parquet` in `dsbf/api/db.py` and `dsbf/eda/profile_engine.py`) and polars doesn't need it. Put it in the extra that covers parquet input rather than the base install. The constraint is currently `>=20,<24` after the 23.0.1 bump.
- **Assumptions to confirm:** `[project.optional-dependencies]` resolves with `uv`, and `pip install dsbf[all]` from the built wheel resolves.
- **Acceptance:** `pip install "<wheel>[all]"` in a clean venv passes the test suite.
- **Note from U0-15:** ADR 0009 fixes the extras split (`ml` scikit-learn, `timeseries` statsmodels, `text` rapidfuzz, `pandas`, `sql` Ibis, `server` FastAPI, SQLAlchemy, Alembic). Heavy libraries are imported lazily, and each extra needs a test that the core runs without it.
- **Commit:** `build: declare optional dependency extras`

### U1-06 · Contracts I: subjects, severities, dimensions and evidence
- **Status:** todo · **Size:** M · **Depends on:** U1-01 · **Resolves:** F-10, F-36 (schema side)
- **Note from U0-14:** on the adversarial run `GET /api/runs/{run_key}/associations` and `/sample` return 500 (`Out of range float values are not JSON compliant: nan`), because all-NaN and `inf` columns reach the JSON encoder. `tests/perf/baseline.json` records the 500s; once the contracts or API stop emitting non-finite floats, update the baseline so the benchmark guards the 200s.
- **Note from U0-13:** the legacy `report.json` writes bare `NaN` (invalid JSON) in sample rows (`sample_head`, `sample_tail`); the golden normaliser spells non-finite floats as strings. The validators here must make that impossible in the new contracts.
- **Note from U0-09:** `data_quality_scorer` still splits leakage pair keys with `split("|", 1)` and mis-attributes a pair when the first column name contains `|` (U0-09 only stopped the crash). Pairs become structured records here; include a `|` pair case in the tests.
- **Note from U0-16:** `dsbf profile` on a parquet file with a `datetime64` column fails with `TypeError: Object of type Timestamp is not JSON serializable` in the legacy `report.json` writer (`dsbf/utils/report_utils.py:49`). It fails with pyarrow 23 as well. The contracts' JSON encoding must handle timestamps, with a test.
- **Goal:** `dsbf.model` provides `Subject`, `Severity`, `Confidence`, `Dimension`, `Evidence`, `Interval`, `AssumptionCheck`, `SamplingInfo` and `Timing`, as in REFACTOR_PLAN §4.2.
- **Assumptions to confirm:**
  - pydantic v2 is already a transitive dependency; make it direct.
  - The `Dimension` enum values (the five quality dimensions plus distribution, relationship and temporal).
- **Scope:** In: models, validators (finite floats only; NaN/inf → `None` plus a flag), unit tests including column names with `|`, brackets, unicode and empty strings. Out: Finding and Run (U1-07, U1-08).
- **Acceptance:** Round-trip tests pass. A non-finite float can't produce invalid JSON.
- **Commit:** `feat(model): add subject, evidence and severity contracts`

### U1-07 · Contracts II: findings, notes, check results and run health
- **Status:** todo · **Size:** M · **Depends on:** U1-06 · **Resolves:** F-20 (status model)
- **Goal:** `Finding` (with a stable id), `Guidance`, `Action`, `Note`, `CheckResult` (statuses `ok | not_applicable | skipped | unavailable | error | timeout`) and `RunHealth`.
- **Assumptions to confirm:**
  - The `Finding.id` hash inputs: check id, subject, code.
  - Mapping today's guidance blurbs onto `Guidance`/`Action` loses nothing. Compare with `BaseTask.add_guidance`.
- **Acceptance:**
  - Unit tests pass.
  - A test builds findings from a real legacy `TaskResult.guidance` sample without data loss.
  - `RunHealth` renders the plain-text summary shown in REFACTOR_PLAN §7.1.
- **Commit:** `feat(model): add finding, note, check result and run health contracts`

### U1-08 · Contracts III: profiles and the run manifest
- **Status:** todo · **Size:** M · **Depends on:** U1-07
- **Goal:** `ColumnProfile` (variants by semantic type), `DatasetProfile`, `DatasetManifest`, `PlanManifest` and `Run`, with `schema_version`.
- **Assumptions to confirm:** The profile fields cover everything listed in REFACTOR_PLAN §5.3 that the dashboard needs today. Check against current tab usage.
- **Acceptance:** A hand-built `Run` validates and round-trips through JSON. Its size for titanic-like content is under 200 KB without evidence artifacts.
- **Commit:** `feat(model): add profile and run manifest contracts`

### U1-09 · Configuration and parameter base models
- **Status:** todo · **Size:** M · **Depends on:** U1-06 · **Resolves:** F-22 (loader part)
- **Goal:** A `CheckParams` base class, and a layered config loader (defaults < project `dsbf.yaml` < env vars < CLI < keyword arguments) with deep merge and precise errors.
- **Assumptions to confirm:** The env var naming (`DSBF__SECTION__KEY`), and whether `pydantic-settings` is acceptable.
- **Acceptance:** Tests cover merge order, unknown keys (error naming the key and allowed keys), and type errors.
- **Commit:** `feat(config): add layered configuration and parameter base models`

### U1-10 · Policy rule schema
- **Status:** todo · **Size:** S · **Depends on:** U1-07
- **Goal:** Typed models for `policy.yaml` (`when` matchers, `then` actions `annotate | warn | fail | notify`) and `PolicyOutcome`. There's no evaluator yet (Phase 5).
- **Assumptions to confirm:** The REFACTOR_PLAN §7.2 example covers the matcher operators users need.
- **Acceptance:** The §7.2 example parses. Invalid rules produce precise errors.
- **Commit:** `feat(model): add policy rule schema`

### U1-11 · Schema export, breaking-change gate and TypeScript types
- **Status:** todo · **Size:** M · **Depends on:** U1-08, U1-09, U1-10, U1-03
- **Goal:** `dsbf schema export`. The JSON Schema is committed under `schema/`. CI fails on an unannounced breaking change. TS types are generated into `frontend/src/generated/`.
- **Assumptions to confirm:**
  - The tool choice for TS generation (for example `json-schema-to-typescript`).
  - How a "breaking change" is detected: removed or renamed fields and narrowed types.
- **Acceptance:**
  - Deleting a field in a test branch fails CI.
  - Regenerating the types is a no-op on a clean tree.
  - The Phase 1 exit check (a hand-built `Run` validates, round-trips and exports) passes in a test.
- **Commit:** `feat(model): export JSON Schema and TypeScript types with a breaking-change gate`

---

## Phase 2: Data and profile

- **Capability:** Tier-1 sources (in-memory data and files), a `Dataset` with fingerprint and sampling provenance, and a single profiling pass that replaces per-task rescans.
- **Motivation:** "Measure once, decide many times" (REFACTOR_PLAN §2). It removes the repeated per-column computation and the per-task pandas copies, and fixes type inference cost, listwise deletion and histogram honesty at the source.
- **Boundaries:** No check API and no changes to legacy tasks. The profile is new code, compared against legacy outputs.
- **Sequence and dependencies:** Depends on Phase 1 contracts. Sources, then dataset, then type inference, then column statistics, then dataset-level profile, then parity, then the performance budget.
- **Success criteria:** Profile parity with legacy outputs is documented, and the profile pass runs in under 2 s at 52k × 60 on CI hardware.

### U2-01 · Source adapters for in-memory data and files (tier 1)
- **Status:** todo · **Size:** M · **Depends on:** U1-08
- **Goal:** `dsbf.data.load(source)` accepts pandas, Polars or Arrow objects, or CSV/Parquet/Excel/JSON/NDJSON paths, and returns a Polars-backed `Dataset`. Unreadable input raises a typed error with a hint.
- **Note from U0-16:** upgrade to Polars 2.0 here (the `polars <2` cap; PR #46 was closed as deferred): this unit makes Polars the data layer, so read the 2.0 release notes first and run the adapters on it. Also add pandas 3.x as an input type: a test-matrix leg that loads pandas 3 frames (new string dtype, copy-on-write) through the adapter. That does not lift the `pandas <3` cap for the base install; the legacy tasks still need pandas 2.x until U4-E. Add a parquet source with a `datetime64` column to the table-driven test: `dsbf profile` on one fails today (see the U1-06 note).
- **Assumptions to confirm:** The Excel reader dependency (`fastexcel` vs `openpyxl`); whether pandas input converts zero-copy for common dtypes.
- **Acceptance:** A table-driven test covers each source type, including the adversarial frame and tz-aware datetimes.
- **Commit:** `feat(data): add tier-1 source adapters`

### U2-02 · Dataset fingerprint, sampling provenance and value cache
- **Status:** todo · **Size:** M · **Depends on:** U2-01 · **Resolves:** F-09
- **Goal:**
  - `Dataset.fingerprint`;
  - `Dataset.sample(...)` with a recorded `SamplingInfo` (head, seeded random, stratified-or-error);
  - `Dataset.values(col, dropna=True)`, which returns cached NumPy arrays within a memory bound.
- **Assumptions to confirm:** The hashing approach and its cost at 50k and 1M rows; the cache memory bound default.
- **Acceptance:** The fingerprint is stable across loads and changes when one cell changes. Sampling is reproducible by seed. Cache eviction is tested.
- **Commit:** `feat(data): add dataset fingerprint, sampling provenance and value cache`

### U2-03 · Sample-based semantic type inference
- **Status:** todo · **Size:** M · **Depends on:** U2-02 · **Resolves:** F-37
- **Goal:** Semantic types are inferred from a sample, with confirmation on the full column only where needed. The results agree with `infer_types` on the golden datasets, or the differences are documented.
- **Context:** F-37. `_is_datetime_like` took 8.6 s of `infer_types`' 9.0 s on kitchen5k and grew 10× at 50k rows.
- **Acceptance:** Parity is checked against U0-13 baselines, and the differences are listed in the PR. Inference time is under 0.5 s on kitchen50k.
- **Commit:** `feat(profile): add sample-based semantic type inference`

### U2-04 · Column profile I: counts and numeric statistics
- **Status:** todo · **Size:** M · **Depends on:** U2-03 · **Resolves:** F-32 (per-column statistics)
- **Note from U0-09:** `detect_collinear_features` now only guards on complete-row count (infinities treated as missing). Per-column handling, dropping high-missingness columns first and recording `n_used` stay with the VIF redesign in Phase 4.
- **Goal:** One pass computes counts, nulls, distinct (exact or sketched), zeros, empty strings, non-finite values, and numeric min, max, mean, std, skew, kurtosis, MAD and quantiles, per column on that column's non-null values.
- **Acceptance:** Values match `summarize_numeric`, `summarize_nulls` and `summarize_unique` on the baselines within tolerance.
- **Commit:** `feat(profile): add count and numeric column statistics`

### U2-05 · Column profile II: histograms, top-k, text and datetime
- **Status:** todo · **Size:** M · **Depends on:** U2-04 · **Resolves:** F-12
- **Note from U0-11:** the percentile-binned histogram parked in `d4d949f` is already on `main` (PR #49 merged the ECharts components), so the F-12 defect is live in the dashboard until this unit lands. `origin/wip/echarts-charts` no longer exists; the local `wip/echarts-charts` branch can be deleted.
- **Goal:**
  - equal-width histogram bins, with the rule recorded;
  - top-k values with counts;
  - string length and pattern statistics;
  - datetime range, inferred frequency, gaps and timezone.
- **Assumptions to confirm:** The default binning rule (Freedman–Diaconis with a Sturges fallback) and the top-k default.
- **Acceptance:** Histogram counts sum to `n_non_null`. The adversarial frame profiles without error. Values match `summarize_value_counts` and `categorical_length_stats`.
- **Commit:** `feat(profile): add histograms, top-k, text and datetime profiles`

### U2-06 · Dataset profile and stage inference
- **Status:** todo · **Size:** S · **Depends on:** U2-05 · **Resolves:** F-08
- **Goal:** Shape, a hash-based duplicate-row count, a missingness co-occurrence summary, and stage inference from the profile with one vocabulary shared with check metadata.
- **Acceptance:** Stage inference gives the same answer for pandas and Polars input of the same data (the F-08 regression test).
- **Commit:** `feat(profile): add dataset-level profile and stage inference`

### U2-07 · Profile parity report against legacy outputs
- **Status:** todo · **Size:** M · **Depends on:** U2-06
- **Goal:** An automated comparison between the new profile and the legacy `summarize_*` and `infer_types` outputs on every golden dataset, plus a committed `docs/parity/profile.md` listing accepted differences with reasons.
- **Acceptance:** The parity test passes with only the listed differences.
- **Commit:** `test(profile): add legacy parity comparison and accepted differences`

### U2-08 · Profile performance budget
- **Status:** todo · **Size:** S · **Depends on:** U2-07, U0-14 · **Resolves:** F-30 (profiling part)
- **Note from U0-14:** `tests/perf/scenarios.py` has no 1M x 50 scenario, because today's engine would take hours on it. Add it here if the new profile pass can run it; the REFACTOR_PLAN §11 budget names it. The baseline shows where the legacy time goes: `fuzzy_duplicate_detection` (65% of titanic, 46% of kitchen5k) and `detect_stationarity` (40% of kitchen50k).
- **Goal:** Add the profile pass to the perf harness, with the budget enforced.
- **Acceptance:** Profile pass under 2 s at kitchen50k (52.5k × 60) on CI hardware, and under 2.5× peak memory over the in-memory data size. Or record the actual numbers and amend the budget with reasons.
- **Commit:** `test(perf): enforce profiling pass budget`

---

## Phase 3: Engine and check API

- **Capability:** The check API, registry and packs, the planner, an isolated, parallel and cached executor, library-friendly logging, a legacy adapter running every existing task, and the public `dsbf.profile()` entry point.
- **Motivation:** "Choose your own analysis" and report-only failure handling (REFACTOR_PLAN §1, R1 and R2) live in the engine. The adapter lets checks migrate one at a time against a running system.
- **Boundaries:** No check migrations (Phase 4), and no storage, API or dashboard changes (Phase 6).
- **Sequence and dependencies:** Depends on Phase 2. Interface, registry, planner, executor (isolation, then timeouts and events, then parallelism), cache, logging, adapter, entry point.
- **Success criteria:** Golden runs reproduce through the adapter. A forced exception in any check still yields a complete `Run` with a note. Changing one parameter re-executes only the affected checks.

### U3-01 · Check interface and check context
- **Status:** todo · **Size:** M · **Depends on:** U2-08, U1-09
- **Goal:** The `@check` decorator and metadata (id, version, kind, params, `applies_to`, `requires`, `after`, cost, extras, packs, dimensions). `CheckContext` provides filtered `columns`, `values()`, `profile`, `finding()`, `note()` and `output()`.
- **Assumptions to confirm:** The decorator vs. class-based API (REFACTOR_PLAN §6.2 shows the decorator); naming per ADR D10.
- **Acceptance:** Three toy checks (rule, analysis, synthesis) run in a test harness and produce valid `CheckResult`s. A check never sees columns outside `applies_to`.
- **Commit:** `feat(engine): add check interface and check context`

### U3-02 · Registry, packs and plugin discovery
- **Status:** todo · **Size:** M · **Depends on:** U3-01 · **Resolves:** F-23
- **Goal:** An explicit `Registry` object (no import side effects), the `dsbf.checks` and `dsbf.packs` entry-point groups, local plugin directories imported under unique module names, and reload safety.
- **Acceptance:**
  - A test plugin package installed in a venv is discovered.
  - Two plugin dirs that each contain a `utils.py` coexist.
  - Loading twice in one process is a no-op.
- **Commit:** `feat(engine): add registry, packs and entry-point plugin discovery`

### U3-03 · Planner I: selection and dependency resolution
- **Status:** todo · **Size:** M · **Depends on:** U3-02 · **Resolves:** F-21
- **Goal:** Select checks by packs, include/exclude lists, depth presets, cost budget, dimension and tag. Resolve hard requirements (auto-included and marked) and soft `after` ordering. Report cycles as plan errors.
- **Acceptance:** Table-driven tests cover selection. A soft-after dependency that errors doesn't skip its dependent (the F-21 scenario).
- **Commit:** `feat(engine): add plan selection and dependency resolution`

### U3-04 · Planner II: validation, expected statuses and `dsbf plan`
- **Status:** todo · **Size:** M · **Depends on:** U3-03 · **Resolves:** F-07, F-22
- **Goal:**
  - Validate parameters against each check's model, with precise errors.
  - Annotate the plan with `will_run`, `not_applicable` or `unavailable` (missing extra), with reasons.
  - Print a readable plan via a new `dsbf plan` CLI command.
- **Acceptance:**
  - A misspelled parameter is an error naming the check, key and allowed keys (the F-07 class).
  - A missing extra shows as `unavailable` with an install hint.
  - `dsbf plan data.csv` output has a snapshot test.
- **Commit:** `feat(engine): validate plans and add dsbf plan`

### U3-05 · Executor I: isolation, statuses and notes
- **Status:** todo · **Size:** M · **Depends on:** U3-04 · **Resolves:** F-06, F-20 (engine side)
- **Goal:** Sequential execution where every exception becomes `error` with a note (type, message, hint, traceback artifact), hard dependents become `skipped` with the blocker named, and the run always completes with `RunHealth`.
- **Acceptance:** A forced exception in each toy check yields a complete `Run`. The health summary lists the reason. A check returning a failed result is never counted as ok (the F-06 regression test).
- **Commit:** `feat(engine): execute checks in isolation with explicit statuses`

### U3-06 · Executor II: timeouts, resource accounting and events
- **Status:** todo · **Size:** M · **Depends on:** U3-05 · **Resolves:** F-24
- **Goal:** Per-check timeouts (→ `timeout` plus a note); an optional run budget; wall, CPU and sampled peak memory per check, labelled approximate; and an event hook API (`check_started`, `check_finished`, `note`).
- **Assumptions to confirm:** How timeouts work for thread-based checks (cooperative cancellation vs. process isolation for heavy checks).
- **Acceptance:** A sleeping toy check times out without stalling the run. Events arrive in order. Memory figures are within reason of `/usr/bin/time` on a perf dataset.
- **Commit:** `feat(engine): add timeouts, resource accounting and execution events`

### U3-07 · Executor III: parallel execution
- **Status:** todo · **Size:** M · **Depends on:** U3-06
- **Goal:** Independent checks at each DAG level run concurrently (threads for compiled-code checks, a process pool for pure-Python ones). Results are deterministic regardless of worker count.
- **Acceptance:** The same `Run` (excluding timing) with 1 worker and N workers. A speed-up is measured on a perf dataset.
- **Commit:** `feat(engine): run independent checks in parallel`

### U3-08 · Result cache
- **Status:** todo · **Size:** M · **Depends on:** U3-07
- **Goal:** Results cached by `(fingerprint, check id, check version, params hash, sampling)` on local disk under `DSBF_HOME`, with invalidation of dependents.
- **Acceptance:** Changing one parameter re-executes only that check and its dependents (a test asserts the execution counts). Bumping a check version invalidates its entries.
- **Commit:** `feat(engine): cache check results by data fingerprint and parameters`

### U3-09 · Library-friendly logging
- **Status:** todo · **Size:** S · **Depends on:** U3-05 · **Resolves:** F-19
- **Goal:**
  - a `dsbf` logger hierarchy with no global logging changes;
  - user data never interpreted as markup;
  - an `error` level;
  - the CLI attaches handlers;
  - per-run file logs, with handlers closed afterwards.
- **Acceptance:** Importing `dsbf` leaves `logging.getLoggerClass()` unchanged. Logging a column named `[/x]` produces no logging error. Task-level messages appear in the run log file.
- **Commit:** `refactor(logging): make logging library-friendly and markup-safe`

### U3-10 · Legacy task adapter
- **Status:** todo · **Size:** M · **Depends on:** U3-08, U3-09
- **Goal:** `LegacyTaskCheck` wraps any `BaseTask`. It provides a legacy `AnalysisContext` shim (data, config, `semantic_types` from the new profile) and maps `TaskResult` statuses onto the new statuses.
- **Assumptions to confirm:** Which `context.metadata` keys legacy tasks read from each other (grep `get_metadata`, `context.results`).
- **Acceptance:** Five representative legacy tasks (a rule-like task, a pandas-heavy one, a plotting one, a synthesis scorer, and a time-series one) run through the new engine with their outputs unchanged.
- **Commit:** `feat(engine): add legacy task adapter`

### U3-11 · All legacy tasks through the new engine with findings
- **Status:** todo · **Size:** M · **Depends on:** U3-10
- **Goal:** Every registered legacy task runs on the new engine. Guidance blurbs convert into `Finding`s with evidence where it can be derived.
- **Acceptance:** Golden runs (U0-13) reproduce through the adapter, with differences listed in the PR. The adversarial frame produces a complete `Run` with no unhandled errors.
- **Commit:** `feat(engine): run all legacy tasks through the new engine`

### U3-12 · `dsbf.profile()` entry point and run health output
- **Status:** todo · **Size:** M · **Depends on:** U3-11 · **Resolves:** F-16, F-17, F-18, F-34
- **Goal:** The public `dsbf.profile(source, packs=…, checks=…, target=…, time_index=…, sample=…)` returns an in-memory `Run` with no filesystem writes. Version comes from `importlib.metadata`. `run.health.summary()` matches REFACTOR_PLAN §7.1.
- **Acceptance:**
  - A test asserts no files are created during `dsbf.profile()`.
  - The README library example works.
  - Phase 3 exit checks pass: golden via the adapter, forced exception gives a complete `Run`, a parameter change re-executes only the affected checks.
- **Commit:** `feat: add dsbf.profile() library entry point`

---

## Phase 4: Migrate checks (placeholder)

- **U4-00 · Plan Phase 4** (`todo`, depends on U3-12): break down the batches below into full units using Phase 3 learnings and golden-run evidence. Update REFACTOR_PLAN if needed. Carry in from U0-15 (ADR 0009): shrink the base install to the core dependency set as the legacy tasks migrate, moving the rest behind extras and deleting dependencies nothing imports (for example matplotlib, plotly, seaborn, networkx, graphviz, pydot and missingno, to be verified by grep).
- U4-A · Rule checks onto the profile, about 20 tasks (`placeholder`). From U0-03: the migrated guards only check that some column matches; tasks still loop over every numeric or object dtype column, not `matched_cols`, so they also analyse columns whose semantic type was excluded. And a migrated task run without `infer_types` now skips instead of analysing all numeric columns. Decide the intended behaviour per check. From U0-08: the time-series test configs still carry an inert nested `time_series.tasks.<name>` block that no task reads (tasks read `tasks.<task name>`); drop it when those tests migrate. From U0-07: `get_recommendation_tip` (`dsbf/utils/reco_engine.py`) still runs `eval` on the YAML `condition` strings (F-31); replace it with a declarative condition format (`{"field": ..., "op": ">", "value": ...}`) when the 14 tasks that call it migrate
- U4-B · Analysis checks with statistical fixes: F-11, F-13, confidence intervals, bias-corrected Cramér's V (`placeholder`). From U0-09: `summarize_modes` iterates all columns rather than `matched_cols`, and the leakage correlation scan uses `select_dtypes` (F-11); the collinear check needs per-column VIF handling and `n_used` (F-32). From U0-16: when the VIF check is redesigned, lift the `statsmodels <0.15` cap and decide the intended VIF definition (centred, with an intercept, as 0.15 computes it, or the uncentred one the baselines record); when bimodality is migrated, lift `scikit-learn <1.8` and decide how the BIC comparison is reported (the absolute BIC and `delta / abs(bic1)` both depend on the log-likelihood convention; the delta itself did not move). Regenerate those golden snapshots in the same PR, with the differences explained
- U4-C · Heavy checks redesigned with budgets: fuzzy duplicates (F-35), time series (F-39), drift against reference profiles (`placeholder`). From U0-13: `fuzzy_duplicate_detection` builds candidate pairs from a token index over sets of strings and stops at `max_candidates`, so which pairs it keeps (and the count) follows `PYTHONHASHSEED`; make the order deterministic, then the golden runner no longer needs the pinned seed. On kitchen5k, `decompose_time_series` cannot infer a seasonal period (frequency `None` for a daily index with repeated dates), and the golden baseline records that.
- U4-D · Synthesis: quality dimensions, ML readiness, target-based leakage (F-26) (`placeholder`)
- U4-E · Remove the legacy adapter, backend plotting and plotting dependencies (N-A2, F-38) (`placeholder`). From U0-16: once no legacy task needs pandas, lift the `pandas <3` cap (PR #47 was closed as deferred; the golden suite and the pandas extra's tests must pass on 3.x)

## Phase 5: Policy, sinks, CLI v2. First release for embedded use (placeholder)

- **U5-00 · Plan Phase 5** (`todo`, depends on the Phase 4 units). Carry in from U0-13: `tests/validation/run_validation.py` without `--report` still takes the newest `report.json` under `dsbf/outputs/`, whatever its dataset (it checked `near_clean` against a stale 9,217-row run on 2026-10-16), and the README keeps a warning about it; make it profile into its own output directory, as the golden runner does, then drop the warning. Its `clean` checklist also says 10 columns where the dataset has 9. Carry in from U0-11: (1) `dsbf profile` defaults `--depth` to `standard`, while `default_config.yaml`, `dsbf quickstart` and `dsbf run` use `full`: pick one default for CLI v2. (2) The default config's `message_verbosity` comment lists `quiet | info | debug`, but the logger also accepts `warn` and `stage`. (3) Dead config keys `metadata.output_format`, `engine.severity_thresholds` and `metadata.layout_name` (F-28) go with the config redesign.
- Placeholders: policy evaluator (report-only default) · JSON/NDJSON sinks · Parquet sinks with a consumer schema guide · static HTML report · CLI v2 (`profile`, `check`, `plan`, `schema export`, run health summary) · pipeline and consumer docs · install smoke test from a temp dir · release `1.0.0a1`

## Phase 6: Store, service and dashboard v2. Standalone local (placeholder)

- **U6-00 · Plan Phase 6** (`todo`, depends on the Phase 5 units)
- Placeholders: run repository on SQLite with Alembic · artifact store and per-run data snapshot · `dsbf runs …` · API v2 · local job runner · generated TS client · dashboard shell and run health · findings explorer · column view (reuse `wip/echarts-charts` components with real histogram bins) (U0-16 saw the light theme draw chart titles and card headings such as "Value Counts" almost white on a light background; same on echarts 5) · relationships · quality · ML readiness · time series · run compare and history · settings → rerun · frontend component and end-to-end tests (F-29) · bundle the dashboard into the wheel with `dsbf dashboard` · remove API v1 and the old store (F-15, F-25) · release `1.0.0`

## Phase 7: Scale and sources (placeholder)

- **U7-00 · Plan Phase 7**. Carry in from U0-15: re-check ADR 0004, that the profile interface can be expressed as an Ibis aggregation (with the approximate-statistics caveats), before committing to the pushdown design; amend or supersede the ADR if it can't.
- Placeholders: larger-than-memory files and object storage (T2) · Ibis sources with DuckDB/Postgres, then Snowflake/BigQuery (T3) · recurring datasets, history and profile-based drift (T4) · scheduling recipes

## Phase 8: Team and cloud (placeholder)

- **U8-00 · Plan Phase 8**
- Placeholders: Postgres repository · object-storage artifacts · auth (token, OIDC) · job queue · Docker image and Helm chart · GitHub Action · Airflow, Prefect and Dagster adapters · dbt recipe · notifiers and OpenLineage · MCP server

## Phase 9: Modalities and workflow phases (placeholder)

- **U9-00 · Plan Phase 9**
- Placeholders: text-corpus modality · domain packs · modelling-prep exports (ML guidance actions as sklearn pipelines) · modelling diagnostics
