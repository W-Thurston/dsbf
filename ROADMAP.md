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
| `scipy <1.18` (found in U1-05, no PR) | 1.18 makes `kstest(x, "norm", args=(mu, sigma))` raise `TypeError: ndtr() takes from 1 to 2 positional arguments but 3 were given`, so `run_ks` and two normality tests fail on a fresh `pip install`; DSBF imported scipy without declaring it, so only the lock kept it on 1.16.3 | the unit that migrates the normality tests (Phase 4; U4-00 assigns it), or sooner if a unit touches `normality_tests.py` |
| `plotly <7` (PR #62, closed 2026-10-16) | 7 removes the `mapbox` layout property that `custom_plotly_templates.py` sets, so it fails at import; the backend plotting that uses plotly is deleted anyway | U4-E (plotly is deleted; drop the cap and the ignore entry) |

`.github/dependabot.yml` carries an `ignore` entry for each row, so Dependabot doesn't re-raise the held-back version. Removing a cap removes its entry.

---

## Phase 0: Stabilise and baseline

**Done (2026-10-16), 17 units.** Detailed unit entries were condensed here at close-out; the per-unit notes that later units depend on were carried into those units as "Note from U0-xx" lines.

- **Capability delivered:** a trustworthy, measured baseline on the target toolchain. CI runs lint (ruff from `uv.lock`), type check (pyrefly with a baseline), the fast gate and the full suite on Python 3.12 and 3.13, CodeQL, and the `golden` and `perf` workflows (nightly, on demand and on labelled PRs). Dependencies are managed with uv. Tests run hermetically from a temporary directory, every registered task completes on a hostile frame, and ADRs D1–D11 are accepted.
- **Success criteria, measured:**
  - Every Phase 0 unit is `done`: 17 of 17.
  - Fast gate (`unit and not slow`) 29 s; full suite 5 min 14 s (858 tests, U0-06). The `titanic` quickstart from a temporary directory went from 4 failed tasks to 0 (U0-07); `kitchen5k` from 3 failed tasks to 0, with 5 skipped by design (U0-09).
  - Golden baselines: ten snapshots (the 8 archetypes, `titanic`, `kitchen5k`), 60 successful and 5 skipped tasks each, 0 errors; two full runs matched byte for byte (U0-13). Re-checked on 2026-10-16 against the upgraded dependency set: 10 of 10 pass.
  - Performance baseline, taken on `ubuntu-latest` (4 CPUs): `titanic` 26 s, adversarial frame 16 s, `kitchen5k` 147 s, `kitchen50k` 357 s with a 2.08 GB peak, 0 failed tasks (U0-14).
  - pyrefly baseline: 895 existing errors across `dsbf/`, `tests/` and `docs/practice_run/` (U0-12); 0 new.
  - ADRs D1–D11 are `Accepted`: [0001](docs/adr/0001-library-first-layering.md), [0002](docs/adr/0002-findings-as-primary-output.md), [0003](docs/adr/0003-polars-arrow-data-representation.md), [0004](docs/adr/0004-ibis-warehouse-pushdown.md), [0005](docs/adr/0005-pydantic-contracts.md), [0006](docs/adr/0006-report-only-failure-semantics.md), [0007](docs/adr/0007-entry-point-plugins.md), [0008](docs/adr/0008-sqlalchemy-alembic-fsspec-storage.md), [0009](docs/adr/0009-single-distribution-with-extras.md), [0010](docs/adr/0010-check-naming.md), [0011](docs/adr/0011-toolchain.md).
  - Security alerts: 0 open code-scanning alerts and 0 open Dependabot alerts, re-listed on 2026-10-16 after #59 merged (alerts 3–10 closed when CodeQL re-scanned `main`).
- **Key decisions:** uv with the `uv_build` backend and `uv.lock` (ADR 0011); pyrefly as a pinned dev dependency, `default` preset, committed baseline that only shrinks; Python 3.14 out of the CI matrix (nine locked packages had no cp314 wheel); golden runs pin `PYTHONHASHSEED=0` and rank value counts with a stable sort (`ranked_value_counts`); `titanic` is vendored for the golden suite (seaborn-data has no license); perf numbers are compared only against the CI-hardware baseline; U0-16 closed the code-scanning alerts by returning generic API error text and logging the exception, and by giving the workflows `permissions: contents: read`.
- **Deviations from the phase block:**
  - The phase promised "no new features" and cheap safety fixes only. U0-09 also fixed `RichHandler(markup=True)`, which made seven tasks crash on column names such as `[/x]` (the largest crash cause); U0-13 fixed unstable tie ordering in three tasks; U0-06 added API tests.
  - "Dependabot backlog merged" took three units: U0-04, U0-16 and U0-17. U0-16 was widened on 2026-10-16 from the security alerts to the whole backlog, so Phase 1 starts with no open PR. It took #34, #38, #42, #43 and #45, and closed #46 (polars 2) and #47 (pandas 3) as deferred. Dependabot then raised four more PRs as soon as #59 merged, so U0-17 took #60 (pandas 2.3.3, polars 1.44.2; its pyrefly baseline lost one entry the new typings fixed), #61 (ipykernel 7) and #63 (rich 15), all on green CI with golden 10 of 10, and closed #62 (plotly 7, which fails at import and is deleted with the legacy plotting in U4-E).
  - The golden suite showed that two upgrades in #43 change results, so they stay behind caps: `statsmodels <0.15` (VIF includes the intercept) and `scikit-learn <1.8` (`GaussianMixture` log-likelihood on discrete columns). With `pandas <3`, `polars <2` and `plotly <7` they are listed under "Dependency upgrades" at the top of this file, each with the unit that revisits it, and mirrored by `ignore` entries in `.github/dependabot.yml`.
  - The README's examples were all rebuilt (U0-11) because most no longer ran.
- **PRs:** #29, #31, #32, #40, #41, #48, #50, #51, #52, #53, #54, #55, #56, #57, #58, #59 (U0-16, merged as `751ad9d`), and #64 (U0-17, merged as `de863ab`). Dependency PRs merged along the way: #23, #26, #24, #28, #27, #20, #22 and #36 (U0-04); #38, #42, #45, #34 and #43 (U0-16); #61, #63 and #60 (U0-17).

---

## Phase 1: Contracts

- **Capability:** The `src/` layout, a top-level frontend on pnpm, runtime paths under `DSBF_HOME`, optional extras, and the versioned result contracts (`dsbf.model`) with JSON Schema and TypeScript export.
- **Motivation:** All three usage modes depend on stable, published contracts (REFACTOR_PLAN §4). Embedded users consume the schema without the rest of DSBF.
- **Boundaries:** Packaging, paths and models only. No profiling or engine code. The legacy engine keeps working throughout.
- **Sequence and dependencies:** Depends on Phase 0. Layout and toolchain first, then contracts from the leaves (evidence) to the root (`Run`), then export.
- **Success criteria:** A hand-built `Run` validates, round-trips, and exports as JSON Schema and TypeScript, and the breaking-change gate is active in CI.

### U1-01 · Move the Python package to a `src/` layout
- **Status:** done (2026-10-16) · **Size:** M · **Depends on:** U0-15 · **Resolves:** groundwork (REFACTOR_PLAN §3.1)
- **Goal:** `dsbf/` lives at `src/dsbf/` with no behaviour change.
- **Assumptions to confirm:**
  - Code that relies on the repo layout: `Path(__file__).parents[2]` in `storage/schema.py`, `api/main.py` and `api/db.py`, plus test paths.
  - The build backend's package discovery config.
  - CI caching keys.
- **Note from U0-05:** `pyproject.toml` has `[tool.uv.build-backend] module-root = ""` plus `source-exclude` and `wheel-exclude` entries for `dsbf/outputs/**`, `dsbf/dsbf.db` and the frontend directories (`uv_build` ignores `.gitignore`). Drop `module-root` (the `src/` default applies) and re-point the excludes. Rebuild with `uv build` and compare the wheel's file list with `git ls-files src`.
- **Scope:** In: the move, imports, packaging config, path fixes, tests. Out: the frontend (U1-02) and runtime paths (U1-04).
- **Acceptance:** All tests and golden runs pass unchanged. `uv build` produces a wheel that installs in a clean venv, and `dsbf version` works from a temp dir.
- **Commit:** `refactor: move package to src layout`
- **Done notes:** the package, its packaged data and the frontend source now sit under `src/dsbf/`. The default database (`<repo>/dsbf/dsbf.db`), `dsbf/outputs/` and the stored relative figure paths are unchanged, so existing runs keep working; U1-04 moves them. Widened beyond the listed assumptions: the metadata export path, two tests, `migrate_runs.py`, and `[tool.pyrefly] search-path = ["src", "."]` with a regenerated baseline (888 entries before and after). The wheel's file list equals `git ls-files src`. Merged as PR #66 (`bfc43e1`).

### U1-02 · Move the frontend to a top-level `frontend/`
- **Status:** done (2026-10-16) · **Size:** S · **Depends on:** U1-01
- **Goal:** The dashboard source no longer sits inside the Python package.
- **Assumptions to confirm:** The paths that reference `dsbf/dashboard/frontend`: Dependabot directories, `.nvmrc`, docs, and `.gitignore`.
- **Note from U1-01:** the frontend now sits at `src/dsbf/dashboard/frontend`, and `pyproject.toml` excludes `node_modules` and `dist` there (`source-exclude` relative to the project root, `wheel-exclude` relative to `src/dsbf`). The root `.gitignore` entries for `dist/` and `.vite/` also name that path.
- **Note from U0-05:** the wheel and sdist still ship the frontend source (`dsbf/dashboard/frontend`), as they did under Poetry, with only `node_modules/` and `dist/` excluded in `pyproject.toml`. Once the frontend leaves the package, remove those two exclude entries and check with `uv build` that the wheel holds no frontend files.
- **Scope:** In: the move and path updates; `dsbf/dashboard/__init__.py` cleanup. Out: bundling the built frontend into the wheel (Phase 6).
- **Acceptance:** `npm ci && npm run build` works in `frontend/`. The dev proxy still reaches the API. Dependabot config points at the new path.
- **Commit:** `refactor(dashboard): move frontend to top-level directory`
- **Done notes:** the frontend sits at `frontend/` (renames, history kept). `src/dsbf/dashboard/` is deleted: it held an empty `__init__.py` and a stray empty `package-lock.json`, and nothing imported it. The `source-exclude` and `wheel-exclude` keys are gone from `pyproject.toml`; the wheel still equals `git ls-files src` and the sdist holds no frontend files. Docs, `.gitignore` and 18 source header comments name the new path. Deviation: `.github/dependabot.yml` never had an npm entry, so there was no path to repoint; U1-03 adds the entry. The `.nvmrc` stays at the root. Merged as PR #67 (`b99633a`).

### U1-03 · Frontend toolchain: pnpm, lint and unit tests
- **Status:** done (2026-10-16) · **Size:** M · **Depends on:** U1-02 · **Resolves:** F-29 (tooling part)
- **Goal:** `frontend/` uses pnpm with `lint`, `build` and `test:run` scripts. ESLint (with `eslint-plugin-vue`) and Vitest are configured, and CI runs all three.
- **Context:** The frontend has only `dev`, `build` and `preview` scripts, plus a Vite 8 beta pin with `overrides` (F-29). The escaped search highlighter is a natural first unit test.
- **Assumptions to confirm:**
  - The pnpm version through the `packageManager` field and corepack.
  - A stable Vite release that can replace the beta pin.
  - **Ask the user:** include Prettier, or rely on ESLint stylistic rules?
- **Scope:** In: `pnpm-lock.yaml` (delete `package-lock.json`), scripts, ESLint and Vitest config with a few utility tests, a CI job, and a Dependabot `npm` entry for `/frontend` (the config has none today; U1-02 found nothing to repoint). Out: refactoring components beyond lint autofixes. Remaining violations are suppressed per rule and recorded as follow-ups.
- **Acceptance:** `pnpm lint && pnpm build && pnpm test:run` pass locally and in CI.
- **Commit:** `build(dashboard): move frontend to pnpm with lint and unit tests`
- **Done notes:** `frontend/` uses pnpm 12.14.0 (`packageManager`, `pnpm-lock.yaml`) and stable Vite 8.3.4; ESLint 10 with `eslint-plugin-vue`, Prettier (the user's choice over ESLint stylistic rules) and Vitest 5 are configured, with 31 unit tests for `utils.js` and `actionMeta.js`. `pnpm lint` runs ESLint and then the Prettier check. A `Frontend` CI job runs lint, build and tests, and Dependabot has an `npm` entry for `/frontend`. The Prettier pass reflowed 49 files; compiled templates match `HEAD` apart from formatting, attribute order and `:colA` → `:col-a`. Deviations: 20 existing findings (16 unused variables, 1 side effect in a computed property in `TimeSeriesTab.vue`, 3 props without defaults) are warnings, not errors, and `vue/no-v-html` is off because `highlightHtml` escapes its input. The dashboard was not viewed in a browser. Merged as PR #68 (`c82fe45`).
- **Note for U1-11:** the generated TypeScript in `frontend/src/generated/` must be listed in `frontend/.prettierignore` and in the `ignores` of `frontend/eslint.config.js`, or `pnpm lint` fails on it.

### U1-04 · Runtime paths under `DSBF_HOME`
- **Status:** done (2026-10-17) · **Size:** M · **Depends on:** U1-01 · **Resolves:** F-16 and F-17 for the legacy engine
- **Goal:** Outputs, the database, and run records go to one configurable home, so nothing depends on the CWD or writes into the package.
- **Context:**
  - `base_engine.py:93` and `:99` (outputs, `dsbf_run.json`);
  - `storage/schema.py:26` (DB inside the package);
  - `task_registry.py:276` (metadata export on every run);
  - API path resolution relative to the repo root.
- **Assumptions to confirm:**
  - The default home: `platformdirs.user_data_dir("dsbf")`, or `./.dsbf` per project. **Ask the user.**
  - Absolute paths are stored in the DB from now on. Old rows stay as they are.
- **Note from U1-01:** the source paths in the notes below now sit under `src/dsbf/`. The three repo-root lookups (`storage/schema.py`, `api/main.py`, `api/db.py`) use `Path(__file__).resolve().parents[3]`, which is wrong for an installed wheel, as it was before. The metadata export writes the CWD-relative `src/dsbf/static_metadata/task_metadata.json`. `tests/test_hermetic_runtime.py` and `tests/test_utils/test_reco_engine.py` assert that CWD-relative `src/dsbf/...` paths don't exist.
- **Note from U0-06:** `run_column_correlations` and `read_run_sample` in `dsbf/api/main.py` call the db helpers without the module's `_DB_PATH` override, so they ignore it. Route every handler through one resolved path. `tests/test_api/test_endpoints.py` sets both `_DB_PATH` and `DSBF_DB_PATH` to work around it; drop the environment variable there once the handlers agree.
- **Note from U0-07:** the test session now forces `DSBF_DB_PATH` and runs from a temporary CWD, so the output and record locations you move here stay hermetic. Four tests in `tests/eda/test_engine/test_profile_engine.py` still set the dead `engine.output_path` key, and `test_metadata_structure_and_keys` rebuilds the default `dsbf/outputs/<timestamp>` path by hand. Use `engine.output_dir` there once the location moves.
- **Note from U0-16:** the dashboard gets a 404 for figure files (`/api/figures/<id>/file`) of a run profiled outside the repo root, because the database holds the relative `dsbf/outputs/...` path and the API resolves it against the repo root. Check the acceptance with a run started from a temporary directory.
- **Scope:** In: a `dsbf.paths` module; outputs, DB and record paths; metadata export becomes opt-in (`dsbf tasks export`); `record_run` removed or capped; API path resolution. Out: a new storage design (Phase 6).
- **Acceptance:** Running from any directory writes only under the home. The dashboard shows figures for a run started outside the repo. Tests pass.
- **Commit:** `feat: resolve runtime paths from DSBF_HOME`
- **Done notes:** `dsbf.paths` resolves the home (`DSBF_HOME`, else the per-user data directory via `platformdirs`, chosen by the user on 2026-10-17 so one build serves a laptop and a cloud job that mounts a volume). Outputs, the default database (`<home>/dsbf.db`; `DSBF_DB_PATH` still overrides it), figure paths and dataset paths are absolute; every API handler uses one database path; `record_run` and `dsbf_run.json` are gone; `dsbf tasks export` replaces the automatic metadata export. Verified on a real run from a scratch directory: 357 of 357 figures served by the API from another directory. Golden 10 of 10 unchanged. Deviations: `platformdirs` became a direct dependency, and the old `.gitignore`, pre-commit and ruff entries stay to hide local leftovers. Object-storage or server-database homes remain Phase 6 (ADR 0008). Merged as PR #71 (`77d1992`).

### U1-05 · Declare optional-dependency extras
- **Status:** done (2026-10-17) · **Size:** S · **Depends on:** U1-01
- **Goal:** `pyproject.toml` defines `ml`, `timeseries`, `text`, `pandas`, `sql`, `server` and `all` as in REFACTOR_PLAN §3.1. The base install doesn't change yet; legacy tasks still need everything.
- **Note from U0-04:** no DSBF code imports `pyarrow`; pandas uses it as its parquet engine (`pd.read_parquet` in `dsbf/api/db.py` and `dsbf/eda/profile_engine.py`) and polars doesn't need it. Put it in the extra that covers parquet input rather than the base install. The constraint is currently `>=20,<24` after the 23.0.1 bump.
- **Assumptions to confirm:** `[project.optional-dependencies]` resolves with `uv`, and `pip install dsbf[all]` from the built wheel resolves.
- **Acceptance:** `pip install "<wheel>[all]"` in a clean venv passes the test suite.
- **Note from U0-15:** ADR 0009 fixes the extras split (`ml` scikit-learn, `timeseries` statsmodels, `text` rapidfuzz, `pandas`, `sql` Ibis, `server` FastAPI, SQLAlchemy, Alembic). Heavy libraries are imported lazily, and each extra needs a test that the core runs without it.
- **Commit:** `build: declare optional dependency extras`
- **Done notes:** `pyproject.toml` declares `ml`, `timeseries`, `text`, `pandas`, `sql`, `server` and `all` (the union, as `dsbf[...]`); the base install is unchanged, so each extra repeats a base constraint or adds a package nothing imports yet (ibis-framework, rapidfuzz, sqlalchemy, alembic). `tests/test_packaging.py` guards the set of extras and keeps repeated specifiers identical. The wheel with `[all]` in a clean venv passes the suite (1006 passed). Deviations: the first clean install failed 3 normality tests because DSBF never declared scipy and a fresh install gets 1.18, which breaks `kstest(..., args=...)`; scipy is now declared as `>=1.16,<1.18` with a Dependabot ignore and a held-back row. pyarrow stays in the base, as the design lists it, not in an extra as the U0-04 note wanted. Merged as PR #72 (`9f9d2c8`).

### U1-06 · Contracts I: subjects, severities, dimensions and evidence
- **Status:** done (2026-10-17) · **Size:** M · **Depends on:** U1-01 · **Resolves:** F-10, F-36 (schema side)
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
- **Done notes:** `dsbf.model` (`src/dsbf/model/`) holds `Subject`, `Severity`, `Confidence`, `Dimension`, `Evidence`, `Interval`, `AssumptionCheck`, `SamplingInfo` and `Timing`, and imports only pydantic (now a direct dependency, `pydantic>=2.12,<3`). Every contract derives from `Contract` (frozen, unknown fields rejected), and sequences are tuples, not the sketch's lists. `Evidence` nulls NaN, infinity and `NaT`, names them in `non_finite_metrics`, and turns NumPy scalars and dates into plain values and ISO strings; this also repairs a stored `NaN` token on read. `Subject` keeps columns as an ordered tuple, so the `a|b` pair collision cannot happen. `Severity` and `Confidence` are ordered string enums. Deviations: `Metric` also admits `bool`, because pydantic turned `True` into `1.0`; structural floats (interval bounds, timings) are rejected, not nulled, when non-finite. Checked on the adversarial frame: 5 of 5 non-finite metrics flagged, all output strict JSON, and the legacy `report.json` for the same run does not parse strictly (269 `NaN`, 19 `Infinity`). Merged as PR #74 (`837aea9`).

### U1-07 · Contracts II: findings, notes, check results and run health
- **Status:** done (2026-10-17) · **Size:** M · **Depends on:** U1-06 · **Resolves:** F-20 (status model)
- **Note from U1-06:** `dsbf.model` conventions to follow: models derive from `dsbf.model.base.Contract` (frozen, `extra="forbid"`), sequences are `tuple[...]`, structural floats use `FiniteFloat`, and data-derived metric values go through `Evidence` (nulled and flagged, never rejected). `Subject` keeps a pair's columns in the order given, so a symmetric check must pick a canonical order before `Finding.id` hashes it. `Severity` and `Confidence` compare by rank (`Severity.WARN >= Severity.NOTICE`); ordering `Run.findings` by severity can use that.
- **Goal:** `Finding` (with a stable id), `Guidance`, `Action`, `Note`, `CheckResult` (statuses `ok | not_applicable | skipped | unavailable | error | timeout`) and `RunHealth`.
- **Assumptions to confirm:**
  - The `Finding.id` hash inputs: check id, subject, code.
  - Mapping today's guidance blurbs onto `Guidance`/`Action` loses nothing. Compare with `BaseTask.add_guidance`.
- **Acceptance:**
  - Unit tests pass.
  - A test builds findings from a real legacy `TaskResult.guidance` sample without data loss.
  - `RunHealth` renders the plain-text summary shown in REFACTOR_PLAN §7.1.
- **Commit:** `feat(model): add finding, note, check result and run health contracts`
- **Done notes:** `dsbf.model` gained `Finding` (id from `Finding.build`, checked on every validation), `Guidance`, `Action`, `Note`, `CheckStatus`, `CheckResult` (a non-`ok` status needs a note and has no findings), `StatusCounts`, `HealthIssue` and `RunHealth` (derived by `from_results`; `render` reproduces the §7.1 summary exactly). Deviations: real titanic guidance showed the sketch was lossy, so a legacy `good` level is `Severity.INFO` plus the tag `good`, an `Action` keeps its own `columns` and extra `params`, `Evidence.metrics` admits one level of list or mapping, and each blurb becomes one `Finding` with a phase-only `Guidance`. All 589 titanic blurbs map to findings and back with no loss. `pydantic` moved to 2.14.1. Merged as PR #75 (`07c2f53`).

### U1-08 · Contracts III: profiles and the run manifest
- **Status:** done (2026-10-17) · **Size:** M · **Depends on:** U1-07
- **Note from U1-07:** 589 titanic findings serialise to 682 KB (about 1.2 KB each), so the size budget below covers the profile and manifest only; measure findings separately and say so in the acceptance.
- **Goal:** `ColumnProfile` (variants by semantic type), `DatasetProfile`, `DatasetManifest`, `PlanManifest` and `Run`, with `schema_version`.
- **Assumptions to confirm:** The profile fields cover everything listed in REFACTOR_PLAN §5.3 that the dashboard needs today. Check against current tab usage.
- **Acceptance:** A hand-built `Run` validates and round-trips through JSON. Its size for titanic-like content is under 200 KB without evidence artifacts.
- **Commit:** `feat(model): add profile and run manifest contracts`
- **Done notes:** `dsbf.model` gained `ColumnProfile` (seven variants chosen by `semantic_type`, with `identifier` and `unknown` added to the §5.3 list because the legacy `infer_types` emits them), `DatasetProfile`, `DatasetManifest`, `PlanManifest` and `Run` (`SCHEMA_VERSION = "0.1.0"`). `Run.build` derives `findings` and `health`, and its validator checks that the parts agree; `Run.render_health` supplies the header for `RunHealth.render`. Histograms are equal-width with the rule recorded (F-12), and undefined statistics are nulled and named in `non_finite_stats`. On a titanic profile computed in the tests, profile plus manifests are about 10 KB (budget 200 KB); findings are about 1.1 KB each, so a full 589-finding run is about 650 KB, doubled by `Run.findings` repeating each result's findings. Deviations: `Run.policy` waits for U1-10; mode and mode proportion are derived from top-k rather than stored; character-class and pattern statistics are left to U2-05. Merged as PR #76 (`4b76948`).

### U1-09 · Configuration and parameter base models
- **Status:** done (2026-10-17) · **Size:** M · **Depends on:** U1-06 · **Resolves:** F-22 (loader part)
- **Goal:** A `CheckParams` base class, and a layered config loader (defaults < project `dsbf.yaml` < env vars < CLI < keyword arguments) with deep merge and precise errors.
- **Assumptions to confirm:** The env var naming (`DSBF__SECTION__KEY`), and whether `pydantic-settings` is acceptable.
- **Acceptance:** Tests cover merge order, unknown keys (error naming the key and allowed keys), and type errors.
- **Commit:** `feat(config): add layered configuration and parameter base models`
- **Done notes:** `dsbf.model.CheckParams` is a frozen `Contract` whose fields all need defaults (checked when the class is defined). `dsbf.config.load_config(model, defaults=, project_file=, env=, cli=, overrides=)` merges the five layers (mappings merge, everything else replaces, inputs untouched) and validates; `ConfigError` lists every issue with its dotted path, the layer that supplied the value and, for an unknown key, the allowed keys. Env names are `DSBF__SECTION__KEY`, lowercased; a value starting `[` or `{`, or `null`, is read as JSON. Deviations: `pydantic-settings` was not adopted (it can't name the layer behind a bad key, and would add a base dependency); the loader is generic over any model because the root config model isn't designed yet; the legacy CLI and `EDA(...)` are not wired to it.

### U1-10 · Policy rule schema
- **Status:** todo · **Size:** S · **Depends on:** U1-07
- **Note from U1-06:** `Severity` and `Confidence` are ordered `StrEnum`s, so a `>=error` matcher can compare members directly. `Dimension` is a plain `StrEnum`.
- **Note from U1-08:** `Run` has no `policy` field yet. Add `policy: PolicyOutcome | None = None` to `dsbf.model.run.Run` (an additive change) and cover it in `tests/test_model/test_run.py`.
- **Goal:** Typed models for `policy.yaml` (`when` matchers, `then` actions `annotate | warn | fail | notify`) and `PolicyOutcome`. There's no evaluator yet (Phase 5).
- **Assumptions to confirm:** The REFACTOR_PLAN §7.2 example covers the matcher operators users need.
- **Acceptance:** The §7.2 example parses. Invalid rules produce precise errors.
- **Commit:** `feat(model): add policy rule schema`

### U1-11 · Schema export, breaking-change gate and TypeScript types
- **Status:** todo · **Size:** M · **Depends on:** U1-08, U1-09, U1-10, U1-03
- **Note from U1-04:** this unit edits `frontend/`, so apply the Dependency upgrades rule there. Dependabot PRs #69 (vue-router 4.6.4 → 5.4.0) and #70 (plotly.js-dist-min 3.7.0 → 4.1.2) were open on 2026-10-17. Both are major bumps with no unit yet; take them with the dashboard build and view the dashboard in a browser, or close them with a cap and a row in the held-back table.
- **Note from U1-06:** the models in `dsbf.model` set `additionalProperties: false`, and `Dimension`, `Severity` and `Confidence` export as string `enum`s. The ordering of the last two is not in the schema, so a TypeScript consumer needs its own rank table. `Timing.started_at` is an ISO 8601 string with an offset.
- **Note from U1-08:** `Run` serialises every finding twice, under `results[*].findings` and again in `Run.findings` (the §4.2 sketch). A full titanic run is about 1.3 MB for that reason. Decide before the schema is published whether `Run.findings` should be derived on read instead of serialised. Removing it later would be a breaking change. `ColumnProfile` is a discriminated union (`oneOf` with a `semantic_type` discriminator), so check how the TypeScript generator renders it. `Run.schema_version` is validated against the current major.
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
- **Note from U1-05:** pyarrow is still a base dependency (the design keeps it there), though nothing but pandas' parquet engine imports it. Confirm whether the Polars parquet path needs it before settling which install carries it.
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
- **Note from U1-08:** the contracts exist in `dsbf.model.profile`: `Histogram` (equal-width, `rule` is `fd | sturges | scott | sqrt | fixed`), `ValueCount`, `LengthStats` and `DatetimeProfile`. Character-class and pattern-signature statistics (REFACTOR_PLAN §5.3) are not in `TextProfile` yet; add them as optional fields, which is additive. `tests/test_model/titanic_profile.py` computes a reference profile with pandas, usable as an oracle for the parity report.
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
- **Note from U1-04:** `record_run` no longer exists, so `tests/perf/baseline.json` still carries a `record_run` stage (0.0 s) in each scenario. Regenerate the baseline with the perf harness when this unit adds the profile pass, which drops the entry.
- **Note from U0-14:** `tests/perf/scenarios.py` has no 1M x 50 scenario, because today's engine would take hours on it. Add it here if the new profile pass can run it; the REFACTOR_PLAN §11 budget names it. The baseline shows where the legacy time goes: `fuzzy_duplicate_detection` (65% of titanic, 46% of kitchen5k) and `detect_stationarity` (40% of kitchen50k).
- **Note from U0-17:** the perf comparison is noisy at task level on shared CI runners. Two runs of the same tree (PR #64, 2026-10-16) flagged different single tasks: the first `generate_dataset_summary_plots` (+52% on kitchen5k, +28% on kitchen50k, about 1.1 s each, just over the 1 s floor), the second `detect_stationarity` on kitchen50k (89 s → 121 s, +36%), and neither flagged a stage or a total. A local A/B of the plotting task under the old and new dependencies showed no difference (median 1.11 s vs 1.03 s). Before the budget is enforced as a gate, make the check robust: compare the median of repeated runs for the kitchen scenarios, or give long tasks a wider tolerance, so a clean tree doesn't fail by chance.
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
- **Note from U1-07:** `CheckResult.output` is an open `dict[str, Any]` (made JSON-safe, undefined numbers nulled and named in `non_finite_outputs`). Typed per-check output models belong here.
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
- **Note from U1-07:** blurbs carry no `code`, `dimension`, `confidence` or (always) `n_used`; the adapter supplies them. A `code` must be unique within a check and subject and must not depend on blurb order (the contract test's `<phase>_<index>` shifts when a task reorders blurbs). Map `good` to `Severity.INFO` plus the tag `good`, `__dataset__` to `Subject.dataset()`, and legacy `column`/`columns` on an action to `Action.columns`. `tests/test_model/test_legacy_guidance.py` holds the reference mapping.
- **Goal:** `LegacyTaskCheck` wraps any `BaseTask`. It provides a legacy `AnalysisContext` shim (data, config, `semantic_types` from the new profile) and maps `TaskResult` statuses onto the new statuses.
- **Assumptions to confirm:** Which `context.metadata` keys legacy tasks read from each other (grep `get_metadata`, `context.results`).
- **Acceptance:** Five representative legacy tasks (a rule-like task, a pandas-heavy one, a plotting one, a synthesis scorer, and a time-series one) run through the new engine with their outputs unchanged.
- **Commit:** `feat(engine): add legacy task adapter`

### U3-11 · All legacy tasks through the new engine with findings
- **Status:** todo · **Size:** M · **Depends on:** U3-10
- **Note from U1-06:** `tests/perf/baseline.json` still records the HTTP 500s on the adversarial run (`/associations` and `/sample` emit `NaN`; the U0-14 note is kept under U1-06). The legacy writer and API don't use `dsbf.model` yet. Once legacy results are routed through `Evidence` (nulled and flagged), or the API stops emitting non-finite floats, regenerate the baseline with the perf harness so the benchmark guards the 200s.
- **Note from U1-07:** use the mapping and the rules in the U3-10 note; the tests there convert a real titanic sample without loss.
- **Goal:** Every registered legacy task runs on the new engine. Guidance blurbs convert into `Finding`s with evidence where it can be derived.
- **Acceptance:** Golden runs (U0-13) reproduce through the adapter, with differences listed in the PR. The adversarial frame produces a complete `Run` with no unhandled errors.
- **Commit:** `feat(engine): run all legacy tasks through the new engine`

### U3-12 · `dsbf.profile()` entry point and run health output
- **Status:** todo · **Size:** M · **Depends on:** U3-11 · **Resolves:** F-16, F-17, F-18, F-34
- **Note from U1-01:** `dsbf version` runs `git describe --tags` (`utils/versioning.py`), so outside a checkout it prints `dev` plus a `fatal: not a git repository` line, even from an installed wheel. Read the version from `importlib.metadata`.
- **Goal:** The public `dsbf.profile(source, packs=…, checks=…, target=…, time_index=…, sample=…)` returns an in-memory `Run` with no filesystem writes. Version comes from `importlib.metadata`. `run.health.summary()` matches REFACTOR_PLAN §7.1.
- **Acceptance:**
  - A test asserts no files are created during `dsbf.profile()`.
  - The README library example works.
  - Phase 3 exit checks pass: golden via the adapter, forced exception gives a complete `Run`, a parameter change re-executes only the affected checks.
- **Commit:** `feat: add dsbf.profile() library entry point`
- **Note from U1-09:** `dsbf.config.load_config(model, ...)` is generic over any model. This unit defines the root config model (engine, limits, `profiling_depth`, `backend`, `sample_strategy` as enums, so F-22's third bullet is covered) and builds `profile(...)` keyword arguments into the `overrides` layer. The caller's config must not be mutated (F-22's second bullet): `load_config` copies its inputs.

---

## Phase 4: Migrate checks (placeholder)

- **U4-00 · Plan Phase 4** (`todo`, depends on U3-12): break down the batches below into full units using Phase 3 learnings and golden-run evidence. Update REFACTOR_PLAN if needed. Carry in from U0-15 (ADR 0009): shrink the base install to the core dependency set as the legacy tasks migrate, moving the rest behind extras and deleting dependencies nothing imports (for example matplotlib, plotly, seaborn, networkx, graphviz, pydot and missingno, to be verified by grep). Carry in from U1-05: (1) each extra needs a test that the core runs without it and reports `unavailable` (ADR 0009), added by the unit that removes its package from `dependencies`; (2) CI runs the extras as an install matrix, one leg per extra, once the base shrinks and an extra's absence is observable; (3) decide pyarrow's home: the design keeps it in the base, while the U0-04 note wanted it in the extra for parquet input (pandas is its only parquet user today); (4) declare `numpy` (imported in 45 places, never declared) with a tested range when the core dependency set is fixed, and declare each other directly imported package at the same time; (5) the `scipy <1.18` cap: change the `kstest` call in `normality_tests.py` when migrating it, then lift the cap.
- U4-A · Rule checks onto the profile, about 20 tasks (`placeholder`). From U0-03: the migrated guards only check that some column matches; tasks still loop over every numeric or object dtype column, not `matched_cols`, so they also analyse columns whose semantic type was excluded. And a migrated task run without `infer_types` now skips instead of analysing all numeric columns. Decide the intended behaviour per check. From U0-08: the time-series test configs still carry an inert nested `time_series.tasks.<name>` block that no task reads (tasks read `tasks.<task name>`); drop it when those tests migrate. From U0-07: `get_recommendation_tip` (`dsbf/utils/reco_engine.py`) still runs `eval` on the YAML `condition` strings (F-31); replace it with a declarative condition format (`{"field": ..., "op": ">", "value": ...}`) when the 14 tasks that call it migrate
- U4-B · Analysis checks with statistical fixes: F-11, F-13, confidence intervals, bias-corrected Cramér's V (`placeholder`). From U0-09: `summarize_modes` iterates all columns rather than `matched_cols`, and the leakage correlation scan uses `select_dtypes` (F-11); the collinear check needs per-column VIF handling and `n_used` (F-32). From U0-16: when the VIF check is redesigned, lift the `statsmodels <0.15` cap and decide the intended VIF definition (centred, with an intercept, as 0.15 computes it, or the uncentred one the baselines record); when bimodality is migrated, lift `scikit-learn <1.8` and decide how the BIC comparison is reported (the absolute BIC and `delta / abs(bic1)` both depend on the log-likelihood convention; the delta itself did not move). Regenerate those golden snapshots in the same PR, with the differences explained
- U4-C · Heavy checks redesigned with budgets: fuzzy duplicates (F-35), time series (F-39), drift against reference profiles (`placeholder`). From U0-13: `fuzzy_duplicate_detection` builds candidate pairs from a token index over sets of strings and stops at `max_candidates`, so which pairs it keeps (and the count) follows `PYTHONHASHSEED`; make the order deterministic, then the golden runner no longer needs the pinned seed. On kitchen5k, `decompose_time_series` cannot infer a seasonal period (frequency `None` for a daily index with repeated dates), and the golden baseline records that.
- U4-D · Synthesis: quality dimensions, ML readiness, target-based leakage (F-26) (`placeholder`)
- U4-E · Remove the legacy adapter, backend plotting and plotting dependencies (N-A2, F-38) (`placeholder`). From U0-16: once no legacy task needs pandas, lift the `pandas <3` cap (PR #47 was closed as deferred; the golden suite and the pandas extra's tests must pass on 3.x)

## Phase 5: Policy, sinks, CLI v2. First release for embedded use (placeholder)

- **U5-00 · Plan Phase 5** (`todo`, depends on the Phase 4 units). Carry in from U1-09: CLI v2 builds its `cli` layer for `dsbf.config.load_config` and needs a `--set a.b=c` parser (U1-09 left it out); `dsbf run -c` still uses the YAML as the whole config (F-22, first bullet) and goes away with the legacy CLI. Carry in from U0-13 and U1-04: `tests/validation/run_validation.py` without `--report` looks for the newest `report.json` under `dsbf/outputs/`, which DSBF stopped writing to in U1-04 (runs now go under `$DSBF_HOME/outputs/`), so it finds nothing new, and used to take whatever dataset's report was newest (it checked `near_clean` against a stale 9,217-row run on 2026-10-16), and the README keeps a warning about it; make it profile into its own output directory, as the golden runner does, then drop the warning. Its `clean` checklist also says 10 columns where the dataset has 9. Carry in from U0-11: (1) `dsbf profile` defaults `--depth` to `standard`, while `default_config.yaml`, `dsbf quickstart` and `dsbf run` use `full`: pick one default for CLI v2. (2) The default config's `message_verbosity` comment lists `quiet | info | debug`, but the logger also accepts `warn` and `stage`. (3) Dead config keys `metadata.output_format`, `engine.severity_thresholds` and `metadata.layout_name` (F-28) go with the config redesign. Carry in from U1-04: a configured `output_dir` is used as given, so a relative one still stores relative figure paths (only the default location is absolute); resolve it to absolute in the config redesign.
- Placeholders: policy evaluator (report-only default) · JSON/NDJSON sinks · Parquet sinks with a consumer schema guide · static HTML report · CLI v2 (`profile`, `check`, `plan`, `schema export`, run health summary) · pipeline and consumer docs · install smoke test from a temp dir · release `1.0.0a1`

## Phase 6: Store, service and dashboard v2. Standalone local (placeholder)

- **U6-00 · Plan Phase 6** (`todo`, depends on the Phase 5 units)
- Placeholders: run repository on SQLite with Alembic · artifact store and per-run data snapshot · `dsbf runs …` · API v2 · local job runner · generated TS client · dashboard shell and run health · findings explorer · column view (reuse `wip/echarts-charts` components with real histogram bins) (U0-16 saw the light theme draw chart titles and card headings such as "Value Counts" almost white on a light background; same on echarts 5) · relationships · quality · ML readiness · time series · run compare and history · settings → rerun · frontend component and end-to-end tests (F-29) · clear the 20 frontend lint warnings from U1-03 and raise `no-unused-vars`, `vue/no-side-effects-in-computed-properties` and `vue/require-default-prop` to `error` (16 dead variables in 12 files, `acfSelectedColData` in `TimeSeriesTab.vue` assigns inside a `computed`, route props in `RunDetailView.vue` and `RunListView.vue`); revisit `vue/no-v-html` if another source appears · code-split the dashboard bundle (over 500 kB, Plotly and ECharts load eagerly) · bundle the dashboard into the wheel with `dsbf dashboard` · remove API v1 and the old store (F-15, F-25) · release `1.0.0`

## Phase 7: Scale and sources (placeholder)

- **U7-00 · Plan Phase 7**. Carry in from U0-15: re-check ADR 0004, that the profile interface can be expressed as an Ibis aggregation (with the approximate-statistics caveats), before committing to the pushdown design; amend or supersede the ADR if it can't.
- Placeholders: larger-than-memory files and object storage (T2) · Ibis sources with DuckDB/Postgres, then Snowflake/BigQuery (T3) · recurring datasets, history and profile-based drift (T4) · scheduling recipes

## Phase 8: Team and cloud (placeholder)

- **U8-00 · Plan Phase 8**
- Placeholders: Postgres repository · object-storage artifacts · auth (token, OIDC) · job queue · Docker image and Helm chart · GitHub Action · Airflow, Prefect and Dagster adapters · dbt recipe · notifiers and OpenLineage · MCP server

## Phase 9: Modalities and workflow phases (placeholder)

- **U9-00 · Plan Phase 9**
- Placeholders: text-corpus modality · domain packs · modelling-prep exports (ML guidance actions as sklearn pipelines) · modelling diagnostics
