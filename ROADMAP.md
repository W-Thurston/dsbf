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
| `pandas <3` (PR #47, closed 2026-10-16) | 3.0 fails the 3.12 and 3.13 test jobs (string dtype, copy-on-write); the legacy tasks run on pandas until they migrate | U4-E (the base cap; pandas 3 frames already load as an input type, U2-01) |
| `statsmodels <0.15` | 0.15's `variance_inflation_factor` includes the intercept: VIFs drop and four columns on the `clean` archetype stop being flagged | U4-B (VIF redesign, F-32) |
| `scikit-learn <1.8` | 1.8 changes `GaussianMixture` log-likelihood on discrete columns (titanic `pclass` BIC 1016 → 1763), so `detect_bimodal_distribution` reports other BIC values and finding titles | U4-B (bimodality is an analysis check) |
| `scipy <1.18` (found in U1-05, no PR) | 1.18 makes `kstest(x, "norm", args=(mu, sigma))` raise `TypeError: ndtr() takes from 1 to 2 positional arguments but 3 were given`, so `run_ks` and two normality tests fail on a fresh `pip install`; DSBF imported scipy without declaring it, so only the lock kept it on 1.16.3 | the unit that migrates the normality tests (Phase 4; U4-00 assigns it), or sooner if a unit touches `normality_tests.py` |
| `plotly <7` (PR #62, closed 2026-10-16) | 7 removes the `mapbox` layout property that `custom_plotly_templates.py` sets, so it fails at import; the backend plotting that uses plotly is deleted anyway | U4-E (plotly is deleted; drop the cap and the ignore entry) |
| `vue-router <5` (PR #69, ignored from 2026-10-17) | 5 is a major bump of the router the current dashboard is built on; checking it needs the dashboard viewed in a browser, and the dashboard is rebuilt in Phase 6 | U6-00 |
| `plotly.js-dist-min <4` (PR #70, ignored from 2026-10-17) | 4 is a major bump of the library behind the legacy charts; checking it needs the dashboard viewed in a browser, and the charts are rebuilt in Phase 6 | U6-00 |

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

**Done (2026-10-17), 11 units.** Detailed unit entries were condensed here at close-out; the per-unit notes that later units depend on were carried into those units as "Note from U1-xx" lines.

- **Capability delivered:** the `src/` layout, a top-level `frontend/` on pnpm with ESLint, Prettier and Vitest in CI, runtime paths under `DSBF_HOME`, the optional-dependency extras, and the versioned result contracts in `dsbf.model` (subjects, evidence, findings, check results, run health, profiles, manifests, `Run`, policy rules) with a layered config loader (`dsbf.config`). `dsbf schema export` publishes `schema/run.schema.json` and `frontend/src/generated/run.d.ts`, and a CI `contracts` job fails on a stale export or on a schema change its version number does not announce.
- **Success criteria, measured:**
  - Every Phase 1 unit is `done`: 11 of 11.
  - A hand-built `Run` validates, round-trips through JSON, validates against the exported schema with `jsonschema` and compiles against the generated types under `tsc --strict` (U1-11). The `ColumnProfile` union narrows exhaustively.
  - Size: profile plus manifests for titanic are about 10 KB against a 200 KB budget (U1-08); a 61-check, 610-finding run is 338 KB, down from 644 KB before `Run.findings` became derived (U1-11).
  - Breaking-change gate, exercised on a real field removal: stale export exit 1, unannounced removal exit 1, with `SCHEMA_VERSION` raised exit 0 (U1-11).
  - Packaging: the wheel's file list equals `git ls-files src` (U1-01); the wheel with `[all]` in a clean venv passes the suite, 1006 tests (U1-05).
  - Paths: a run started from a scratch directory writes only under the home, and 357 of 357 figures are served by the API from another directory; golden 10 of 10 unchanged (U1-04).
  - Non-finite values: 5 of 5 on the adversarial frame are nulled and flagged, and all output is strict JSON, where the legacy `report.json` has 269 `NaN` and 19 `Infinity` (U1-06).
  - Frontend: `pnpm lint && pnpm build && pnpm test:run` pass with 31 unit tests and 20 lint warnings recorded as follow-ups (U1-03).
- **Key decisions:** pydantic v2 contracts that are frozen, reject unknown fields and use tuples ([ADR 0005](docs/adr/0005-pydantic-contracts.md)); undefined numbers become `None` plus a flag, never an error; `Finding.id` hashes check, subject and code; the home is `DSBF_HOME`, else the per-user data directory, chosen by the user on 2026-10-17 so one build serves a laptop and a cloud job ([ADR 0008](docs/adr/0008-sqlalchemy-alembic-fsspec-storage.md)); extras follow [ADR 0009](docs/adr/0009-single-distribution-with-extras.md) while the base install is unchanged; Prettier was chosen over ESLint stylistic rules (U1-03); the config loader is hand-written, because `pydantic-settings` cannot name the layer behind a bad key; the published schema describes a dumped run (serialization mode, defaults required) and treats any change to a type constraint as breaking, with the version rule in `CONTRIBUTING.md`; `Run.findings` is derived from `results`, not serialised.
- **Deviations from the phase block:**
  - The phase promised "packaging, paths and models only". U1-04 also made the metadata export opt-in (`dsbf tasks export`) and removed `record_run`; U1-05 declared scipy, which a clean install showed was imported but never declared, and capped it below 1.18; U1-09 added the config loader and U1-10 the policy schema, which the phase block did not list by name.
  - The policy matcher set grew beyond the REFACTOR_PLAN §7.2 example (all six comparison operators; `check`, `confidence`, `tag` and `column` matchers; `then` accepts a list) because the example cannot scope a rule to a check or column (U1-10).
  - `identifier` and `unknown` were added to the profile variants because the legacy `infer_types` emits them (U1-08). `pyarrow` stayed in the base install, as the design lists it (U1-05).
  - `Policy` is not in the published schema, because its loose spellings normalise on load; it ships with its loader (carried into U5-00). The dashboard was not viewed in a browser, and Dependabot's vue-router 5 and plotly.js 4 are held back (rows above, revisited in U6-00).
  - Phase 2 is already broken into full units, so it has no `U2-00`; U2-01 is next.
- **PRs:** #66 (U1-01, `bfc43e1`), #67 (U1-02, `b99633a`), #68 (U1-03, `c82fe45`), #71 (U1-04, `77d1992`), #72 (U1-05, `9f9d2c8`), #74 (U1-06, `837aea9`), #75 (U1-07, `07c2f53`), #76 (U1-08, `4b76948`), #77 (U1-09, `4af82d8`), #78 (U1-10, `3f39519`), and #79 (U1-11, `3b76aa7`).

---

## Phase 2: Data and profile

- **Capability:** Tier-1 sources (in-memory data and files), a `Dataset` with fingerprint and sampling provenance, and a single profiling pass that replaces per-task rescans.
- **Motivation:** "Measure once, decide many times" (REFACTOR_PLAN §2). It removes the repeated per-column computation and the per-task pandas copies, and fixes type inference cost, listwise deletion and histogram honesty at the source.
- **Boundaries:** No check API and no changes to legacy tasks. The profile is new code, compared against legacy outputs.
- **Sequence and dependencies:** Depends on Phase 1 contracts. Sources, then dataset, then type inference, then column statistics, then dataset-level profile, then parity, then the performance budget.
- **Success criteria:** Profile parity with legacy outputs is documented, and the profile pass runs in under 2 s at 52k × 60 on CI hardware.

**Done (2026-10-18), 8 units.** Detailed unit entries were condensed here at close-out; the per-unit notes that later units depend on were carried into those units as "Note from U2-xx" lines.

- **Capability delivered:** `dsbf.data.load` reads pandas, Polars and Arrow objects and CSV, TSV, Parquet, Excel, JSON and NDJSON files into a Polars-backed `Dataset` with a content fingerprint, seeded sampling provenance and a bounded value cache. `dsbf.profile` runs one pass: `infer_column_types` (sample-based semantic types), `profile_columns` (counts, numeric statistics, equal-width histograms, top values, string shape, datetime frequency and gaps) and `profile_dataset` (shape, duplicate rows, missingness pairs), with `infer_stage` reading only the profile. Typed errors carry a hint.
- **Success criteria, measured:**
  - Parity is documented: `tests/test_profile/accepted.py` lists 23 accepted differences in 7 areas, `docs/parity/profile.md` explains each, and a test fails if the two drift (U2-07). Counts, distinct values and numeric statistics equal the legacy tasks within `rel=1e-9` on all 10 golden datasets with no accepted differences (U2-04).
  - The profile pass at kitchen50k (52,500 × 60) takes 0.44 s on the 4-CPU runner against the 2 s budget, with the pass adding 3 MB to a 47 MB frame (`memory_ratio` 1.06 against 2.5). On 1M × 50 floats it takes 9.5 s and the ratio is 1.12 (U2-08). The legacy type inference alone took 25.95 s on kitchen50k; the new one takes 0.2–0.3 s (U2-03).
  - A 52.5k × 60 profile serialises to 59 KB and a 1M × 50 profile to 155 KB, with no non-finite token (U2-08).
  - Perf harness: both profile scenarios are in `tests/perf/baseline.json`, recorded on `ubuntu-latest`, and the budgets are checked whether or not a baseline exists.
- **Key decisions:** Polars and Arrow as the data representation, with the frame read-only after load ([ADR 0003](docs/adr/0003-polars-arrow-data-representation.md)); profile and semantic-type contracts are pydantic models ([ADR 0005](docs/adr/0005-pydantic-contracts.md)); the profile reads a seeded sample for type rules but counts over the whole column, and distinct counts are exact; the profile pass is timed apart from loading, and its memory is measured as the frame plus what the pass adds (`/proc/self/clear_refs` resets the peak after load).
- **Deviations from the phase block:**
  - "No changes to legacy tasks" held, but U2-01 lifted the Polars cap (golden 10 of 10 unchanged) and added the `dsbf[excel]` extra, which the unit did not list.
  - Profile parity is mostly vocabulary and typing: 9 of 10 golden datasets get a different stage name (5 only by vocabulary, 4 because high cardinality counts categorical columns only), and integer codes typed `categorical` carry no min and max.
  - `SCHEMA_VERSION` went 0.1.0 → 0.3.0 over U2-05 and U2-06, both additive.
  - Phase 3 is already broken into full units, so it has no `U3-00`; U3-01 is next.
- **PRs:** #80 (U2-01, `876a17c`), #81 (U2-02, `62efb95`), #82 (U2-03, `83a62b1`), #83 (U2-04, `c472927`), #84 (U2-05, `2a15752`), #85 (U2-06, `7fc2643`), #86 (U2-07, `fd7906e`), and U2-08 (number and merge SHA recorded when the next unit starts).

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
- **Note from U2-06:** `infer_stage` rates data by its mean column null ratio, so titanic is `cleaned` although `cabin` is 77% null and 709 of 891 rows have a null. When the inferred stage drives selection here, decide whether a per-column rule (the worst column, or the share of columns over a null limit) is needed.
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
- **Note from U2-08:** on 1M × 50 floats the profile pass takes 9.5 s on the 4-CPU runner (type inference 1.5 s, `profile_dataset` 8.0 s), well inside the 3 min standard-plan target, so no optimisation was needed. If measuring the standard plan on that frame shows the profile as a cost, the options are arithmetic histogram binning with an edge fix-up (about 1.3 s of the column pass at 1M × 50 locally), a sketched distinct count above a row threshold, and sharing `n_unique` and null counts between type inference and the count pass. The run health output's memory figure should say what it measures: `Dataset.frame.estimated_size()` undercounts a CSV-loaded frame (kitchen50k is 47 MB estimated and about 180 MB resident after load). The task-level legacy perf comparison is noisy on shared runners (U0-17); the profile scenarios were within 2% across 3 runs.
- **Note from U2-06:** call `dsbf.profile.profile_dataset` and `infer_stage` (with `StageThresholds.from_config`) from the entry point, and remove the legacy `eda/stage_inference.py` and its call in `ProfileEngine` when the engine path is replaced.
- **Note from U2-03:** the identifier and period-label rules use the share of distinct values, so a row sample can type a column differently from the full data (kitchen50k's `high_cardinality_random` is `categorical` whole and `identifier` in a 2000-row random sample). Decide whether `sample=…` types the full dataset first and profiles the sample, or types the sample and records it in `semantic_sample_rows`.
- **Note from U2-02:** `Dataset.sample(rows, strategy=, seed=, stratify_by=)` is the sampling to build `sample=…` on. The legacy `data_sampling` still has the F-09 crash and the `stratified` alias for head; remove it when the legacy engine path is replaced. A stratified or random sample records its seed in `SamplingInfo`.
- **Note from U2-01:** the legacy `dsbf profile` still fails on a parquet with a `datetime64` column (`TypeError: Object of type Timestamp is not JSON serializable`, in the legacy report writer). It goes away when results are written from the `Run` contract; add a datetime64 parquet to this unit's end-to-end test to confirm.
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
- U4-A · Rule checks onto the profile, about 20 tasks (`placeholder`). From U2-07: a categorical column of integer codes has no min and max, where the legacy `summarize_numeric` summarised it; the check that replaces it decides whether to read such codes as numeric. No golden dataset gives `summarize_text_fields` output, so `avg_word_length` was never compared to the legacy task; add a text-heavy golden dataset when the text checks move. From U2-05: a categorical column with mostly unique strings lists first-seen one-off values as top values (text columns skip values seen once); decide with the checks that consume top values whether categorical should also require a repeat. From U2-03: `load` decodes a non-UTF-8 CSV lossily with no error (a UTF-16 file reads with U+FFFD and NUL characters in names and values), so add a rule check that flags U+FFFD and NUL in column names and values; a column that mixes date formats (kitchen5k's `inconsistent_datetime`) is typed `categorical`, so a check could report the mix. From U0-03: the migrated guards only check that some column matches; tasks still loop over every numeric or object dtype column, not `matched_cols`, so they also analyse columns whose semantic type was excluded. And a migrated task run without `infer_types` now skips instead of analysing all numeric columns. Decide the intended behaviour per check. From U0-08: the time-series test configs still carry an inert nested `time_series.tasks.<name>` block that no task reads (tasks read `tasks.<task name>`); drop it when those tests migrate. From U0-07: `get_recommendation_tip` (`dsbf/utils/reco_engine.py`) still runs `eval` on the YAML `condition` strings (F-31); replace it with a declarative condition format (`{"field": ..., "op": ">", "value": ...}`) when the 14 tasks that call it migrate
- U4-B · Analysis checks with statistical fixes: F-11, F-13, confidence intervals, bias-corrected Cramér's V (`placeholder`). From U0-09: `summarize_modes` iterates all columns rather than `matched_cols`, and the leakage correlation scan uses `select_dtypes` (F-11); the collinear check needs per-column VIF handling and `n_used` (F-32). From U0-16: when the VIF check is redesigned, lift the `statsmodels <0.15` cap and decide the intended VIF definition (centred, with an intercept, as 0.15 computes it, or the uncentred one the baselines record); when bimodality is migrated, lift `scikit-learn <1.8` and decide how the BIC comparison is reported (the absolute BIC and `delta / abs(bic1)` both depend on the log-likelihood convention; the delta itself did not move). Regenerate those golden snapshots in the same PR, with the differences explained
- U4-C · Heavy checks redesigned with budgets: fuzzy duplicates (F-35), time series (F-39), drift against reference profiles (`placeholder`). From U0-13: `fuzzy_duplicate_detection` builds candidate pairs from a token index over sets of strings and stops at `max_candidates`, so which pairs it keeps (and the count) follows `PYTHONHASHSEED`; make the order deterministic, then the golden runner no longer needs the pinned seed. On kitchen5k, `decompose_time_series` cannot infer a seasonal period (frequency `None` for a daily index with repeated dates), and the golden baseline records that.
- U4-D · Synthesis: quality dimensions, ML readiness, target-based leakage (F-26) (`placeholder`). From U2-03: the identifier rule treats any string column over 90% distinct as `identifier`, so near-unique free text (kitchen50k's `long_text`) is not `text`; revisit the identifier-before-text order with the checks that consume the type.
- U4-E · Remove the legacy adapter, backend plotting and plotting dependencies (N-A2, F-38) (`placeholder`). From U0-16: once no legacy task needs pandas, lift the `pandas <3` cap (PR #47 was closed as deferred; the golden suite and the pandas extra's tests must pass on 3.x)

## Phase 5: Policy, sinks, CLI v2. First release for embedded use (placeholder)

- **U5-00 · Plan Phase 5** (`todo`, depends on the Phase 4 units). Carry in from U1-09: CLI v2 builds its `cli` layer for `dsbf.config.load_config` and needs a `--set a.b=c` parser (U1-09 left it out); `dsbf run -c` still uses the YAML as the whole config (F-22, first bullet) and goes away with the legacy CLI. Carry in from U0-13 and U1-04: `tests/validation/run_validation.py` without `--report` looks for the newest `report.json` under `dsbf/outputs/`, which DSBF stopped writing to in U1-04 (runs now go under `$DSBF_HOME/outputs/`), so it finds nothing new, and used to take whatever dataset's report was newest (it checked `near_clean` against a stale 9,217-row run on 2026-10-16), and the README keeps a warning about it; make it profile into its own output directory, as the golden runner does, then drop the warning. Its `clean` checklist also says 10 columns where the dataset has 9. Carry in from U0-11: (1) `dsbf profile` defaults `--depth` to `standard`, while `default_config.yaml`, `dsbf quickstart` and `dsbf run` use `full`: pick one default for CLI v2. (2) The default config's `message_verbosity` comment lists `quiet | info | debug`, but the logger also accepts `warn` and `stage`. (3) Dead config keys `metadata.output_format`, `engine.severity_thresholds` and `metadata.layout_name` (F-28) go with the config redesign. Carry in from U1-04: a configured `output_dir` is used as given, so a relative one still stores relative figure paths (only the default location is absolute); resolve it to absolute in the config redesign. Carry in from U1-10: (1) the policy loader (`--policy`) turns the validation failures of `yaml.safe_load` + `Policy.model_validate` into a `ConfigError`-style report with the rule name; (2) the evaluator decides what `new_since: previous_run` does without a previous run; (3) a range on one setting (`>0.1` and `<0.5`) needs a list of comparisons per `metric` key, `severity` or `confidence`: add it if users ask. Carry in from U1-11: publish the `Policy` schema (`schema/policy.schema.json`, with the loose spellings documented) together with its loader, because the model's own schema describes only the normalised form.
- Placeholders: policy evaluator (report-only default) · JSON/NDJSON sinks · Parquet sinks with a consumer schema guide · static HTML report · CLI v2 (`profile`, `check`, `plan`, `schema export`, run health summary) · pipeline and consumer docs · install smoke test from a temp dir · release `1.0.0a1`

## Phase 6: Store, service and dashboard v2. Standalone local (placeholder)

- **U6-00 · Plan Phase 6** (`todo`, depends on the Phase 5 units). Carry in from U1-11: (1) the dashboard is JavaScript, so it can use `frontend/src/generated/run.d.ts` only through JSDoc `import()` types or after a move to TypeScript; (2) a TypeScript consumer needs its own rank table for `Severity` and `Confidence`, which the schema cannot carry; (3) revisit the `vue-router <5` and `plotly.js-dist-min <4` caps in the held-back table.
- Placeholders: run repository on SQLite with Alembic · artifact store and per-run data snapshot · `dsbf runs …` · API v2 · local job runner · generated TS client · dashboard shell and run health · findings explorer · column view (reuse `wip/echarts-charts` components with real histogram bins; the profile now carries equal-width bins from U2-05, while the dashboard still draws the percentile-binned histogram of F-12) (U0-16 saw the light theme draw chart titles and card headings such as "Value Counts" almost white on a light background; same on echarts 5) · relationships · quality · ML readiness · time series · run compare and history · settings → rerun · frontend component and end-to-end tests (F-29) · clear the 20 frontend lint warnings from U1-03 and raise `no-unused-vars`, `vue/no-side-effects-in-computed-properties` and `vue/require-default-prop` to `error` (16 dead variables in 12 files, `acfSelectedColData` in `TimeSeriesTab.vue` assigns inside a `computed`, route props in `RunDetailView.vue` and `RunListView.vue`); revisit `vue/no-v-html` if another source appears · code-split the dashboard bundle (over 500 kB, Plotly and ECharts load eagerly) · bundle the dashboard into the wheel with `dsbf dashboard` · remove API v1 and the old store (F-15, F-25) · release `1.0.0`

## Phase 7: Scale and sources (placeholder)

- **U7-00 · Plan Phase 7**. Carry in from U0-15: re-check ADR 0004, that the profile interface can be expressed as an Ibis aggregation (with the approximate-statistics caveats), before committing to the pushdown design; amend or supersede the ADR if it can't.
- **Carry in from U2-01:** `dsbf.data.load` reads one local file, the first Excel sheet only, and rejects directories, globs and URLs (`UnsupportedSourceError`). Sheet choice, globs and `s3://`-style paths belong to T2.
- Placeholders: larger-than-memory files and object storage (T2) · Ibis sources with DuckDB/Postgres, then Snowflake/BigQuery (T3) · recurring datasets, history and profile-based drift (T4) · scheduling recipes

## Phase 8: Team and cloud (placeholder)

- **U8-00 · Plan Phase 8**
- Placeholders: Postgres repository · object-storage artifacts · auth (token, OIDC) · job queue · Docker image and Helm chart · GitHub Action · Airflow, Prefect and Dagster adapters · dbt recipe · notifiers and OpenLineage · MCP server

## Phase 9: Modalities and workflow phases (placeholder)

- **U9-00 · Plan Phase 9**
- Placeholders: text-corpus modality · domain packs · modelling-prep exports (ML guidance actions as sklearn pipelines) · modelling diagnostics
