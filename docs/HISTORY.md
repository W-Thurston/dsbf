# DSBF Program History

> **Reference document.** This file is the record of finished programs. Upcoming work is in
> [`ROADMAP.md`](../ROADMAP.md), which keeps one row per finished program and links here.

Each finished program has a summary, written at its closeout (see
[`.claude/commands/closeout.md`](../.claude/commands/closeout.md)). It covers the capability
delivered, the success criteria with measured results, the key decisions, the deviations from the
phase block, and the PRs. The summaries are moved here unchanged. A "Note from Ux-yy" mentioned in
a summary was carried into a later unit's entry in `ROADMAP.md`; once that unit is done, its PR
holds the note.

How each finding in [`REVIEW_FINDINGS.md`](REVIEW_FINDINGS.md) stands after these programs is
recorded in that file, next to the finding.

## Units and PRs

Every unit is one squash-merged PR. The commit subject is the PR title.

| Unit | Commit subject | PR | Merge | Merged |
|---|---|---|---|---|
| U0-01 | docs: adopt unit workflow and add refactor plan, review findings and walkthrough | #29 | `85f8697` | 2026-10-15 |
| U0-02 | ci: pin ruff to the locked version and move actions to Node 24 | #31 | `4b80da8` | 2026-10-15 |
| U0-03 | fix(core): keep task-specific empty result shapes and repair failing tests | #32 | `fb6d825` | 2026-10-15 |
| U0-04 | chore(deps): record dependency backlog outcomes | #40 | `7ed185b` | 2026-10-16 |
| U0-05 | build: migrate dependency management from Poetry to uv | #41 | `81d53f5` | 2026-10-16 |
| U0-06 | test: classify tests by tier and add a fast gate | #48 | `b01529d` | 2026-10-16 |
| U0-07 | test: isolate test runs and load package data via importlib.resources | #50 | `db9e08a` | 2026-10-16 |
| U0-08 | fix(config): move shared time_series settings out of the tasks block | #51 | `40f4db9` | 2026-10-16 |
| U0-09 | fix(tasks): guard against all-null, non-finite and delimiter-bearing columns | #52 | `ad53cc4` | 2026-10-16 |
| U0-10 | fix(dashboard): escape names before highlighting search matches | #53 | `016f587` | 2026-10-16 |
| U0-11 | chore: remove stray files and dead modules, correct the README | #54 | `4c1baf3` | 2026-10-16 |
| U0-12 | build: add pyrefly type checking with a baseline for existing code | #55 | `ecc2353` | 2026-10-16 |
| U0-13 | test: add golden-run baselines for archetypes, titanic and kitchen5k | #56 | `9115786` | 2026-10-16 |
| U0-14 | test: add performance benchmark harness and baseline | #57 | `33fcc3f` | 2026-10-16 |
| U0-15 | docs(adr): record architecture decisions for the library-first redesign | #58 | `26beda8` | 2026-10-16 |
| U0-16 | fix(security): clear code-scanning alerts and the dependency backlog | #59 | `751ad9d` | 2026-10-16 |
| U0-17 | chore(deps): take the second dependency batch and defer plotly 7 | #64 | `de863ab` | 2026-10-17 |
| U1-01 | refactor: move package to src layout | #66 | `bfc43e1` | 2026-10-17 |
| U1-02 | refactor(dashboard): move frontend to top-level directory | #67 | `b99633a` | 2026-10-17 |
| U1-03 | build(dashboard): move frontend to pnpm with lint and unit tests | #68 | `c82fe45` | 2026-10-17 |
| U1-04 | feat: resolve runtime paths from DSBF_HOME | #71 | `77d1992` | 2026-10-17 |
| U1-05 | build: declare optional dependency extras | #72 | `9f9d2c8` | 2026-10-17 |
| U1-06 | feat(model): add subject, evidence and severity contracts | #74 | `837aea9` | 2026-10-17 |
| U1-07 | feat(model): add finding, note, check result and run health contracts | #75 | `07c2f53` | 2026-10-17 |
| U1-08 | feat(model): add profile and run manifest contracts | #76 | `4b76948` | 2026-10-17 |
| U1-09 | feat(config): add layered configuration and parameter base models | #77 | `4af82d8` | 2026-10-17 |
| U1-10 | feat(model): add policy rule schema | #78 | `3f39519` | 2026-10-17 |
| U1-11 | feat(model): export JSON Schema and TypeScript types with a breaking-change gate | #79 | `3b76aa7` | 2026-10-17 |
| U2-01 | feat(data): add tier-1 source adapters | #80 | `876a17c` | 2026-10-17 |
| U2-02 | feat(data): add dataset fingerprint, sampling provenance and value cache | #81 | `62efb95` | 2026-10-18 |
| U2-03 | feat(profile): add sample-based semantic type inference | #82 | `83a62b1` | 2026-10-18 |
| U2-04 | feat(profile): add count and numeric column statistics | #83 | `c472927` | 2026-10-18 |
| U2-05 | feat(profile): add histograms, top-k, text and datetime profiles | #84 | `2a15752` | 2026-10-18 |
| U2-06 | feat(profile): add dataset-level profile and stage inference | #85 | `7fc2643` | 2026-10-18 |
| U2-07 | test(profile): add legacy parity comparison and accepted differences | #86 | `fd7906e` | 2026-10-18 |
| U2-08 | test(perf): enforce profiling pass budget | #87 | `74cd9d1` | 2026-10-18 |
| U3-01 | feat(engine): add check interface and check context | #88 | `729d288` | 2026-10-18 |
| U3-02 | feat(engine): add registry, packs and entry-point plugin discovery | #89 | `7f33080` | 2026-10-18 |
| U3-03 | feat(engine): add plan selection and dependency resolution | #90 | `61db3e0` | 2026-10-18 |
| U3-04 | feat(engine): validate plans and add dsbf plan | #91 | `4a22481` | 2026-10-18 |
| U3-05 | feat(engine): execute checks in isolation with explicit statuses | #92 | `b233bc8` | 2026-10-18 |
| U3-06 | feat(engine): add timeouts, resource accounting and execution events | #93 | `294c5e4` | 2026-10-18 |
| U3-07 | feat(engine): run independent checks in parallel | #94 | `75674f4` | 2026-10-18 |
| U3-08 | feat(engine): cache check results by data fingerprint and parameters | #95 | `3474ef0` | 2026-10-18 |
| U3-09 | refactor(logging): make logging library-friendly and markup-safe | #96 | `e401bb0` | 2026-10-19 |
| U3-10 | feat(engine): add legacy task adapter | #97 | `9fcb2bc` | 2026-10-19 |
| U3-11 | feat(engine): run all legacy tasks through the new engine | #98 | `dca2abe` | 2026-10-19 |
| U3-12 | feat: add dsbf.profile() library entry point | #99 | `ec202ed` | 2026-10-19 |

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
  - ADRs D1–D11 are `Accepted`: [0001](adr/0001-library-first-layering.md), [0002](adr/0002-findings-as-primary-output.md), [0003](adr/0003-polars-arrow-data-representation.md), [0004](adr/0004-ibis-warehouse-pushdown.md), [0005](adr/0005-pydantic-contracts.md), [0006](adr/0006-report-only-failure-semantics.md), [0007](adr/0007-entry-point-plugins.md), [0008](adr/0008-sqlalchemy-alembic-fsspec-storage.md), [0009](adr/0009-single-distribution-with-extras.md), [0010](adr/0010-check-naming.md), [0011](adr/0011-toolchain.md).
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
- **Key decisions:** pydantic v2 contracts that are frozen, reject unknown fields and use tuples ([ADR 0005](adr/0005-pydantic-contracts.md)); undefined numbers become `None` plus a flag, never an error; `Finding.id` hashes check, subject and code; the home is `DSBF_HOME`, else the per-user data directory, chosen by the user on 2026-10-17 so one build serves a laptop and a cloud job ([ADR 0008](adr/0008-sqlalchemy-alembic-fsspec-storage.md)); extras follow [ADR 0009](adr/0009-single-distribution-with-extras.md) while the base install is unchanged; Prettier was chosen over ESLint stylistic rules (U1-03); the config loader is hand-written, because `pydantic-settings` cannot name the layer behind a bad key; the published schema describes a dumped run (serialization mode, defaults required) and treats any change to a type constraint as breaking, with the version rule in `CONTRIBUTING.md`; `Run.findings` is derived from `results`, not serialised.
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
- **Key decisions:** Polars and Arrow as the data representation, with the frame read-only after load ([ADR 0003](adr/0003-polars-arrow-data-representation.md)); profile and semantic-type contracts are pydantic models ([ADR 0005](adr/0005-pydantic-contracts.md)); the profile reads a seeded sample for type rules but counts over the whole column, and distinct counts are exact; the profile pass is timed apart from loading, and its memory is measured as the frame plus what the pass adds (`/proc/self/clear_refs` resets the peak after load).
- **Deviations from the phase block:**
  - "No changes to legacy tasks" held, but U2-01 lifted the Polars cap (golden 10 of 10 unchanged) and added the `dsbf[excel]` extra, which the unit did not list.
  - Profile parity is mostly vocabulary and typing: 9 of 10 golden datasets get a different stage name (5 only by vocabulary, 4 because high cardinality counts categorical columns only), and integer codes typed `categorical` carry no min and max.
  - `SCHEMA_VERSION` went 0.1.0 → 0.3.0 over U2-05 and U2-06, both additive.
  - Phase 3 is already broken into full units, so it has no `U3-00`; U3-01 is next.
- **PRs:** #80 (U2-01, `876a17c`), #81 (U2-02, `62efb95`), #82 (U2-03, `83a62b1`), #83 (U2-04, `c472927`), #84 (U2-05, `2a15752`), #85 (U2-06, `7fc2643`), #86 (U2-07, `fd7906e`), and #87 (U2-08, `74cd9d1`).

---

## Phase 3: Engine and check API

- **Capability:** The check API, registry and packs, the planner, an isolated, parallel and cached executor, library-friendly logging, a legacy adapter running every existing task, and the public `dsbf.profile()` entry point.
- **Motivation:** "Choose your own analysis" and report-only failure handling (REFACTOR_PLAN §1, R1 and R2) live in the engine. The adapter lets checks migrate one at a time against a running system.
- **Boundaries:** No check migrations (Phase 4), and no storage, API or dashboard changes (Phase 6).
- **Sequence and dependencies:** Depends on Phase 2. Interface, registry, planner, executor (isolation, then timeouts and events, then parallelism), cache, logging, adapter, entry point.
- **Success criteria:** Golden runs reproduce through the adapter. A forced exception in any check still yields a complete `Run` with a note. Changing one parameter re-executes only the affected checks.

**Done (2026-10-19), 12 units.** Detailed unit entries were condensed here at close-out; the per-unit notes that later units depend on were carried into those units as "Note from U3-xx" lines.

- **Capability delivered:** `dsbf.engine` holds the `@check` interface and `CheckContext`, a `Registry` with packs and directory and entry-point plugins, a planner (`Selection`, `make_plan`, validation, expected statuses, `dsbf plan`) and an executor that isolates each check, applies timeouts and a run budget, accounts for time and memory, emits events, runs a level's checks in parallel (threads, or one process per check with `runs_in="process"`) and caches `ok` results on disk by data fingerprint and parameters. Logging is library-friendly (`dsbf.*` loggers with a `NullHandler`; the CLI attaches handlers). `dsbf.legacy` adapts all 65 registered tasks to checks and turns their guidance into `Finding`s. `dsbf.profile(source, packs=, checks=, depth=, target=, time_index=, sample=, ...)` returns an in-memory `Run` (`Run.notes`, `Run.render_health()`) from a `ProfileConfig`; the version comes from `importlib.metadata`.
- **Success criteria, measured:**
  - Golden runs reproduce through the adapter: `tests/golden/test_engine_parity.py` runs the ten golden datasets through the engine and compares each task with its snapshot. 128 differences are listed in `tests/golden/engine_differences.json`, each with a reason (`types`, `load`, `downstream`); on titanic 55 of 56 `ok` tasks match exactly (U3-10, U3-11). The golden tier is 20 of 20 on the final tree (U3-12).
  - A forced exception still gives a complete `Run`: a plugin check that raises ends `error` with a `check_failed` note and the rest of the run is intact (U3-05, U3-12); the adversarial frame gives 67 results, no `check_failed` (U3-11). A timeout does not stall the run (U3-06).
  - A parameter change re-executes only the affected checks: on titanic with the cache, a first call takes 4.8 s and a second 0.3 s with 52 of 59 results cached, and changing `detect_zeros.flag_threshold` re-ran exactly `detect_zeros` and the two scorers that read it (U3-08, U3-12).
  - No filesystem writes: a test runs `dsbf.profile()` from a scratch directory under an isolated `DSBF_HOME` and finds both empty (U3-12). The README library example is executed by a test.
  - `dsbf.profile()` on titanic at `deep` runs 63 checks (`ok 53 · not applicable 5 · skipped 5`, no error) into an 8.5 MB `Run`; 22.4 s serial and 24.7 s with the default workers locally. kitchen5k takes 104.1 s and 120.3 s. The parallel executor does not speed up the adapted tasks (pandas and Python code under threads); on toy checks over 1M × 50 floats it does: 1.41 s against 1.02 s for compiled checks, and 4.22 s against 1.75 s for pure-Python checks in worker processes (U3-07).
  - Schema: `SCHEMA_VERSION` 0.3.0 → 0.6.0 (U3-07 `Timing.shared_with`, U3-08 `Timing.cached`, U3-12 `Run.notes`, which also removed `PlanManifest.packs`).
- **Key decisions:** a failing check is data, never an exception ([ADR 0006](adr/0006-report-only-failure-semantics.md)), with the returned status trusted and never inferred (F-06); plugins by entry point or directory, and a broken plugin is a note ([ADR 0007](adr/0007-entry-point-plugins.md)); a timeout on a thread is cooperative (the context is cancelled, then the thread is abandoned), and a process check is killed; the cache key includes the digest of each upstream result, so a dependent re-runs only when what it reads changed; `Depth` is a cost ceiling and the inferred stage selects nothing; a library call writes nothing unasked, so the cache is off (`cache=True` opts in) and the two plot-writing legacy checks are left out of an in-memory run (chosen by the user on 2026-10-18); the subpackage `dsbf.profile` was renamed `dsbf.profiling` so `dsbf.profile` is the function (chosen by the user on 2026-10-18); the entry point defaults to the pandas backend, because the legacy tasks are written for pandas frames and the golden snapshots are pandas runs.
- **Deviations from the phase block:**
  - "Golden runs reproduce" holds with 128 listed differences, not byte for byte: mostly the new profile typing a column differently from the legacy inference (`types`), `dsbf.data.load` and `pandas.read_csv` reading the file differently (`load`), and effects of those on dependent tasks (`downstream`).
  - Timeouts are one value for every check, not per check (per-check overrides go to the Phase 5 config). Process isolation is one process per check, not a pool.
  - The CLI's `run`, `profile` and `quickstart` commands still use the legacy engine; `dsbf.profile()` reaches the CLI with CLI v2 in Phase 5. Only `dsbf plan` uses the new engine, and `dsbf version` now reads `importlib.metadata`.
  - The `engine-*` perf scenarios (serial and parallel, titanic and kitchen5k) are added, but their baseline can only be recorded from the CI runner, so it is open (carried into U4-00).
  - `RunHealth` has no `summary()`; the §7.1 text is `Run.render_health()`, and REFACTOR_PLAN §9 says so.
- **PRs:** #88 (U3-01, `729d288`), #89 (U3-02, `7f33080`), #90 (U3-03, `61db3e0`), #91 (U3-04, `4a22481`), #92 (U3-05, `b233bc8`), #93 (U3-06, `294c5e4`), #94 (U3-07, `75674f4`), #95 (U3-08, `3474ef0`), #96 (U3-09, `e401bb0`), #97 (U3-10, `9fcb2bc`), #98 (U3-11, `dca2abe`), and #99 (U3-12, `ec202ed`).
