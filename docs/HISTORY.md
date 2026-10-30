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
| U4-00 | docs: plan the check migration and record finding status | #100 | `0ef2161` | 2026-10-19 |
| U4-01 | perf: record engine baselines and turn check warnings into notes | #101 | `91c064d` | 2026-10-20 |
| U4-02 | feat(checks): add the core check pack with a migration harness | #102 | `aa2703a` | 2026-10-20 |
| U4-03 | feat(profile): add entropy, content hashes and encoding damage counts | #103 | `97c9d6d` | 2026-10-20 |
| U4-04 | feat(checks): add column health rule checks with declarative tips | #104 | `56a24d4` | 2026-10-20 |
| U4-05 | feat(checks): add distribution, type and schema rule checks | #105 | `9f5c5ea` | 2026-10-21 |
| U4-06 | feat(checks): add outlier, normality, bimodality and bounds checks | #108 | `63c2bbe` | 2026-10-21 |
| U4-07 | feat(checks): add association and contingency checks with intervals | #109 | `a6641b7` | 2026-10-22 |
| U4-08 | feat(checks): add group difference and collinearity checks | #110 | `5bb5a65` | 2026-10-22 |
| U4-09 | feat(checks): add missingness mechanism and string structure checks | #111 | `fb07cdb` | 2026-10-22 |
| U4-10 | feat(checks): redesign fuzzy duplicate detection with blocking and budgets | #115 | `05de0dd` | 2026-10-22 |
| U4-11 | feat(checks): redesign time-series checks with resampling and capped lags | #116 | `cddf99d` | 2026-10-23 |
| U4-12 | feat(checks): compare a dataset with a reference | #117 | `70fe926` | 2026-10-23 |
| U4-13 | feat(checks): add encoding, binning, transformation and imputation guidance | #118 | `2564726` | 2026-10-23 |
| U4-14 | feat(checks): detect target leakage and mutual information | #119 | `7df24ea` | 2026-10-23 |
| U4-15 | feat(checks): score quality dimensions and ML readiness from findings | #121 | `9fe8873` | 2026-10-23 |
| U4-16 | test: add a calibration suite of planted-truth datasets | #123 | `111bd9c` | 2026-10-24 |
| U4-17 | feat(cli): run dsbf profile on the new engine | #125 | `73b361a` | 2026-10-25 |
| U4-18 | refactor!: remove the legacy engine, tasks, store and API | #126 | `d75ded1` | 2026-10-25 |
| U4-19 | build: shrink the base install to the core dependency set | #130 | `5c4b0bd` | 2026-10-25 |
| U5-00 | docs: plan Phase 5 against the repository | #132 | `b93ff03` | 2026-10-25 |
| U5-10 | fix(data): refuse column names that Polars reads as patterns | #133 | `24c33cd` | 2026-10-25 |
| U5-01 | feat(policy): load and evaluate user policies | #134 | `b68ae76` | 2026-10-26 |
| U5-02 | feat(model): add the findings query API and notebook display | #135 | `502dbec` | 2026-10-26 |
| U5-03 | feat(sinks): write and read a run as JSON and NDJSON | #136 | `aa0c552` | 2026-10-26 |
| U5-04 | feat(sinks): write a run as Parquet tables | #137 | `569b60b` | 2026-10-26 |
| U5-05 | feat(sinks): write a run as a self-contained HTML report | #138 | `385a80f` | 2026-10-26 |
| U5-06 | feat(cli): add check, layered settings, policy exit codes and a generated reference | #139 | `92810b4` | 2026-10-26 |
| U5-07 | feat(engine): add per-check timeouts, cache settings and cache eviction | #140 | `9c0e862` | 2026-10-26 |
| U5-11 | fix(checks): close the calibration gaps in outlier, bimodality, stationarity and duplicate checks | #141 | `d58077c` | 2026-10-26 |
| U5-08 | docs: add guides, a generated check reference and an example workflow for embedded use | #142 | `06c4148` | 2026-10-26 |
| U5-09 | build: add the wheel smoke install, a Windows leg, coverage and a release workflow for 1.0.0a1 | #143 | `28f330c` | 2026-10-26 |
| U6-00 | docs: plan Phase 6 against the repository | #147 | `a03c940` | 2026-10-27 |
| U6-01 | fix(checks): stop one-character deletion blocking from flooding fuzzy_duplicates with code-space pairs | #148 | `30e0208` | 2026-10-27 |
| U6-02 | feat(engine): let a check attach bulky evidence as artifacts the run, cache and files carry | #149 | `b0066c9` | 2026-10-27 |
| U6-03 | feat(engine): record a dataset key, the columns each check examined and the codes it can raise | #150 | `f41e377` | 2026-10-27 |
| U6-04 | feat(store): keep runs, findings and their history in a SQLite repository | #151 | `b424a1c` | 2026-10-27 |
| U6-05 | feat(store): keep each run's artifacts and a Parquet snapshot of the analysed data in an artifact store | #152 | `e2f978a` | 2026-10-28 |
| U6-06 | feat(store): keep runs in the store from the CLI, compare any two runs and manage them with `dsbf runs` | #153 | `c753135` | 2026-10-28 |
| U6-07 | feat(service): serve stored runs, findings, profiles, artifacts and comparisons over a read-only local API with `dsbf dashboard` | #154 | `0b0ce1b` | 2026-10-28 |
| U6-08 | feat(service): run profiling jobs from the local API, queued in the store with progress events and cancel | #156 | `a89d36a` | 2026-10-28 |
| U6-09 | feat(frontend): rebuild the dashboard shell in TypeScript on a client generated from the OpenAPI document | #157 | `15a6ac1` | 2026-10-28 |
| U6-10 | feat(frontend): show a run's overview with its health, policy and findings in the dashboard, tested in a browser | #160 | `2dfdbf3` | 2026-10-28 |
| U6-11 | feat(frontend): explore a run's findings with filters, sort, evidence and paged artifacts in the dashboard | #163 | `f75b769` | 2026-10-28 |
| U6-12 | feat(frontend): browse a run's columns and see each one's profile, histogram, Q-Q plot, findings and sample in the dashboard | #164 | `95dc699` | 2026-10-29 |
| U6-13 | feat(frontend): explore a run's pairwise relationships with a statistic-aware matrix and a pair plot from the data snapshot | #165 | `8c169d7` | 2026-10-29 |
| U6-14 | feat(frontend): show a run's quality dimensions, ML-readiness gate and per-column coverage in the dashboard | #166 | `fa0c5cd` | 2026-10-29 |
| U6-15 | feat(frontend): show a run's stationarity, ACF and PACF and STL decomposition in the dashboard, with an empty state for runs without a time index | #167 | `f0b441a` | 2026-10-29 |
| U6-16 | feat(frontend): compare any two runs and follow a dataset's lineage and finding history in the dashboard | #168 | `71d77cd` | 2026-10-29 |
| U6-17 | feat(frontend): edit a dataset's saved settings in a generated form and rerun it from the dashboard | #169 | `02a0ad6` | 2026-10-29 |
| U6-18 | feat(schema): record a run's settings, raise the run and table schemas to 1.0.0 and gate the run comparison schema | #170 | `7dba424` | 2026-10-29 |
| U6-19 | feat(dashboard): bundle the dashboard in the wheel and gate coverage | this PR (filled in by the next unit) | | 2026-10-29 |

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

---

## Phase 4: Migrate checks

- **Capability:** Every analysis DSBF performs is a typed check in the `core` pack. It reads the profile or the value cache and declares its parameters, output, cost and extras. The legacy engine, tasks, store, API v1 and backend plotting are gone. A stopgap `dsbf profile` runs the new engine, and the base install holds only the core dependencies.
- **Motivation:** Findings with evidence and statistical fixes are the product (REFACTOR_PLAN §2, §10). Column selection by construction removes F-11. Heavy checks with budgets meet the performance targets (§11). Deleting the legacy stack ends the two-system period that Phase 3 opened.
- **Boundaries:** No policy evaluation, no sinks beyond the stopgap CLI's `run.json`, and no CLI v2 (all Phase 5). No store, API v2 or dashboard (Phase 6). The frontend source stays in the repo unchanged; with API v1 deleted, the dashboard does not run from U4-18 until Phase 6 (decided by the user on 2026-10-19). No new data sources (Phase 7).
- **Sequence and dependencies:** Depends on Phase 3. Foundations (U4-01 to U4-03), then rule checks (U4-04, U4-05), analysis checks (U4-06 to U4-09), heavy checks (U4-10 to U4-12), ML-prep, target-aware and synthesis checks (U4-13 to U4-15), calibration (U4-16), and the cutover (U4-17 to U4-19).
- **Success criteria:**
  - No legacy task remains; `dsbf.legacy`, `dsbf.eda`, `dsbf.core`, `dsbf.storage` and `dsbf.api` are deleted.
  - On the CI runner, the standard plan takes under 5 s on titanic, and under 60 s with under 1 GB peak on kitchen50k (52.5k × 60).
  - Calibration baselines (false-positive and false-negative rates) are recorded for each core check version.
  - Every core check runs on the adversarial frame, and the golden findings snapshots cover every core check.
  - `pip install dsbf` without extras runs `dsbf profile`, and each check that needs an extra reports `unavailable`.

**Done (2026-10-25), 20 units.** Detailed unit entries were condensed here at close-out; the per-unit notes that later units depend on were carried into those units as "Note from U4-xx" lines. Three other PRs merged during the phase and are not units: #112 (cancel superseded pull request runs), #122 (report transform skewness to four decimals and drop the perf pin) and #124 (a limit a worker process can meet in the stalling-run timeout test).

- **Capability delivered:** the `core` pack holds 47 checks: 21 rule checks that read the profile, 23 analysis checks that read values, and 3 synthesis checks. Every one has an `applies_to`, a bounded `CheckParams` model, a typed output, a cost, its extras and the adversarial-frame test. Every finding states `n_used` and the method; interval and effect-size estimates, multiple-comparison adjustment (`fdr_bh`, with `pairs_tested` or `tests_run` next to the number flagged) and shuffled-null floors replace the legacy fixed thresholds; advice is attached after a check returns from declarative `{field, op, value}` tips (no `eval`). Beyond the migrations: fuzzy duplicates and the time-series checks were redesigned with budgets, `reference_drift` compares a dataset or a stored `Run`, `target_leakage` and `mutual_information` use one `target` setting, and `quality_dimensions` and `ml_readiness` score from findings, with a coverage per input. `tests/calibration/` plants flaws in 70 seeded scenarios and commits each check's detection and false-alarm rate. `dsbf profile` runs `dsbf.profile()` and writes `run.json`. The legacy engine, its 65 tasks, the store, API v1, backend plotting and the legacy config are deleted (about 290 files and 148 000 lines), and the base install is ten packages.
- **Success criteria, measured:**
  - No legacy task remains: `dsbf.legacy`, `dsbf.eda`, `dsbf.core`, `dsbf.storage` and `dsbf.api` are gone and nothing in `src` or `tests` imports them (U4-18). `snapshots/core/` is byte-identical under two hash seeds, which shows the core pack never read a legacy task.
  - Performance on the CI runner, from the nightly on `main` of 2026-10-25 (`a2bda9e`): `standard-titanic` 2.1 s (21.3 s at U4-01 with the legacy tasks) and `standard-kitchen50k` 10.0 s with a 809 MB peak (266.7 s and 2 318 MB at U4-01), inside 5 s and 60 s with 1 GB. U4-19 enforces both budgets in `tests/perf`, with or without a baseline; the PR's labelled perf run is the first measurement made under the enforcing code.
  - Calibration baselines: `tests/calibration/baseline.json` holds the detection rate, false-alarm rate (Wilson intervals) and probe means of every core check and its version, over 70 scenarios; all 70 give the same rates on scikit-learn 1.9.1, scipy 1.18.1 and statsmodels 0.15.0 as on the versions it was recorded with.
  - Adversarial frame and golden coverage: the Hypothesis test and the fixed adversarial frame run every core check; 10 golden datasets (11 tests) compare each core check's findings with `snapshots/core/`.
  - Lean install: a wheel built from the final tree installs 21 packages and 533 MB in a clean venv (the base of `main` was 54 packages and 750 MB; with `ml,timeseries,text,excel,pandas` it is 38 packages and 691 MB). `dsbf profile` on titanic from it gives 32 ok, 8 not applicable and 6 unavailable (`autocorrelation`, `seasonal_decomposition`, `stationarity` need `timeseries`; `bimodality`, `multivariate_outliers`, `mutual_information` need `ml`), each with its install line, and `fuzzy_duplicates` (`text`) is unavailable at the `deep` depth. `run.json` validates against the schema; with the five extras the run gives 34 ok and 12 not applicable.
- **Key decisions:** core checks run beside the legacy tasks they supersede, each compared with its task on the ten golden datasets through a harness that accepts a difference only with a reason; a statistical fix is an accepted difference whose reason says why the new result is right, and U4-18 deleted the harness; the user chose on 2026-10-19 to delete the legacy stack, API v1 and the old store in this phase, with a stopgap CLI, and to retire the 13 profile-restating summaries; a check that needs an extra is `unavailable` per check, and it imports the extra inside its function ([ADR 0009](adr/0009-single-distribution-with-extras.md)); the executor imports the installed extras on the calling thread before checks run side by side, because a first import on a worker thread deadlocked kitchen5k in 4 of 4 runs; the four Python caps became next-major ceilings, `scikit-learn` has a floor of 1.8 and `numpy` of 2.3 (the tested range); a check with nothing to examine reports `not_applicable` where the legacy task reported `skipped`, and `unavailable` is for a missing extra.
- **Deviations from the phase block:**
  - The dashboard does not run from U4-18 until Phase 6 (the user's decision, recorded in the phase block). `tests/validation/DASHBOARD_CHECKLIST.md` and the frontend source were kept for Phase 6.
  - The performance criterion was met by the nightly run before U4-19 enforced it; the enforcing code has run only locally (3.4 s and 299 MB on titanic, 18.9 s and 682 MB on kitchen50k on a machine slower than the runner) until the PR's perf run.
  - Several gaps are recorded, not closed: `fuzzy_duplicates` unaided cross-system matches and single-character typos, `bimodality` below 300 values, `stationarity` on random walks, and the skewed-null false alarms of `multivariate_outliers` (U5-00 decides). A column named `*` or `^a$` still breaks the profiler (U5-00).
  - `fuzzy_duplicates` runs at the `deep` depth only, so the standard depth no longer reports fuzzy duplicates, and the `text` extra shows as unavailable only there.
  - Schema: `SCHEMA_VERSION` went from 0.6.0 to 0.10.0 over the phase (notes with a count, profile fields for rule checks, datetime fields, and others), all additive before the 1.x guarantee.
  - U4-19 lifted the caps and moved two golden snapshots: `wide` (three `normality` p-values in the sixth significant digit, scipy 1.18) and `kitchen5k` (`multivariate_outliers` scores, scikit-learn 1.8 and later).
- **PRs:** #100 (U4-00, `0ef2161`), #101 (U4-01, `91c064d`), #102 (U4-02, `aa2703a`), #103 (U4-03, `97c9d6d`), #104 (U4-04, `56a24d4`), #105 (U4-05, `9f5c5ea`), #108 (U4-06, `63c2bbe`), #109 (U4-07, `a6641b7`), #110 (U4-08, `5bb5a65`), #111 (U4-09, `fb07cdb`), #115 (U4-10, `05de0dd`), #116 (U4-11, `cddf99d`), #117 (U4-12, `70fe926`), #118 (U4-13, `2564726`), #119 (U4-14, `7df24ea`), #121 (U4-15, `9fe8873`), #123 (U4-16, `111bd9c`), #125 (U4-17, `73b361a`), #126 (U4-18, `d75ded1`), #130 (U4-19, `5c4b0bd`).

---

## Phase 5: Policy, sinks, CLI v2. First release for embedded use

- **Capability:** User policies over findings (report-only by default); JSON, NDJSON, Parquet and static HTML sinks; a findings query API and notebook display; CLI v2 (`profile`, `check`, `plan`, `schema export`, the health summary on every run); docs for embedded use; release engineering; `1.0.0a1` on PyPI.
- **Motivation:** Mode 1 (embedded) needs structured outputs, exit codes that only a user's policy sets, and a published package (REFACTOR_PLAN §1, §7, §8.1, §9).
- **Boundaries:** No store, API or dashboard (Phase 6). No new sources (Phase 7). Schema compatibility guarantees start with this release (§4.3).
- **Sequence and dependencies:** Depends on Phase 4. Policy, then the findings API, then sinks (JSON, Parquet, HTML), then CLI v2 and execution settings, then docs, then the release.
- **Success criteria:**
  - A clean-venv install from a temporary directory passes as a CI job on every PR.
  - An example GitHub Actions workflow writes Parquet findings.
  - Policy exit codes are documented and tested.
  - `1.0.0a1` is on PyPI.

**Done (2026-10-26), 12 units.** Detailed unit entries were condensed here at close-out; the notes later units depend on were carried into them as "Note from U5-xx" lines. Three other PRs are not units: #144 (TestPyPI before PyPI in the release workflow, `921b1a2`), #145 (recording the release, `d63d240`) and #146 (time for the Windows worker leg to start a second spawned check, `6c25974`). The date commit `ff69f86` went to `main` without a PR.

- **Capability delivered:** `dsbf.profile(policy=, previous=)` evaluates a user policy (report-only unless a rule says `fail`) into `Run.policy`. A `Run` has a findings query view and a notebook display, and `Run.write` and `Run.read` use a name-keyed sink registry: `json`, `ndjson`, three Parquet tables with a versioned column listing, and a self-contained HTML report (78 KB for titanic, no network). `dsbf profile` and `dsbf check` take layered settings (`--config`, `--set`, environment), a policy, a previous run, per-check timeouts, workers and a result cache with eviction; `check` exits 1 when a policy rule fails the run and 2 on a setup error. `dsbf docs export` generates the CLI, settings and 47-check references. Column names Polars reads as patterns (`*`, `^…$`) are refused at load, and four checks' calibration gaps are fixed or stated as limits. Release engineering: a wheel smoke install job on every PR, a Windows leg, coverage in the full suite, a tag-triggered release workflow with trusted publishing, a hand-kept changelog, and the version `1.0.0a1`.
- **Success criteria, measured:**
  - Clean-venv install from a temporary directory as a CI job on every PR: the `package` job builds the wheel, installs it with no extras and no lock and runs `dsbf check` with a passing policy (exit 0) and a failing one (exit 1). Run locally on the built wheel (5 × 3 CSV, 46 checks, 6 `unavailable` without the `ml` and `timeseries` extras); the CI run is the PR's.
  - Example GitHub Actions workflow writes Parquet findings: `examples/github-actions/dsbf-check.yml`, whose check step a test runs with a failing and a passing policy (U5-08).
  - Policy exit codes are documented and tested (U5-06, `docs/guides/policy.md`).
  - `1.0.0a1` on PyPI: met on 2026-10-27, after the PR merged. The user registered the publishers and pushed the tag; #144 first made the workflow publish to TestPyPI and install from it before PyPI. A clean-venv install from PyPI ran `dsbf check` (exit 0, and 1 with a failing policy).
- **Key decisions:** a policy rule reads a finding's evidence metric by name and a first run with `new_since: previous_run` matches nothing and adds a note; sinks are a registry that the CLI and `Run.write` share; the HTML report draws only the health, findings and per-column profiles and leaves quality, ML readiness and relationships to the Phase 6 dashboard; the cache is off by default; the Parquet and check schemas are guarded by their own versions, and U5-09 added the policy schema (breaking changes need `Policy.version` raised; additive ones need no bump); the PyPI name `dsbf` with trusted publishing from a tag workflow behind a `pypi` environment; no alias for unsafe column names, a refusal instead; the changelog is hand-kept and the git-chglog workflow is gone.
- **Deviations from the phase block:**
  - `1.0.0a1` was published after the unit's PR merged, with TestPyPI added as a first step in #144.
  - Two units were added after U5-00: U5-10 (safe column names) and U5-11 (calibration fixes).
  - The run, tables and check schema versions did not move to 1.0.0 with the alpha; the compatibility guarantee starts with the stable `1.0.0`.
  - The Windows leg, `twine check`, the release workflow and the coverage step ran for the first time on the U5-09 PR; the perf baseline predates U5-11 and is promoted from the 2026-10-27 nightly (carried to U6-00).
  - The HTML report omits the policy outcome (the CLI prints it); the dashboard or report shows it in Phase 6.
- **PRs:** #132 (U5-00, `b93ff03`), #133 (U5-10, `24c33cd`), #134 (U5-01, `b68ae76`), #135 (U5-02, `502dbec`), #136 (U5-03, `aa0c552`), #137 (U5-04, `569b60b`), #138 (U5-05, `385a80f`), #139 (U5-06, `92810b4`), #140 (U5-07, `9c0e862`), #141 (U5-11, `d58077c`), #142 (U5-08, `06c4148`), #143 (U5-09, `28f330c`).

---

## Phase 6: Store, service and dashboard v2. Standalone local

- **Capability:** Artifacts for bulky evidence, a run store (SQLite with Alembic) and an artifact store (fsspec) with a snapshot of the data as profiled, `dsbf runs ...`, API v2 with a local job runner, a generated TypeScript client, and the dashboard rebuilt in TypeScript on API v2 and served by `dsbf dashboard` from the wheel; stable 1.0 schemas and release `1.0.0`.
- **Motivation:** Mode 2 (standalone, local): DSBF is the pipeline, with history and a bundled UI (REFACTOR_PLAN §8.2–§8.4).
- **Boundaries:** Local and single-user only: no Postgres, auth or queue (Phase 8). API v1 and the old store were deleted in U4-18, so this phase builds new and migrates nothing. Drift from stored profiles and recurring-dataset views stay in Phase 7 (T4); this phase keeps the history of a finding across runs. The result schemas become stable at `1.0.0`, so the contract changes the store and dashboard need land before it (U6-02, U6-03, U6-18).
- **Sequence and dependencies:** Depends on Phase 5. The fuzzy-duplicate cost first, so later units measure against a green perf baseline; then the contract changes, the store, `dsbf runs`, the read API, jobs, the TypeScript client, the dashboard shell and its tabs one by one, the stable schemas, and the bundled release.
- **Success criteria:**
  - `pip install "dsbf[server]" && dsbf check data.csv --sink store && dsbf dashboard` works on a clean machine, as a CI job.
  - Opening a run transfers under 200 KB (titanic, measured by the Playwright test).
  - Every published schema is at 1.0.0, guarded by its breaking-change gate.
  - The perf, golden and calibration nightlies are green on the release commit, and `1.0.0` is on PyPI.
- **Decided in U6-00 (2026-10-27), for the whole phase:**
  - Comparison is explicit: `dsbf runs diff`, the compare endpoint and the compare view take any two runs, even of different datasets (train against test, staging against production). The dashboard pre-fills the previous run of the same dataset, nothing more. (Asked the user.)
  - Dataset key, which groups runs into a history: an explicit `dataset` (`--dataset`, `dataset=`) wins; otherwise the canonical source location (the absolute path of a file; a connection and table from Phase 7). An in-memory frame with no key is stored ungrouped: comparable, with no history. Every run records its name, location and fingerprint, so a change of source shows in the lineage. The pipeline guide tells pipelines to set the key.
  - The dashboard is a new TypeScript app that replaces the old one (asked the user): Vue 3, Pinia, vue-router 5, ECharts only. Tests grow with the tabs (Vitest component tests and a Playwright test per tab against a fixture store), and routes are split from the shell, so there is no separate frontend-test or code-split unit.
  - The store keeps a snapshot of the data as profiled, on by default and capped, with a setting to turn it off (asked the user).
  - Artifacts on fsspec as ADR 0008 says, local in this phase.
  - Jobs run on one worker thread in the server, persisted in the store, with progress read by polling.
  - The service binds `127.0.0.1`, rejects a `Host` that is not local, takes POSTs only as JSON, has no CORS unless configured, and has no auth (Phase 8).
  - One release, `1.0.0`, at the end.

**Done (2026-10-29), 20 units.** Detailed unit entries were condensed here at close-out; the notes later units depend on were carried into them. Two success criteria were not met in this program: the review on 2026-10-29 found the dashboard unusable as a visual exploration, so Phase 6b (dashboard design) was added and the `1.0.0` release moved to its end.

- **Capability delivered:** a check can attach bulky evidence as artifacts, which ride on the `Run`, the cache and the `json` sink. `--sink store` keeps runs in a SQLite repository (Alembic revisions 0001–0004: datasets, runs, profiles, check results, findings, artifacts, jobs and per-dataset settings) with an fsspec artifact store and a capped Parquet snapshot of the data as profiled. `dsbf runs` (`list`, `show`, `diff`, `delete`, `gc`, `import`) and `dsbf.compare` read it. `dsbf.service` is a local FastAPI app (`/v2`, `schema/openapi.json`) with read endpoints, a job runner for profiling and rerunning, and a local-only security model. The dashboard is a TypeScript Vue 3 app on a generated client, served with the API by `dsbf dashboard` from the wheel: datasets with lineage and finding history, a run overview with the health panel, findings, columns, relationships, quality and ML readiness, time series, comparison of any two runs, and a settings form with a rerun. The run and Parquet table schemas are at 1.0.0, `RunDiff` joined the breaking-change gates, and the compatibility rules are in `docs/guides/compatibility.md`. The dashboard is a first version with a page per area, mostly tables and cards; the user judged it unusable as a visual exploration (see the deviations).
- **Success criteria, measured:**
  - `pip install "dsbf[server]" && dsbf check data.csv --sink store && dsbf dashboard` as a CI job: the `package` job and the Windows leg build the frontend into the wheel, install it with `[server]` in a clean venv, run `dsbf check --sink store`, start `dsbf dashboard` and fetch `/`, a run summary and a page-route fallback (`.github/scripts/smoke_install.sh`). Run locally on Linux on the built wheel: passed. The release workflow builds the dashboard into the wheel and checks that it is in the TestPyPI install.
  - Opening a run transfers under 200 KB: 151.4 KB for titanic in the Playwright test (U6-10 measured 146 KB).
  - Every published schema at 1.0.0 and gated: the run, tables and `RunDiff` (U6-18), the policy (version 1), each check's own 1.x; OpenAPI stays on its 1.x line (1.8.0) because its gate refuses a version that goes backwards.
  - Nightlies green on the release commit and `1.0.0` on PyPI: **not met in this program.** The release moved to the end of Phase 6b, so the version stays `1.0.0a1` and the changelog's section is `[Unreleased]`.
- **Key decisions:** comparison is explicit and takes any two runs, with the previous run of a dataset only pre-filled (asked the user); the dataset key is an explicit `dataset` or else the file's absolute path; the dashboard is a new TypeScript app that replaced the old one (asked the user); the data snapshot is on by default and capped (asked the user); artifacts on fsspec as in ADR 0008; jobs on one worker thread persisted in the store; the service binds local addresses only, checks `Host`, takes JSON writes only and has no auth; the frontend is built in CI and the release workflow and never committed, and a checkout's `frontend/dist` wins over the packaged copy; the coverage floor is 95 (measured 96%).
- **Deviations from the phase block:**
  - The phase block's acceptance for the dashboard was functional (the right data, empty states, a test and a size budget per page), with no visual acceptance. Reviewing the finished dashboard on 2026-10-29, the user found it "entirely unusable": tables, counts and cards, "more like a knowledge base page" than a visual exploration of data, and less than the old dashboard. The user chose to merge U6-19 without tagging and to add Phase 6b (dashboard design and the `1.0.0` release); U6-19 therefore did not release.
  - Three units were added after U6-00: the fuzzy-duplicate regression (U6-01), the contract fields the store needs (U6-03) and the stable schemas (U6-18). Several tab units added the API endpoints their tab needed (columns, relationships, coverage, lineage, settings) beyond U6-07's.
  - Cancelling a job needed an engine change (`execute(cancel=)`), and the health panel moved `not_applicable` checks into `RunHealth.issues` (run schema 0.13.0).
  - OpenAPI is not at 1.0.0.
  - The archetype fixture stores for the dashboard checklist (`severe_multicollinearity`, `wide`, `all_categorical`, `single_column`; quality and readiness for the other archetypes) were not built, and those lines are not Playwright tests.
  - `RunDiff` reports count-scaled statistics as changed when two runs differ in sample size; the user deferred it (a documented 1.x limitation, carried to U7-00).
  - `gc` leaves a dataset's settings overlay behind when it deletes the dataset's last run (carried to U7-00).
- **PRs:** #147 (U6-00, `a03c940`), #148 (U6-01, `30e0208`), #149 (U6-02, `b0066c9`), #150 (U6-03, `f41e377`), #151 (U6-04, `b424a1c`), #152 (U6-05, `e2f978a`), #153 (U6-06, `c753135`), #154 (U6-07, `0b0ce1b`), #156 (U6-08, `a89d36a`), #157 (U6-09, `15a6ac1`), #160 (U6-10, `2dfdbf3`), #163 (U6-11, `f75b769`), #164 (U6-12, `95dc699`), #165 (U6-13, `8c169d7`), #166 (U6-14, `fa0c5cd`), #167 (U6-15, `f0b441a`), #168 (U6-16, `71d77cd`), #169 (U6-17, `02a0ad6`), #170 (U6-18, `7dba424`), and U6-19 (this PR, filled in by the next unit).
