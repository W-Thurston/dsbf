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

Phase 6 is broken down in full (U6-00). Phases 7–9 have a locked phase block and an ordered list of
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

### U6-00 · Plan Phase 6
- **Status:** done (2026-10-27, #147, `a03c940`) · **Size:** S · **Depends on:** the Phase 5 units
- **Note from U5-08:** a `codes` field on the check spec would make the check reference's finding codes exact (scanned from source today). **Note from U5-06:** the HTML report omits `Run.policy` (the CLI prints the verdict); show it in the dashboard or report. **Note from U5-09:** (1) promote the perf baseline from the 2026-10-27 scheduled Performance run on `main` (the committed one predates U5-11's calibration changes); (2) re-check that `dsbf.interfaces.reference` still duck-types Typer commands on a Typer bump (0.27 bundles click as `typer._click`); (3) the `sql` and `server` extras still need an importer and an install-matrix leg together; (4) a coverage floor is undecided, the first number is in the `full` job's step summary.
  - **Carry in from U4-00:** the dashboard does not run from U4-18 until this phase's API v2 and shell land (decided 2026-10-19). F-15 (dataset identity, the snapshot of the data as profiled) and F-25 (pagination, error mapping, migrations, garbage collection, encoded path parameters) are design inputs for the store and API, not fixes to old code.
  - **Carry in from U4-18:** `tests/validation/DASHBOARD_CHECKLIST.md` still describes the old dashboard's browser checks; rewrite it for the rebuilt one. `.github/ISSUE_TEMPLATE/dashboard_ui.yml` and the `frontend/` source were kept.
  - **Carry in from U1-11:** (1) the dashboard is JavaScript, so it can use `frontend/src/generated/run.d.ts` only through JSDoc `import()` types or after a move to TypeScript; (2) a TypeScript consumer needs its own rank table for `Severity` and `Confidence`, which the schema cannot carry; (3) revisit the `vue-router <5` and `plotly.js-dist-min <4` caps in the held-back table.
  - **Carry in from U4-07 and U4-11:** `associations` returns the tested pairs and no correlation matrix, so the column view computes a matrix on demand; `seasonal_decomposition` keeps strengths, not the component series, so the store keeps those arrays.
  - **Carry in from U4-19:** the `sql` and `server` extras have no install-matrix leg or `unavailable` test, since no code imports them; add both with the first code that does. `--perf-cprofile` went with the legacy perf scenarios; add one to `tests/perf/measure_engine.py` only if a check-level profile is wanted.
  - **Carry in from U1-03 and U0-16:** clear the 20 frontend lint warnings and raise `no-unused-vars`, `vue/no-side-effects-in-computed-properties` and `vue/require-default-prop` to `error` (16 dead variables in 12 files; `acfSelectedColData` in `TimeSeriesTab.vue` assigns inside a `computed`; route props in `RunDetailView.vue` and `RunListView.vue`); revisit `vue/no-v-html` if another source appears. The light theme draws chart titles and card headings such as "Value Counts" almost white on a light background (same on ECharts 5).
- **Outcome:** Phase 5 is confirmed closed, with its records corrected (#145 and #146, three finding statuses, a stale branch reference). The eighteen placeholders became U6-01 to U6-19: three units were added (the fuzzy-duplicate regression found in the audit, the contract fields the store needs, the stable 1.0 schemas), and the frontend-test and code-split placeholders were folded into the dashboard units. The user chose the regression unit, a TypeScript replacement dashboard, a snapshot on by default, and explicit comparison with a dataset key.
- **Goal:** Confirm Phase 5 is closed, turn the eighteen placeholders into units with assumptions and steps, decide the questions the notes and carry-ins leave open, and ask the user about the regression, the dashboard rebuild, the data snapshot and dataset identity.

### U6-01 · Fuzzy-duplicate cost and the perf baseline
- **Status:** done (2026-10-27, #148, `30e0208`) · **Size:** S · **Depends on:** U6-00
- **Note from U6-00:** the 2026-10-27 nightly (run `40480178400`, at `06c4148`) failed. `fuzzy_duplicates` on kitchen5k took 9.35 s against a 0.92 s baseline, and 1.58 s at `9c0e862` against 9.24 s on `main` when both were run on one machine, so the cost came with U5-11 (`d58077c`): leave-one-column-out scoring and one-character deletion blocking. Every other scenario, and import time, rose about 1.5× on that run, which points at the runner rather than the code. The committed baseline predates U5-11.
- **Outcome:** The cost was the one-character-deletion blocking, not the scoring: sequential ids and equal-width hashes made 746,476 candidate pairs on kitchen5k. A deletion key now blocks only for 2–4 distinct tokens, only tokens of 3–64 characters are expanded, and the variants are built in Polars. 8.5 s → 0.66 s in-process (0.30 s before U5-11); 1.27× the pre-U5-11 time through the perf harness on one machine, recorded as the budget (1.3×). `fuzzy_duplicates` is 1.2.1; no calibration rate moved, and the golden `sms_spam` snapshot moved only in `pairs_compared`. A typo in a token over 64 characters is no longer blocked with its source. The perf baseline is promoted from this PR's `perf` run (`9bb9d90`), whose runner was about 1.5× slower than the one that recorded the old baseline on every scenario, including ones this change does not touch (titanic, `profile-wide1m`); with that drift removed, kitchen5k's `fuzzy_duplicates` is about 1.2× the old baseline (1.84 s against 0.92 s). A nightly on a quiet runner will therefore pass with room to spare. numpy 2.5 and polars 2.1 (#131) are not taken here.
- **Goal:** `fuzzy_duplicates` on kitchen5k costs within 25% of its pre-U5-11 time, or within a budget this unit justifies and records, and finds the same pairs on U5-11's calibration scenarios. The nightly is green against a baseline promoted from a CI run of the fixed code.
- **Assumptions to confirm:** (1) The cost is in the leave-one-column-out scores or the deletion variants; profile both with `cProfile` on kitchen5k before changing anything. (2) The `standard-*` scenarios do not run `fuzzy_duplicates` (it is heavy and runs at the deep depth only), so the standard budgets are not at stake. (3) The perf workflow runs on a PR with its label, so the baseline can be promoted from the PR's own CI run rather than a nightly after merge. (4) Whether the other scenarios are back near the baseline on a quiet runner, or the runner drift needs recording when the baseline is promoted.
- **Steps:** profile the check; cut the cost where the profile points (for example score a left-out column only for pairs above a lower bound, reuse each column's scores across the left-out variants, cap the deletion variants per token); a unit test that the U5-11 scenarios flag the same pairs; `-m calibration` and `-m golden`; label the PR for perf and promote `tests/perf/baseline.json` from that run with the harness. Then take numpy 2.5 and polars 2.1 (Dependabot #131) on top, measured separately: take them if the golden snapshots don't move, or make them their own explained step.
- **Acceptance:** On the CI runner, kitchen5k's `fuzzy_duplicates` is within the recorded budget. Its calibration rates are no worse than U5-11's baseline, and the baseline is promoted from a green CI run of the fixed code.

### U6-02 · Artifacts for bulky evidence
- **Status:** done (2026-10-27, #149, `b0066c9`) · **Size:** M · **Depends on:** U6-01 · **Resolves:** REFACTOR_PLAN §8.1 (bulky evidence behind `details_ref`)
- **Carry in from U4-10 and U4-11:** `fuzzy_duplicates` keeps only its 20 closest pairs and the counts in its output; keep the full flagged-pair list. `seasonal_decomposition` keeps the trend and seasonal strengths; keep the observed, trend, seasonal and remainder series.
- **Outcome:** `ctx.attach` returns a `<check_id>/<name>` ref; `CheckResult.artifacts` lists refs; payloads (Arrow table or JSON value) ride privately on the `Run`, the worker pickle and the result cache (a file beside each entry; a missing one is a miss), and the `json` sink writes them under `artifacts/` where `Run.read` finds them lazily. A `details_ref` must name an artifact of its run. An `error` result's traceback is the `<check_id>/traceback` artifact and `Note.detail` keeps its last line. `fuzzy_duplicates` 1.3.0 attaches `flagged_pairs`, `seasonal_decomposition` 1.2.0 attaches `decomposition` (one long table); run schema 0.11.0. `CheckResult` and `Run` compare fields only, since pydantic equality would compare the payloads. **Deviation:** kitchen5k flags no fuzzy pairs, so the pair-list acceptance was checked on sms_spam (90 pairs); kitchen5k carries the decomposition.
- **Goal:** A check can attach bulky data to its result without doing I/O. `ctx.attach(name, payload, description=)` returns a ref; `CheckResult.artifacts` lists each artifact's ref, media type, rows and bytes; `Evidence.details_ref` points at one. The payloads ride on the in-memory `Run`, not in its JSON, survive process workers and the result cache, and the `json` sink writes them beside `run.json` (`artifacts/<check_id>/<name>.parquet` or `.json`), where `Run.read` finds them.
- **Decided in U6-00:** (1) A payload is an Arrow table (written as Parquet) or a JSON-safe value (written as JSON), nothing else. (2) The ref is `<check_id>/<name>`, unique within a run, so the store and API address it without the check's code. (3) `Run.artifact(ref)` returns the payload when it is in memory or beside the file read, and says which refs it lacks otherwise. (4) An `error` result's full traceback becomes an artifact; `Note.detail` keeps its summary. (5) The run schema takes a minor bump.
- **Assumptions to confirm:** `details_ref` is unused (audited 2026-10-27: only the field exists); a process worker returns its `CheckResult` by pickling, so an Arrow table crosses unchanged; the result cache keeps a result as JSON, so a payload needs a file beside it in the cache; the full fuzzy pair list on kitchen5k fits a size this unit records, or a parameter caps it.
- **Steps:** the contract (an `ArtifactRef` on `CheckResult`, and a check that each `details_ref` names an artifact of the run); `CheckContext.attach`; transport through the worker and the cache; the `json` sink and `Run.read`; the fuzzy pair list, the STL series and tracebacks as artifacts; schemas, TypeScript types and golden snapshots regenerated.
- **Acceptance:** A run written with the `json` sink and read back has equal artifacts. A cache hit and a process-isolated check both return theirs intact. Titanic's `run.json` grows by the refs only. On kitchen5k the fuzzy pair list and each decomposed series are artifacts that a `details_ref` points at.

### U6-03 · Contract fields the store needs
- **Status:** done (2026-10-27, #150, `f41e377`) · **Size:** S · **Depends on:** U6-02
- **Outcome:** `DatasetManifest.key` (explicit `dataset=`/`--dataset`/`dataset` setting, else the file's absolute path, else empty), `CheckResult.columns` (eligible columns, narrowed by the new `ctx.examined`), and `@check(codes=...)` on the 44 checks that raise findings; the check reference reads the declarations and the source scanner is now a test cross-check. Run schema 0.12.0; result-cache key scheme 3. **Deviation:** a dataset-level check lists no columns, so the columns acceptance covers checks that read columns and synthesis checks; the undeclared-code test runs in the golden tier and a static scan cross-check runs in the unit tier.
- **Note from U5-08:** a `codes` field on the check spec would make the check reference's finding codes exact (scanned from source today). **Carry in from U4-15:** quality coverage is per input check; showing it per column needs the columns a check examined on `CheckResult`.
- **Goal:** Three additive fields before the 1.0 freeze. `DatasetManifest.key` holds the dataset key (`dsbf.profile(dataset=)`, `--dataset`; derived from the source location when not given, and empty for an in-memory frame without one). `CheckResult.columns` lists the columns a check examined. `@check(codes=...)` declares the finding codes a check can emit, and the check reference reads them.
- **Decided in U6-00:** the key rule is the phase's (above); "examined" means the eligible columns the check analysed after exclusions, filled by the engine from eligibility unless the check reports its own; a finding with an undeclared code fails in tests, not at run time.
- **Assumptions to confirm:** eligibility computes each check's column set (`engine/eligibility.py`); checks that choose columns inside their body can report them; the source scan in `interfaces/check_reference.py` lists every code today, so the declarations can start from its output; `SourceRef.location` holds an absolute path for a file.
- **Steps:** the three fields and their defaults; `dataset=` on `dsbf.profile` and `--dataset` on the CLI; `codes=` on the 47 core checks; a test over the golden runs that every emitted code is declared; the reference generator reads the declarations and the scanner goes; schemas, TypeScript types, references and golden snapshots regenerated.
- **Acceptance:** Every core check declares its codes, and a test fails on an undeclared one. `CheckResult.columns` is filled for every `ok` result in the titanic and kitchen5k golden runs. A file run without `--dataset` carries its absolute path as the key, and a frame without `dataset=` has none. `dsbf docs export --check` passes.

### U6-04 · Run repository on SQLite
- **Status:** done (2026-10-27, #151, `b424a1c`) · **Size:** M · **Depends on:** U6-03 · **Resolves:** F-15 (dataset identity), F-25 (migrations)
- **Outcome:** `dsbf.store` (`server` extra): `RunRepository` protocol, `SqliteRunRepository` on SQLAlchemy Core with Alembic revision `0001` (tables `datasets`, `runs`, `profiles`, `check_results`, `findings`, `finding_columns`), `Page`, `RunSummary` and `FindingOccurrence` views, `open_repository()` at `<DSBF_HOME>/dsbf.db`. **Deviations:** artifact payloads are files under `<DSBF_HOME>/artifacts/<run_id>/` in the `json` sink's layout until the next unit's fsspec store replaces them; a `finding_columns` table serves the column query; a dataset row outlives its last run. Took #120 and #127; #114 (`virtualenv`) has nothing newer in the lock, so it stays with Dependabot.
- **Note from U5-09 and U4-19:** the `server` extra gets its first importer here, so it gets its install-matrix leg and an `unavailable` test together; the `sql` extra's pair moves to U7-00.
- **Note from U6-02:** the store keeps artifact payloads too (`Run.artifact(ref)` is the read path; `ArtifactRef.rows` and `bytes` describe the stored file), so `get` returns a run whose artifacts read back equal, and `delete` removes them.
- **Goal:** `dsbf.store`, behind the `server` extra: a `RunRepository` protocol and a SQLite implementation on SQLAlchemy Core, its schema built by Alembic migrations applied on open. It holds datasets, runs, check results, findings and profiles. It can save a run, get it back equal, list runs (by dataset, paged), query findings (by run, dimension, severity, code and column, paged), give the history of one finding id or code across a dataset's runs, and delete a run.
- **Decided in U6-00:** (1) Each row keeps its item's JSON beside the columns that are queried, so `get` rebuilds the `Run` without loss (F-20: the old store dropped most of a result) and one finding or result is read without the whole run. (2) The database is `<DSBF_HOME>/dsbf.db` in WAL mode with one writer. (3) The migration scripts ship in the wheel, and the store refuses a database whose revision is newer than the code. (4) A `datasets` row is keyed by `DatasetManifest.key`; ungrouped runs have no dataset row.
- **Assumptions to confirm:** SQLAlchemy 2.1 and Alembic 1.20 work on Python 3.12 and 3.13 from the lock; `uv_build` puts the migration scripts in the wheel; `Finding.id` is stable across runs for the same check, subject and code (`Finding.make_id`); the base install imports nothing from `dsbf.store`.
- **Steps:** tables and indexes (`datasets`, `runs`, `check_results`, `findings`, `profiles`); the Alembic environment and first revision; the repository; the CI install-matrix leg for `server` and the `unavailable` path (importing `dsbf.store` without the extra names `dsbf[server]`); round-trip tests on titanic, kitchen5k and the adversarial frame; paging tests. Take the tooling upgrades from Dependabot (#127 `setup-uv`, #114 `virtualenv`, #120 `filelock`) here, since this unit changes CI.
- **Acceptance:** `get(save(run)) == run` for the golden datasets. A migration test upgrades an empty database to head and refuses a newer revision. Finding queries return the documented order and pages. A base install imports nothing from `dsbf.store`.

### U6-05 · Artifact store and the data snapshot
- **Status:** done (2026-10-27, #152, `e2f978a`) · **Size:** M · **Depends on:** U6-04 · **Resolves:** F-15 (the snapshot of the data as profiled)
- **Outcome:** `dsbf.store.artifacts` (`ArtifactStore` protocol, `FsspecArtifactStore`), Alembic revision `0002` (`artifacts` table with ref, media type, rows, bytes, SHA-256, sampling), repository reads (`artifacts`, `read_artifact`, `read_artifact_rows`, `snapshot`) and `Run.analysed_frame` (memory only, set by the executor). `store.snapshot_rows` (default 10,000, 0 off) caps a seed-0 random snapshot recorded as `SamplingInfo`. kitchen50k's snapshot is 3.5 MB. **Deviations:** `encode` writes Parquet in 10,000-row groups so range reads are cheap; `Run.artifact` still reads from a local directory only (a loader hook for non-local stores belongs with Phase 8's S3 store).
- **Goal:** An `ArtifactStore` on fsspec (`<DSBF_HOME>/artifacts/` in this phase) keeps each stored run's artifacts and a Parquet snapshot of the data as profiled. An `artifacts` table records each one's ref, media type, rows, bytes and checksum. An artifact is read whole or by row range.
- **Decided in U6-00:** (1) The snapshot is on by default: at most 10,000 rows of the analysed data, a seeded random sample above that, recorded as `SamplingInfo` is. The `store.snapshot_rows` setting changes the cap, and 0 turns the snapshot off. (2) The core never writes the snapshot: the `Run` holds its analysed frame in memory (not serialised, not sent to workers), and the store writes the snapshot from it. A run read from `run.json` has no snapshot, and the dashboard says so.
- **Assumptions to confirm:** a row range of a Parquet file reads without loading the whole file (row groups through `pyarrow`; measure on the fuzzy pair list); holding the frame on the `Run` adds no copy; the snapshot of kitchen50k at 10,000 rows is under a size this unit records; fsspec is small enough for the `server` extra (pure Python).
- **Steps:** fsspec in the `server` extra; the protocol and the local implementation; the `artifacts` table (a second Alembic revision); the snapshot writer and its setting; deleting a run removes its files; tests for round trips, range reads, and a missing file reported as missing.
- **Acceptance:** A stored run's artifacts and snapshot read back equal. A range read of the fuzzy pair list returns exactly the rows asked for. Deleting a run leaves no files. The snapshot keeps the cap and says how it was sampled.

### U6-06 · Store sink and `dsbf runs`
- **Status:** done (2026-10-27, #153, `c753135`) · **Size:** M · **Depends on:** U6-05 · **Resolves:** REVIEW_FINDINGS part 2 D (run management), F-25 (delete and garbage collection)
- **Outcome:** `--sink store` and `dsbf.store.save`; `dsbf.compare` (module `dsbf.comparison`, since a module named `compare` would shadow the lazy function) returning `RunDiff`, exported as `schema/diff.schema.json`; `dsbf runs list | show | diff | delete | gc | import` with `--json`; `RunRepository.gc` and `ArtifactStore.run_ids`; `PolicyOutcome.render()` shared by `check`, `runs show` and the HTML report's new Policy section. **Deviations:** `store` is handled beside the sink registry, not in it (a registry sink writes one file to a path); `compare` ignores p-value metrics, which move by orders of magnitude between near-identical runs.
- **Note from U5-06:** the HTML report omits `Run.policy`; show it. **Note from U5-09 (2):** after any Typer bump, re-check that `dsbf.interfaces.reference` still duck-types Typer commands.
- **Goal:** `--sink store` and `dsbf.store.save(run)`; `dsbf runs list | show | diff | delete | gc | import`; `dsbf.compare(base, head)` in the core, which the CLI and the API share; the HTML report and `runs show` print the policy verdict.
- **Decided in U6-00:** (1) A diff takes two run ids or `run.json` paths, any two runs, and matches findings by `Finding.id` into new, resolved and changed (severity, or a metric beyond a tolerance), plus check status changes; it says when the two runs are of different datasets. (2) `gc` keeps the newest N runs per dataset and the runs younger than an age, both settings, and removes orphaned artifact files. (3) `runs import <dir>` stores a run written by the `json` sink, artifacts included, so a CI run can be browsed locally (mode 3).
- **Assumptions to confirm:** a sink that writes to the store rather than a directory fits the U5-03 registry (check `write_run`'s contract); the policy evaluator's `new_since` already matches findings by id, so `compare` can share that logic; `--sink store` without the extra is a setup error (exit 2) naming `dsbf[server]`.
- **Steps:** the sink and `dsbf.store.save`; `dsbf.compare` returning a `RunDiff` contract with an exported schema; the six commands, with `--json` output; the policy section in the HTML report; the CLI reference regenerated; end-to-end tests from a temporary directory.
- **Acceptance:** `dsbf check data.csv --sink store` followed by `dsbf runs list` shows the run. `runs diff` of two titanic runs with a planted change reports the expected new and resolved findings. `gc` keeps exactly the runs it should and leaves no orphaned files. The HTML report shows the policy verdict.

### U6-07 · API v2 read endpoints
- **Status:** done (2026-10-28, #154, `0b0ce1b`) · **Size:** M · **Depends on:** U6-06 · **Resolves:** F-25 (paging, error mapping, encoded path parameters)
- **Outcome:** `dsbf.service` (twelve read operations, problem details, local-`Host` check, optional CORS, `frontend/dist` served after the API), `dsbf dashboard`, `schema/openapi.json` with `--check` and a breaking-change comparison announced by `API_VERSION`, four repository reads, FastAPI 0.162, `httpx2` in the dev group, `starlette` declared in `server`. **Deviations:** artifacts are run-scoped; the findings page defaults to 50 (100 titanic findings are 165 KB); a column also has a `?column=` route, since `..` cannot travel in a path; `dsbf schema export` needs the `server` extra.
- **Note from U6-07:** a slim finding view (without `evidence.details`) would allow larger pages.
- **Goal:** A FastAPI app, `dsbf.service`, with the REFACTOR_PLAN §8.3 read endpoints: datasets, a dataset's runs and the history of a finding, a run summary (manifest, health, policy and counts by dimension and severity, without results), findings filtered and paged, one column's profile, one check's result without artifact payloads, an artifact or the snapshot by row range, and compare. `dsbf dashboard` starts it; until U6-19 bundles the frontend it serves the API, plus `frontend/dist` when that exists, for development. `dsbf schema export` writes the OpenAPI document to `schema/openapi.json`, guarded by `--check` and the breaking-change gate.
- **Decided in U6-00:** (1) Offset and limit pages with a total; the limit is capped at 500. (2) Errors are RFC 9457 problem details; a missing run is a 404 that names its id. (3) The phase's local-service rules (above). (4) Responses are the `dsbf.model` contracts or thin views of them, so the generated client matches `run.d.ts`. (5) Titanic's run summary is under 50 KB and its first findings page under 100 KB, which leaves room for the overview inside 200 KB.
- **Dependency upgrades:** FastAPI is capped `<0.144`, and Dependabot #131 proposes 0.162; take it here.
- **Assumptions to confirm:** FastAPI and uvicorn from the lock accept the frozen pydantic contracts (with `AwareDatetime`) as response models; `app.openapi()` exports without starting a server; a column name with `/`, `#` or `%` round-trips encoded in a path.
- **Steps:** the app factory with the store injected; the endpoints; problem details; the host check; `dsbf dashboard --host --port --no-open`; the OpenAPI export and gate; `TestClient` tests on a fixture store (titanic, kitchen5k, the adversarial names); size tests for the summary and first page.
- **Acceptance:** Every endpoint is tested, including its 404 and 422. Hostile column names round-trip. Titanic's summary is under 50 KB and its first findings page under 100 KB. `dsbf schema export --check` covers `openapi.json`.

### U6-08 · Job API and local runner
- **Status:** done (2026-10-28, #156, `a89d36a`) · **Size:** M · **Depends on:** U6-07
- **Outcome:** `POST /v2/jobs`, `GET /v2/jobs`, `GET /v2/jobs/{id}?after=` and `POST /v2/jobs/{id}/cancel`; a `JobRunner` thread over `jobs` and `job_events` tables (Alembic `0003`), `interrupted` after a restart; `JsonWritesOnly` refuses non-JSON and foreign-`Origin` writes; `API_VERSION` 1.1.0. **Deviations:** cancelling needed an engine change (`execute(cancel=)` raising `RunCancelled`), since a hook cannot stop a run; a running job stops after the check in flight; `POST` does not take `policy` or `previous`.
- **Note from U6-08:** `settings.plugins` lets a job import code from server directories; Phase 8 (auth) should decide who may set it.
- **Goal:** `POST /v2/jobs` profiles a local source with settings (the `ProfileConfig` the CLI builds, plus the dataset key and name) and saves the run to the store. `GET /v2/jobs/{id}` returns the job's state and the progress events after a cursor, `GET /v2/jobs` lists jobs, and a job can be cancelled between checks.
- **Decided in U6-00:** (1) One worker thread in the server runs jobs one at a time; checks keep their own process isolation. (2) Jobs are rows in the store (a `jobs` table and Alembic revision), so after a restart a job that was running shows as `interrupted`. (3) Progress events are the engine's hooks (`CheckStarted`, `CheckFinished`, `NoteRecorded`), numbered; the dashboard polls.
- **Assumptions to confirm:** `profile_with` runs off the main thread (the warning-filter handling in `library.py` changes process-wide state, so check it is safe on a thread); the process pool starts from a non-main thread on Linux, macOS and Windows; cancelling between checks is possible through a hook or the run budget.
- **Steps:** the `jobs` table; the runner; the endpoints; cancel; events; tests for a failing source (the job ends `failed` with the setup error), a restart, and two queued jobs.
- **Acceptance:** A job profiles titanic, and its run appears in the store and under `/v2/runs`. Progress events arrive in order and end with the run id. A bad path ends the job `failed` with the message `dsbf check` would print. A cross-origin form POST is refused.

### U6-09 · TypeScript frontend on a generated client
- **Status:** done (2026-10-28, #157, `15a6ac1`) · **Size:** M · **Depends on:** U6-08 · **Resolves:** the `vue-router <5` and `plotly.js-dist-min <4` caps
- **Outcome:** `frontend/` is a TypeScript Vue 3 app (Vite, `vue-tsc` in `pnpm build`, Pinia, Vue Router 5, ECharts) with a shell, a `HomeView` and a client over `openapi-fetch`; `dsbf schema export --types` also writes `api.d.ts` and `ranks.ts`, covered by `--check`. The old app, Plotly and axios are deleted; `DsbfChart` and `chartTheme` are ported; both caps and their Dependabot entries are gone. **Deviations:** `json-schema-to-typescript` stays for `run.d.ts` (`Run` is not in `openapi.json`, so one generator cannot serve both); TypeScript is held at `~6.0.3` because `typescript-eslint` does not support 7; the dev proxy targets `dsbf dashboard` on :8765; `DsbfChart`'s unused SVG export was removed.
- **Carry in from U1-11:** a TypeScript consumer needs its own rank table for `Severity` and `Confidence`, which the schema cannot carry. **Carry in from U1-03 and U0-16:** the 20 lint warnings, and `no-unused-vars`, `vue/no-side-effects-in-computed-properties` and `vue/require-default-prop` at `error`; revisit `vue/no-v-html` if another source appears.
- **Goal:** `frontend/` becomes a TypeScript Vue 3 app (Vite, `vue-tsc` in `pnpm build`, Pinia, vue-router 5, ECharts only). `dsbf schema export` generates its API client from `schema/openapi.json` (`openapi-typescript` types, used through `openapi-fetch`) and a module of the `Severity` and `Confidence` ranks. The old app, written for API v1, is deleted; `DsbfChart`, `chartTheme` and the card styles are kept, converted to TypeScript, for the tabs to reuse.
- **Decided in U6-00:** TypeScript and a replacement, not a migration (asked the user); the lint rules are at `error` from the first commit, so there are no warnings to clear; Plotly goes with the old app (N-A1).
- **Assumptions to confirm:** `openapi-typescript` and `openapi-fetch` have versions at least two weeks old that fit the lock; one generator can serve both `run.d.ts` and the client (decide whether `json-schema-to-typescript` stays); the new lockfile supersedes the Dependabot npm group (#128).
- **Steps:** delete the old views, tabs, components and Plotly; the TypeScript toolchain; the client and the ranks generated through `dsbf schema export`, with `--check`; port `DsbfChart` and `chartTheme`; a Vitest test of the client against a mocked response; remove both caps, their rows in the held-back table and their `.github/dependabot.yml` entries.
- **Acceptance:** `pnpm lint`, `pnpm build` (with type checking) and `pnpm test:run` pass with no warnings. `dsbf schema export --check` covers the client. The bundle holds no Plotly.

### U6-10 · Dashboard shell and the Run health panel
- **Status:** done (2026-10-28, #160, `2dfdbf3`) · **Size:** M · **Depends on:** U6-09 · **Resolves:** F-20 (Run health panel), REVIEW_FINDINGS part 2 D (Run health panel)
- **Carry in from U4-18:** rewrite `tests/validation/DASHBOARD_CHECKLIST.md` for the rebuilt dashboard and update `.github/ISSUE_TEMPLATE/dashboard_ui.yml`. **Note from U5-06:** show `Run.policy`. **Carry in from U0-16:** the light theme draws chart titles and card headings almost white on a light background.
- **Outcome:** `/`, `/datasets/:datasetKey` and `/runs/:runId` (a lazy chunk each) with light and dark theme tokens checked at WCAG AA, paged dataset and run lists, and the overview: header, sampling notice, policy verdict, Run health panel (every status with its reason, hint and `blocked_by`, plus run notes), finding counts by dimension and severity, and the top 10 findings. Playwright runs over a fixture store (`frontend/e2e`, job `dashboard-e2e`): opening titanic transfers 146 KB; `pnpm check:bundle` holds the entry chunk to 110 KB and the bundle to 140 KB. **Deviations:** `not_applicable` checks joined `RunHealth.issues` (asked the user; run schema 0.13.0) because the overview had no source for their reasons; `dsbf dashboard` falls back to `index.html` for page routes so reloads work; echarts registers inside `DsbfChart` to keep it out of the entry chunk.
- **Note from U6-04:** the store tests build their own `Run` fixtures from `tests/test_store/conftest.py`; the fixture store for the Playwright tests is built here.
- **Goal:** The app shell (layout, light and dark themes, a lazy chunk per route), the dataset and run lists, and a run's overview: the health panel (each status with its notes and hints, `blocked_by`, unavailable checks with their extra, the sampling notice, run notes), the policy verdict, finding counts by dimension and severity, and the top findings. A Playwright test against a fixture store opens titanic and measures what opening a run transfers.
- **Decided in U6-00:** each tab unit after this one adds its Vitest component tests and a Playwright test, and turns its part of the checklist into them; CI checks the bundle against a size budget set from this unit's first build.
- **Assumptions to confirm:** Playwright runs on the ubuntu runner with a cached browser; the test setup can build a fixture store with `dsbf`; the theme tokens reach WCAG AA contrast for text in both themes.
- **Steps:** the shell and routes; paged lists; the overview and health panel; both themes checked; Playwright with the transfer test and a CI job; the checklist rewritten for the overview; the issue template updated.
- **Acceptance:** Opening titanic transfers under 200 KB, measured by Playwright. Every status other than `ok` shows its reason. The policy verdict shows when there is one. The CI frontend job runs Playwright and the bundle budget.

### U6-11 · Findings explorer
- **Status:** done (2026-10-28, #163, `f75b769`) · **Size:** M · **Depends on:** U6-10 · **Resolves:** N-A5
- **Outcome:** `/runs/:runId/findings` keeps filters, sort, page and the open finding in the URL; a finding opens to its evidence, assumptions and guidance, and its `details_ref` artifact pages by row range on request. The overview's top findings link to it. `listFindings` gained `check`, `min_confidence` and `sort` (confidence read from the stored JSON, so no migration). **Deviations:** the Playwright fixture store gained a `cities` run, because titanic has no finding with an artifact; the severity and confidence filters are "at least", as `Findings.filter` is.
- **Note from U6-02:** a finding's `details_ref` names an artifact (the full fuzzy pair list, the STL series); the explorer should offer it through the API's artifact route.
- **Goal:** Every finding of a run, filtered by dimension, severity, confidence, check, code and column, sorted and paged from the API, with the filters in the URL. A finding opens to its message, evidence (metrics, `n_used` of `n_total`, method, interval, assumptions, sampling), guidance actions, and its artifact on demand.
- **Assumptions to confirm:** the findings endpoint takes each of these filters; a pair-subject finding matches a filter on either column, as `Findings.filter` does.
- **Steps:** the list with filters in the URL; the detail panel; an artifact table viewer that pages by row range; component and Playwright tests; the checklist's findings items.
- **Acceptance:** On titanic, the explorer's counts under each filter equal `run.findings.filter` (a test compares them). A pair-subject finding appears under both of its columns. The fuzzy pair list pages without loading whole.

### U6-12 · Column view
- **Status:** done (2026-10-28, #164, `95dc699`) · **Size:** M · **Depends on:** U6-11 · **Resolves:** F-12, N-A3
- **Outcome:** `/runs/:runId/columns` lists the columns with name and type filters in the URL, and `/runs/:runId/columns/:column` shows the profile, the histogram on a value axis from the stored edges, top values, the Q-Q plot from the `normality` output, the column's findings and a paged snapshot sample. **Deviations:** the unit added `GET /v2/runs/{run_id}/columns` (`listColumns`), because no endpoint listed a run's columns; a column that passed normality has no stored Q-Q data, so the page says why instead of drawing one; the fixture store gained a `bare` run without a snapshot; the bundle budget's total rose from 140 KB to 880 KB for the lazy ECharts chunk.
- **Note from U6-11:** the explorer's `check` and `code` filters are free text; if the column view adds an endpoint of distinct values, reuse it to make them selects.
- **Goal:** A column browser and a page per column: the profile's statistics, its equal-width histogram on a value axis with the real edges, top values, the Q-Q plot from the `normality` output when there is one, the column's findings, and a sample of its values from the snapshot.
- **Decided in U6-00:** the charts draw stored bins and quantiles only; nothing is recomputed in the browser.
- **Assumptions to confirm:** the column endpoint returns the bins and top values; the `normality` output keeps the Q-Q quantiles and reference line of each column it flagged; datetime and string columns have their own profile shapes to draw.
- **Steps:** the browser and page; histogram, top values and Q-Q cards built on `DsbfChart`; the snapshot sample table; tests; the checklist's column items.
- **Acceptance:** Titanic's `Age` histogram matches the profile's edges and counts. A column that failed normality shows its Q-Q plot with the reference line. A run without a snapshot says so where the sample would be.

### U6-13 · Relationships
- **Status:** done (2026-10-29, #165, `8c169d7`) · **Size:** M · **Depends on:** U6-12
- **Outcome:** `/runs/:runId/relationships` draws one heatmap panel per statistic family (correlation, η, V, ω²) from `GET /v2/runs/{run_id}/relationships`, which builds cells from the stored pairs with their statistic; an untested pair has no cell. A table groups pairs by statistic, then by strength. Cards show collinearity and target leakage, and the contingency counts and group difference of the chosen pair. `GET /v2/runs/{run_id}/snapshot/pair` serves up to 2,000 evenly thinned points for the pair plot, with the snapshot's sampling. **Deviations:** no repository method was added (the endpoint reads the snapshot and selects two columns); heatmap squares are not clickable, since `DsbfChart` forwards no chart events, so a pair is chosen from the table or two selects (a later frontend unit if wanted); the archetype lines of the checklist stay open until the quality unit builds the archetype stores.
- **Carry in from U4-07:** `associations` returns the tested pairs and no matrix, so the view builds one on demand.
- **Goal:** The association matrix, collinearity, target leakage, contingency and group differences, and a pair plot of two columns from the snapshot.
- **Decided in U6-00:** an API endpoint builds the matrix from the stored pairs, not the core and not the browser; an untested pair is blank, not zero; each cell names its statistic, and sorting groups by statistic before value, since η² and r are not on one scale (F-25).
- **Assumptions to confirm:** the `associations` output names each pair's statistic and strength class; the snapshot holds both columns of a pair.
- **Steps:** the matrix endpoint (regenerating the client); the tab's cards; the pair plot; tests; the checklist's relationship items.
- **Acceptance:** Titanic's matrix cells equal the `associations` output, with untested pairs blank. Sorting groups by statistic. The pair plot draws from the snapshot and says how it was sampled.

### U6-14 · Quality and ML readiness
- **Status:** done (2026-10-29) · **Size:** M
- **Outcome:** `/runs/:runId/quality` shows the ML-readiness gate, five readiness and five quality dimension cards (level, coverage, input states, findings with `ml` actions), clean columns and per-column coverage, from `GET /v2/runs/{run_id}/coverage` and the two checks' outputs. A dimension without a level reads "Not assessed" with its reason, never green; without a target Leakage is unassessed on every archetype, so `clean` is `ready` with a caveat. The `clean` and `high_missingness` checklist lines are Playwright tests. **Deviations:** the relationships archetype lines and the other archetypes' quality lines stay open (no fixture stores for them); the coverage endpoint loads the whole run. · **Depends on:** U6-13
- **Note from U6-13:** the checklist's archetype lines for the relationships tab (block structure on `severe_multicollinearity`, 60 columns on `wide`, Cramér's V only, a single column, the Summary card counts) need the archetype fixture stores this unit builds; convert them to Playwright tests then.
- **Note from U6-10:** the overview shows counts by dimension only; the quality dimensions and the ML-readiness gate are this unit's.
- **Carry in from U4-15:** the tab draws the outputs of `quality_dimensions` and `ml_readiness`: levels, gate, per-input states and coverage.
- **Goal:** The five quality dimensions with their levels and findings, the ML-readiness gate with its per-input states, and coverage per input check and per column (from `CheckResult.columns`, U6-03).
- **Decided in U6-00:** a dimension that is `not assessed` has its own state (neutral and explained), never green.
- **Assumptions to confirm:** the two outputs carry every level, state and coverage figure the tab draws; the archetype datasets build fixture stores in the test setup.
- **Steps:** the quality and ML-readiness views; per-column coverage; Playwright tests from the checklist's quality and ML-readiness items on the `clean` and `high_missingness` archetypes.
- **Acceptance:** The checklist's quality and ML-readiness items pass as Playwright tests. A dimension that was not assessed shows as such, not green.

### U6-15 · Time series
- **Status:** todo · **Size:** M · **Depends on:** U6-14 · **Resolves:** N-A4
- **Note from U6-14:** the checklist's remaining archetype lines (relationships on `severe_multicollinearity`, `wide`, `all_categorical` and `single_column`; quality and ML readiness for the other archetypes) need fixture stores for those archetypes in `frontend/e2e/fixture_store.py`; add them here if this unit needs more archetypes, else in the last dashboard unit.
- **Carry in from U4-11:** the ACF and PACF values and bands are in the `autocorrelation` output; the STL series are artifacts from U6-02.
- **Goal:** Stationarity results, ACF and PACF with their bands, the four STL series, and an empty state that says how to set the time index (the CLI flag or the setting; U6-17 adds a rerun from the dashboard).
- **Assumptions to confirm:** the outputs and artifacts hold everything the views draw; kitchen5k with its time index builds a fixture store.
- **Steps:** the views; the empty state; tests; the checklist's time-series items.
- **Acceptance:** Kitchen5k with its time index shows the four STL series and the ACF bands. A run with no time index shows the empty state.

### U6-16 · Run compare and history
- **Status:** todo · **Size:** M · **Depends on:** U6-15 · **Resolves:** REVIEW_FINDINGS part 2 D (run comparison UI)
- **Note from U6-06:** `compare` reports most count-scaled statistics (chi², `n_missing`) as changed when two runs have different row counts (107 of 117 titanic findings against a copy with 100 rows dropped); a per-metric tolerance or a list of metrics that scale with n would cut the noise.
- **Goal:** A compare view of any two runs, chosen explicitly and pre-filled with the previous run of the same dataset, showing new, resolved and changed findings and status changes from the compare endpoint. A dataset page with its run lineage (changes of location and fingerprint) and the history of one finding across runs.
- **Decided in U6-00:** comparison is explicit (asked the user); drift from stored profiles is Phase 7.
- **Assumptions to confirm:** the compare and history endpoints from U6-07 return what the views draw; a fixture store can hold two runs of one dataset with a planted change.
- **Steps:** the compare view and its run picker; the dataset page with lineage and history; tests.
- **Acceptance:** Comparing two titanic runs with a planted change shows the expected diff. Comparing runs of two datasets works and says they differ. A changed source location is flagged in the lineage.

### U6-17 · Settings and rerun
- **Status:** todo · **Size:** M · **Depends on:** U6-16 · **Resolves:** REVIEW_FINDINGS part 2 D (settings panel)
- **Note from U6-08:** the job request takes `source`, `name` and `settings` only; a rerun that wants a policy verdict needs `policy` and `previous` added to it.
- **Goal:** A settings form generated from the `ProfileConfig` JSON Schema (target, time index, depth, checks and their parameters), saved as a per-dataset overlay in the store, and a rerun that submits a job, follows its progress and lands on the new run.
- **Decided in U6-00:** the overlay is stored per dataset key; an ungrouped run reruns with its own settings and no overlay.
- **Assumptions to confirm:** `dsbf schema export` publishes the `ProfileConfig` schema, or this unit adds it; a stored run's source location is enough to rerun it when the file still exists.
- **Steps:** the overlay table (an Alembic revision) and endpoints; the generated form; rerun with progress; tests for a source that no longer exists.
- **Acceptance:** Setting a target on titanic and rerunning gives a run in which `class_imbalance` runs. A source that no longer exists fails the job, and the UI shows the message.

### U6-18 · Stable 1.0 schemas
- **Status:** todo · **Size:** S · **Depends on:** U6-17
- **Note from U6-06:** `RunDiff` has a schema file (`schema/diff.schema.json`) but no breaking-change gate yet.
- **Goal:** Review every published contract (the run, the check schemas, the Parquet tables, the policy, `RunDiff`, OpenAPI) for what 1.x has to keep. Raise each to 1.0.0, and document the compatibility rules (additive within 1.x, a deprecation lasting one minor release; REFACTOR_PLAN §4.3), which the breaking-change gates enforce.
- **Assumptions to confirm:** each schema has a version and a gate (the run, tables, check and policy schemas from Phase 5, OpenAPI from U6-07, `RunDiff` from U6-06); nothing in the contracts is still marked as development-era.
- **Steps:** the review, recorded in the PR; version bumps; the compatibility guide in `docs/guides/`; gate tests for a breaking change without a major bump.
- **Acceptance:** Every schema is at 1.0.0, and each gate fails on a breaking change without a major bump.

### U6-19 · Bundle the dashboard and release 1.0.0
- **Status:** todo · **Size:** M · **Depends on:** U6-18 · **Resolves:** REVIEW_FINDINGS part 2 D (`dsbf dashboard`) and E (the frontend in the wheel), F-29
- **Note from U6-12:** the column page's chunk holds ECharts (about 700 KB, 236 KB gzipped); split ECharts into its own chunk and register only the chart types in use before the release gate. `DsbfChart.vue` also hard-codes dark colours for its card, title and tab strip, so the light theme draws them wrongly (the U0-16 carry-in is still open); move them to the theme tokens.
- **Note from U6-10:** `test_the_budget_kills_a_worker_too` (`tests/test_engine/test_process_workers.py`) failed once on the Windows CI job of #160 (`FileNotFoundError` reading the pid file) and passed on a rerun; the PR touched no engine code. Probable cause, not reproduced: `run_budget_seconds=0.8` lets a slow runner kill the worker before it writes its pid file; the interrupt test beside it waits for the file with a deadline. Fix with a deadline wait or a larger budget on Windows before the release gate relies on the Windows job; treat a second failure as a real race.
- **Note from U6-10:** the dashboard is served uncompressed; add gzip to the app if a tab pushes the 200 KB transfer budget. `dsbf dashboard` already serves `index.html` for page routes (`DashboardFiles`); keep that when bundling.
- **Note from U5-09 (4):** a coverage floor is undecided; the first number is in the `full` job's step summary.
- **Goal:** The wheel carries the built dashboard, and `dsbf dashboard` serves it and the API on one port. The `package` CI job builds the frontend, installs the wheel with `[server]` into a clean venv, runs `dsbf check data.csv --sink store`, starts `dsbf dashboard` and fetches the page and a run summary. Release `1.0.0`.
- **Decided in U6-00:** the frontend is built in CI and the release workflow, never committed; a wheel built without it serves the API and says the dashboard is missing; the coverage floor is the measured total rounded down to a multiple of 5.
- **Assumptions to confirm:** `uv_build` includes the built assets from a path the release workflow fills; the Windows leg can start the server and fetch from it; the release workflow can install Node and pnpm.
- **Steps:** the frontend build in the `package` job and release workflow; the static mount with a fallback to `index.html`; the clean-venv smoke on Linux and Windows; the coverage floor; `Development Status :: 5 - Production/Stable`; the changelog; a standalone-local guide and the README; the user pushes the tag after merge.
- **Acceptance:** The phase's success criteria hold, each measured in the PR or the release run.

## Phase 7: Scale and sources (placeholder)

- **Capability:** Larger-than-memory files and object storage (T2), warehouse sources with pushdown through Ibis (T3), recurring datasets with history and profile-based drift (T4), and scheduling recipes.
- **Motivation:** R3: from a laptop to cloud pipelines without redesigning the core (REFACTOR_PLAN §5.1).
- **Boundaries:** No team features (Phase 8) and no new modalities (Phase 9). Checks never see a path or a connection, only a `Dataset` and its profile.
- **Sequence and dependencies:** Depends on Phase 6 (history needs the store). T2, then T3 (DuckDB and Postgres first), then T4, then recipes.
- **Success criteria:** 1M × 50 within budget; a 100M-row Parquet glob profiled with a sample taken at read time and its provenance shown; one warehouse source verified end to end.

- **U7-00 · Plan Phase 7**.
  - **Carry in from U4-19, U5-09 and U6-00:** the `sql` extra has no importer, install-matrix leg or `unavailable` test; add all three with the first Ibis source. (The `server` extra gets its pair in U6-04.)
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
