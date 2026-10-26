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

Phase 5's units were confirmed by U5-00 and are `todo`. U5-10 (safe column names) runs first; U5-11 (calibration fixes) runs before the docs.
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

### U5-00 · Plan Phase 5
- **Status:** done (2026-10-25, PR #132, `b93ff03`) · **Size:** S · **Depends on:** U4-19
- **Note from U4-19:** (1) Promote the perf baseline from the first nightly on `main` after U4-19 merges: the committed one is the 2026-10-25 nightly (`a2bda9e`), before the four dependency caps lifted; the `standard-*` budgets are enforced from U4-19. (2) A column named `^a$` (any name Polars reads as a regular expression) makes `infer_column_type` raise `IndexError`, and a column named `*` makes `profile_columns` raise a `DuplicateError` beside an all-null column, because `pl.col` and `Series` methods read the name as a pattern. Fixing the profiler alone moves the failure to type inference, so fix it in `dsbf.data.load` (refuse such names with a `SourceError`, or carry a safe internal alias) and drop `assume("*" not in frame.columns)` from `tests/test_checks/test_adversarial.py`. (3) Decide whether to add a calibration-fix unit to this phase for the gaps U4-16 recorded: `fuzzy_duplicates` finds the same people in two systems only with `exclude_columns` for the key that differs (a leave-one-column-out score would find them unaided), has no symmetric-delete blocking for single-character typos in every token, and flags look-alike records that differ in a letter and a digit (15% of runs); `bimodality` misses peaks 4 spreads (Ashman's D) apart in most samples below 300 values; `stationarity` misses a random walk in 3 runs of 20; `multivariate_outliers` calls about 24% of the rows of an uncontaminated correlated lognormal table far, while its finding says normal data reaches that in 1% of samples (plant skewed and heavy-tailed nulls, and make the limit or the wording robust); the `quality_dimensions` red share (0.5) is a judgement, with no scenario between 5% and 70% of the columns. (4) The `sql` and `server` extras have no install-matrix leg or `unavailable` test, since no code imports them; add both with the first code that does. (5) scikit-learn 1.8 changed `IsolationForest` scores on kitchen5k (`max_anomaly_score` 0.582 to 0.609, `excess` 0.034 to 0.061; the reference forest on normal data is identical, per-row scores correlate 0.94); the cause is not traced, and the `ml` extra's floor is 1.8. (6) The `extras` CI job profiles a CSV from the installed command; fold it into the clean-venv job of U5-09. (7) `--perf-cprofile` went with the legacy perf scenarios; add one to `tests/perf/measure_engine.py` only if a check-level profile is wanted.
- **Outcome:** Phase 5 is confirmed against the repository as of 2026-10-25; two units were added (U5-10, U5-11) for the U4-19 notes the unit had to decide, and the decisions are recorded on the units they affect.
- **Goal:** Confirm the outline units below against what Phase 4 delivered, and fill in each one's assumptions and steps. Ask the user how the package will be published (the PyPI name `dsbf` and trusted publishing) and confirm the HTML report's scope.

### U5-10 · Safe column names
- **Status:** done (2026-10-25, PR #133, `24c33cd`) · **Size:** S · **Depends on:** U5-00
- **Note from U4-19:** a column named `^a$` (any name Polars reads as a regular expression) makes `infer_column_type` raise `IndexError`, and a column named `*` makes `profile_columns` raise a `DuplicateError` beside an all-null column, because `pl.col` and `Series` methods read the name as a pattern.
- **Assumptions to confirm:** (1) The failure still reproduces on `main` (U5-00 reproduced `^a$` on 2026-10-25). (2) Every `pl.col(name)` over a user column name sits in `dsbf.data` and `dsbf.profiling`; checks read columns from the profile or through `Dataset` accessors. Grep before choosing between the two options. (3) Whether the nightly on `main` after U4-19 has run; if so, promote the perf baseline from it (the committed one is the 2026-10-25 nightly, `a2bda9e`).
- **Goal:** No column name can make a run crash. `dsbf.data.load` refuses the unsafe names it cannot carry with a `SourceError` that lists them, or carries a safe internal alias; U5-00's decision is the alias if step 2 shows it is local to `dsbf.data`, otherwise the refusal.
- **Steps:** reproduce both names in a test; fix in `dsbf.data.load`; drop `assume("*" not in frame.columns)` from `tests/test_checks/test_adversarial.py`; promote the perf baseline when the nightly exists.
- **Acceptance:** A table with `^a$`, `*`, `.*` and `a.b` columns profiles or is refused with a `SourceError`, never an `IndexError` or `DuplicateError`. The adversarial suite no longer excludes `*`.
- **Outcome:** `dsbf.data.load` refuses a column named `*` or one that starts with `^` and ends with `$` with a `SourceReadError` that lists them; the alias option was dropped because every Polars `Series` method selects its own name as a pattern, so an alias would have to cover every series a check receives. The adversarial suite's strategy skips those two name shapes. The perf baseline was not promoted: no Performance run on `main` exists after U4-19 merged.

### U5-01 · Policy evaluator
- **Status:** done (2026-10-26, PR #134, `b68ae76`) · **Size:** M · **Depends on:** U5-00
- **Note from U5-10:** promote the perf baseline from the first scheduled Performance run on `main` after 2026-10-25 14:35 -06:00 (the committed one is the 2026-10-25 nightly, `a2bda9e`); do it as the first step if that run exists.
- **Note from U4-06:** `out_of_bounds` (plausibility range, `warn`) and `schema` (contract, with min and max per column) both take bounds for a column. Decide whether a policy rule reads one as the other.
- **Note from U1-10:** (1) the policy loader turns `yaml.safe_load` and `Policy.model_validate` failures into a `ConfigError`-style report that names the rule; (2) decide what `new_since: previous_run` does without a previous run; (3) add a range on one setting (`>0.1` and `<0.5`) only if users ask. **Note from U1-11:** publish `schema/policy.schema.json` with the loose spellings documented, together with its loader.
- **Goal:** `dsbf.profile(..., policy=..., previous=...)` evaluates the rules over findings and health and fills `Run.policy`. `fail` only sets the exit code.
- **Decided in U5-00:** (1) A policy rule reads a finding's evidence metric by name and never cross-reads `out_of_bounds` bounds as `schema` min and max; the metric names come from each check's output schema, which the U5-08 reference lists. (2) `new_since: previous_run` without a previous run matches nothing and adds a note to `Run.notes`, because a pipeline's first run has no previous one and report-only is the default (ADR 0006). (3) `policy=` takes a path, a mapping or a `Policy`; `previous=` takes a `Run` or a `run.json` path (reading one needs U5-03, so a path is accepted from then on and a `Run` until then).
- **Assumptions to confirm:** `dsbf.model.policy` already validates the §7.2 example and `Run.policy` already checks its outcomes against the rules (audited 2026-10-25: `Policy`, `Rule`, `Matcher`, `PolicyOutcome` and `Run._check_policy` exist); no evaluator, loader or `policy=` argument exists yet; a finding's evidence exposes the metric names a rule needs.
- **Steps:** evaluator over findings and health; loader from YAML with errors that name the rule; `policy=` and `previous=` on `dsbf.profile`; `schema/policy.schema.json` through `dsbf schema export`, with `--check`; a unit test per matcher key and per action, and the §7.2 example as a fixture.
- **Acceptance:** A REFACTOR_PLAN §7.2 example policy produces the documented `Run.policy`. A run with no rules matched exits 0. The policy schema export is checked in CI.
- **Outcome:** `dsbf.policy` loads a policy (`load_policy`, errors named by rule) and evaluates it (`evaluate_policy`); `dsbf.profile(policy=, previous=)` fills `Run.policy` without revalidating the run, and `schema/policy.schema.json` is exported beside the run schema. The §7.2 example filtered on `null_rate`, a metric no check emits, so it now reads `null_share`. `previous` takes a `Run` only until U5-03; the perf baseline was promoted from the 2026-10-26 run on `main`.

### U5-02 · Findings query API and notebook display
- **Status:** done (2026-10-26, PR #135, `502dbec`) · **Size:** S · **Depends on:** U5-01 · **Resolves:** REVIEW_FINDINGS part 2 D (notebook experience)
- **Goal:** `run.findings.filter(severity=">=warn", dimension=..., column=...)`, `.to_polars()` and `.to_pandas()` (with the `pandas` extra). `Run._repr_html_` shows the health summary and the top findings in a notebook.
- **Assumptions to confirm:** `Run.findings` is a tuple ordered by `order_findings`; `pandas` is an optional extra, so `.to_pandas()` raises the `unavailable` style error without it; `Finding` subjects include column pairs and the whole dataset, so `column=` matches a pair on either column (as the policy matcher does).
- **Steps:** a `Findings` view with `filter`, `to_polars` and `to_pandas`; `Run._repr_html_` built with escaping; the §9 example as a test.
- **Acceptance:** The REFACTOR_PLAN §9 library example runs as a test. The HTML display renders the adversarial run without breaking on odd column names.
- **Outcome:** `Run.findings` is a `Findings` tuple view with `filter` (the policy matcher's keys and operators; the matching code moved to `Matcher.matches`), `to_polars` (11 flat columns, no evidence metrics) and `to_pandas` (`ImportError` with the install hint without the extra, not a loader error). `Run._repr_html_` shows the health summary and the top 10 findings, escaped.

### U5-03 · JSON and NDJSON sinks, and reading a run back
- **Status:** done (2026-10-26, PR #136, `aa0c552`) · **Size:** M · **Depends on:** U5-02 · **Resolves:** N-B2 (output schemas)
- **Note from U5-01:** `dsbf.profile(previous=)` takes a `Run` only; once `Run.read` exists, accept a `run.json` path (read it in `dsbf.library` before evaluating the policy).
- **Note from U4-00 (was on U4-02):** every core check declares a params and output model from U4-02 on, but nothing in Phase 4 reads them as JSON Schema, so their export lands here, with the published run schema.
- **Goal:** `run.write(dir, sinks=["json", "ndjson"])` writes `run.json` and `findings.ndjson`, and `dsbf.Run.read(path)` round-trips. The stopgap CLI's writer moves onto it. `dsbf schema export` also writes each core check's params and output schema (`schema/checks/<id>.schema.json`), and `--check` covers them.
- **Assumptions to confirm:** `Run` serialises through `model_dump_json` with no non-finite numbers (F-36); `schema export` is the one place that writes `schema/`; every core check declares a params and an output model (47 do, per the Phase 4 summary).
- **Steps:** `Run.write` with a sink registry that U5-04 and U5-05 extend; `Run.read`; NDJSON of findings, one finding per line; per-check schemas under `schema/checks/`; move the CLI writer onto `Run.write`; extend the breaking-change gate to the check schemas.
- **Acceptance:** A round-trip test passes. The files validate against the published schema with `jsonschema`, and each check's output in a titanic run validates against its exported schema. `dsbf schema export --check` covers the check schemas.
- **Outcome:** `dsbf.sinks` holds a name-keyed sink registry (`json`, `ndjson`) that U5-04 and U5-05 extend; `Run.write` and `Run.read` (`dsbf.Run`) use it, and `dsbf.profile(previous=)` takes a `run.json` path. `dsbf profile` writes through `Run.write` and gains `--sink`. `dsbf schema export` writes 47 `schema/checks/<id>.schema.json` files (params and output), and `schema check` guards them by each check's `check_version`. Policy schema gating stays with U5-09.

### U5-04 · Parquet sinks and the consumer schema guide
- **Status:** done (2026-10-26, PR #137, `569b60b`) · **Size:** M · **Depends on:** U5-03
- **Note from U5-02:** build the findings table from the columns of `Findings.to_polars` (`id`, `check_id`, `code`, `dimension`, `severity`, `confidence`, `subject_kind`, `columns`, `title`, `message`, `tags`), adding the evidence metrics.
- **Goal:** `findings.parquet`, `profile.parquet` and `results.parquet` with documented, versioned columns, plus a guide for loading them into a warehouse or BI tool.
- **Decided in U5-00:** the Parquet files carry a `schema_version` in their key-value metadata, and the breaking-change check compares column names, types and nullability against a checked-in schema listing.
- **Assumptions to confirm:** the profile's per-column stats and histograms flatten to one row per column, with the bins in a list column; a finding's evidence flattens to JSON text or a map, chosen by what DuckDB reads back; `pyarrow` is in the base install (it is).
- **Steps:** three writers on the U5-03 sink registry; the column listing and its check; the guide in `docs/` with queries that a test runs through DuckDB (a dev dependency only).
- **Acceptance:** The Parquet schemas have their own breaking-change check. DuckDB reads the example queries in the guide back.
- **Outcome:** `findings_parquet`, `profile_parquet` and `results_parquet` sinks (the `parquet` alias writes all three) build their tables from fixed Arrow schemas in `dsbf.sinks.tables`, each starting with `run_id` and carrying `schema_version` 0.1.0 in its metadata. Evidence metrics, check params and output are JSON text. `schema/tables.schema.json` lists the columns, `dsbf schema export` writes it and `schema check` guards it by `TABLES_SCHEMA_VERSION`. `docs/CONSUMING_PARQUET.md` has the guide, whose SQL a test runs through DuckDB (dev group only).

### U5-05 · Static HTML report
- **Status:** done (2026-10-26, PR #138, `385a80f`) · **Size:** M · **Depends on:** U5-04 · **Resolves:** REVIEW_FINDINGS part 2 D (shareable export)
- **Goal:** One self-contained HTML file per run (no server, no CDN) with the health summary, findings by dimension and severity, and per-column profiles drawn from the profile's equal-width bins.
- **Decided in U5-00:** scope is the health summary, findings by dimension and severity, and per-column profiles (histogram, top values, counts) as inline SVG. Quality and ML-readiness sections, relationships and time-series views are left to the Phase 6 dashboard. The size budget is 1 MB for the titanic report (its `run.json` is 665 KB on 2026-10-25), measured on the first build and tightened if the report comes in far under.
- **Assumptions to confirm:** the profile carries the bins each chart needs, so the report reads the `Run` only; no template dependency is needed (string building with `html.escape`); dataset, column and finding text all pass through one escaping function.
- **Steps:** a renderer module in `dsbf.sinks`; the HTML sink on the registry; tests with hostile names (`<script>`, quotes, `&`) and a no-network check (no `http` URL in the output).
- **Acceptance:** The titanic report is under a size budget set in U5-00 and opens offline. User text is escaped (the F-14 lesson).
- **Outcome:** `dsbf.sinks.report` renders one offline page (health, findings by dimension and severity, a card with an inline SVG chart per column) and registers the `html` sink (`report.html`). Every string goes through one escaping function. The titanic report is 78 KB, so the budget is tightened from 1 MB to 150 KB. The report omits the policy outcome.

### U5-06 · CLI v2
- **Status:** done (2026-10-26, PR #139, `92810b4`) · **Size:** M · **Depends on:** U5-05 · **Resolves:** N-B1 (generated CLI help and config reference), REVIEW_FINDINGS part 2 G (pipeline hooks)
- **Note from U3-09:** give the Typer app one callback that attaches the console handler, with a `--verbosity` option, so every command prints log records. **Note from U1-09:** CLI v2 builds the `cli` layer for `dsbf.config.load_config` and needs a `--set a.b=c` parser.
- **Note from U4-05:** `dsbf.profile()` takes no per-check parameters beyond the `target` setting, so a core check's params (such as `schema`'s expectations) reach it only through `make_plan(params=)` and `dsbf plan --param`; add `params=` to `profile()` and a section to `ProfileConfig`. Eligibility ignores parameters, so `dsbf plan` says `class_imbalance` and `schema` "will run" where they end `not_applicable` without a target or schema: let a check declare the parameters it needs.
- **Goal:** `dsbf profile` and `dsbf check` (with `--config`, `--set`, `--policy`, `--previous`, `--out` and `--sink`), plus `plan` and `schema export`. Every run prints the health summary. The exit code is 0, or 1 when the policy fails the run, or 2 on a setup error.
- **Note from U5-05:** the HTML report does not show `Run.policy`; show the outcome there if the CLI exit-code work wants it. The `--sink` help now lists `html`.
- **Note from U5-04:** `known_sinks()` lists the `parquet` alias after the sinks, so the `--sink` help shows it; fold that into the generated CLI reference.
- **Decided in U5-00:** `profile` and `check` share one implementation; `check` differs by requiring a policy-aware exit code (a policy is optional, and with none the exit code is 0 as in §7.1). The `extras` CI job's profile step (U4-19 note) folds into U5-09's clean-venv job.
- **Assumptions to confirm:** `dsbf.config.load_config` takes a `cli` layer; the stopgap `profile` command and `plan` exist (audited 2026-10-25); `schema export` and `schema check` exist, so only their naming is checked against the goal.
- **Steps:** the Typer callback with `--verbosity`; `--config`, `--set`, `--policy`, `--previous`, `--out`, `--sink`; `params=` on `dsbf.profile` and a `ProfileConfig` section; parameter-needed declarations so `plan` stops saying "will run" for a check that ends `not_applicable`; exit codes 0, 1, 2; generated help and config reference.
- **Acceptance:** The commands are tested end to end from a temporary directory, and each exit code is tested.
- **Outcome:** `dsbf profile` and the new `dsbf check` take `--config`, `--set`, `--policy`, `--previous`, `--out`, `--sink` and the check-selection options, building `ProfileConfig` through `load_config` (file, environment, options, `--set`) and running it with `dsbf.library.profile_with`. `check` exits 1 when a policy rule fails the run, 2 on a setup error; `profile` stays report-only. A top-level `--verbosity` attaches the console handler for every command. `Requires(params=…)` makes `plan` and the run mark `class_imbalance`, `mutual_information`, `target_leakage`, `schema` and the time-series checks `not_applicable` until their parameter is set (`schema`'s code is now `no_columns`). `dsbf docs export` writes `docs/CLI_REFERENCE.md` from the Typer app and `ProfileConfig`.

### U5-07 · Execution settings and cache management
- **Status:** done (2026-10-26, PR #140, `9c0e862`) · **Size:** S · **Depends on:** U5-06
- **Note from U3-07:** expose the worker count in the CLI and config, with per-check `runs_in` and timeout overrides (U3-06 deferred those too). **Note from U3-08:** add `dsbf cache clear` and a size or age eviction for `<DSBF_HOME>/cache/results/`, and expose cache on/off in the CLI and config.
- **Assumptions to confirm:** the executor takes `workers` and `Limits` (it does); `DiskCache` has no eviction and no `clear` yet; a per-check `runs_in` and timeout are fields the registry can override from config.
- **Steps:** config section and CLI flags; `dsbf cache clear`; eviction by size and age under `<DSBF_HOME>/cache/results/`.
- **Acceptance:** A per-check timeout set in config applies. Eviction keeps the cache under its cap, which a test checks.
- **Outcome:** `ProfileConfig` gains `check_settings.<id>` (`timeout_seconds`, `runs_in`, carried on the launch because `CheckSpec` is frozen; unknown ids are a `ConfigError`) and `cache` (`enabled`, off by default; `max_megabytes` 500; `max_age_days` 90). `DiskCache.prune` evicts by age then least recent use (a hit refreshes the entry), `clear` empties it, and a run on the default cache prunes afterwards. `profile` and `check` gain `--workers`, `--check-timeout`, `--run-budget` and `--cache/--no-cache`; `dsbf cache clear` is new.

### U5-11 · Calibration fixes
- **Status:** done (2026-10-26, PR #141, `d58077c`) · **Size:** M · **Depends on:** U5-07
- **Note from U4-19:** scikit-learn 1.8 moved `IsolationForest` scores on kitchen5k (`max_anomaly_score` 0.582 to 0.609, `excess` 0.034 to 0.061; per-row scores correlate 0.94). The cause is not traced; trace it here if the work on `multivariate_outliers` makes it cheap, and say so if it does not.
- **Goal:** Close the gaps U4-16 recorded, because the first release should not state a rate the check does not meet: (1) `multivariate_outliers` calls about 24% of the rows of an uncontaminated correlated lognormal table far, while its finding says normal data reaches that in 1% of samples; plant skewed and heavy-tailed nulls, then make the limit or the wording robust. (2) `bimodality` misses peaks 4 spreads (Ashman's D) apart in most samples below 300 values. (3) `stationarity` misses a random walk in 3 runs of 20. (4) `fuzzy_duplicates` finds the same people in two systems only with `exclude_columns` (try a leave-one-column-out score), has no symmetric-delete blocking for single-character typos in every token, and flags look-alike records differing in a letter and a digit in 15% of runs. (5) The `quality_dimensions` red share (0.5) is a judgement with no scenario between 5% and 70% of the columns; add one or record it as a documented default. Items 2 to 5 are each fixed, or recorded as a known limit in the check's documentation and finding text; the unit says which per item.
- **Assumptions to confirm:** The gaps still reproduce in `tests/calibration`; each fix changes only the named check's rates (golden snapshots that move are regenerated and explained in the PR).
- **Acceptance:** The calibration suite shows the `multivariate_outliers` far-row rate matching its stated rate on skewed nulls. Each other item has a planted-truth scenario that passes, or a limit stated where the user reads it. The `calibration` tier passes.
- **Outcome:** (1) `multivariate_outliers` 1.2.0 measures the robust distance and the forest on each column's normal scores and builds the forest's reference with the data's correlation; far rows on skewed correlated tables fall from 30% to none (`joint_skewed_nulls`). A marginal spike is no longer a far row, which is the `outliers` check's finding; about 5% of clean 1 000-row tables still get a low-confidence forest finding. (2) `bimodality` 1.1.0 reads the valley from a kernel density: 4-spread peaks are found in 82% of 100 values and all of 200 or more; below about 100 values it is a stated limit. (3) `stationarity` 1.2.0 reads a unit root left standing by both ADF regressions with trend-KPSS rejecting as non-stationary; a walk of 150 periods is found in 31 of 40 samples and ADF's own 5% remains, both stated as limits. (4) `fuzzy_duplicates` 1.2.0 scores a pair with one column left out when the other columns carry more match evidence than chance, blocks on one-character deletions of long tokens, and caps any value pair whose digits differ; unaided two-system detection 0 to 1.0, look-alike false alarms 15% to 0. (5) The `quality_dimensions` 0.5 red share stays as a documented judgement with scenarios at 40% and 60%. The scikit-learn 1.8 score move was not traced; with the correlation-matched reference kitchen5k's `excess` is 0.014 (was 0.061).


### U5-08 · Docs for embedded use
- **Status:** done (2026-10-26) · **Size:** M · **Depends on:** U5-11
- **Outcome:** `dsbf docs export` also writes the generated `docs/CHECK_REFERENCE.md` (47 checks: parameters with bounds, output, finding codes read from check sources with `ast`), and CI runs `--check` on both references. Guides in `docs/guides/` (pipeline with the logging note, policy, dashboard, write-a-check) have every `python` and `bash` block run by `tests/test_docs`; `examples/github-actions/dsbf-check.yml` has its check step run with a failing and a passing policy. The README describes the `Run`, sinks, `dsbf check` and policy. The `critical` severity that the Parquet schema description and guide listed does not exist and was removed. A plugin directory loads once per process (stated in the guide).
- **Note from U5-11:** the limits U5-11 stated live in check docstrings and parameter descriptions (`bimodality` under about 100 values, `stationarity` ADF size and short-walk power, `multivariate_outliers` rank-normalised distance and its residual forest rate); the generated check reference should carry them.
- **Note from U3-09:** a library call prints nothing unless the host configures logging; say so, and list the accepted levels.
- **Goal:** README rewrite. Guides: "Use DSBF in your pipeline", "Consume DSBF findings in your own dashboard", "Write a check" and "Write a policy", plus a generated check reference (params, outputs, findings codes). An example GitHub Actions workflow writes Parquet findings.
- **Assumptions to confirm:** the README still describes the legacy dashboard (rewrite it); the check reference can be generated from the registry and each check's params and output models; code blocks can be marked so a test runs them.
- **Steps:** guides in `docs/guides/`; a generator for the check reference, with a `--check` mode in CI; the `logging` note; the example workflow under `examples/`.
- **Acceptance:** Every code block in the guides is executed by a test, and the example workflow runs in CI.

### U5-09 · Release readiness and 1.0.0a1
- **Status:** todo · **Size:** M · **Depends on:** U5-08 · **Resolves:** F-31 (install smoke test), REVIEW_FINDINGS part 2 E (PyPI) and F (coverage)
- **Note from U5-11:** promote the perf baseline from the first scheduled Performance run on `main` after 2026-10-25 14:35 -06:00 (carried from U5-10; none had run when U5-11 closed).
- **Note from U5-06:** (1) done in U5-08: CI runs `dsbf docs export --check`. (2) Typer 0.27 bundles click as `typer._click`, so `dsbf.interfaces.reference` duck-types commands instead of importing click; re-check it on a Typer bump.
- **Note from U5-01:** `dsbf schema check` guards only `run.schema.json`; extend the breaking-change gate to `policy.schema.json` before the release, since compatibility guarantees start there.
- **Note from U3-07:** worker processes use `spawn` where there is no `forkserver` (Windows), about 0.44 s per check as measured on Linux, and no CI leg covers that path; add one before the first release.
- **Goal:** A CI job that builds the wheel, installs it in a clean venv and runs `dsbf check` from a temporary directory. A Windows leg. Coverage reporting. A release workflow on tags with trusted publishing, the CHANGELOG, and the `1.0.0a1` release. Publishing is the user's step.
- **Decided in U5-00:** the PyPI name is `dsbf` (unclaimed on 2026-10-25); publishing uses trusted publishing (OIDC) from a tag-triggered workflow behind a GitHub environment. The user registers the pending publisher on PyPI and pushes the tag; the unit prepares everything else, including a build that runs before the publish step.
- **Assumptions to confirm:** `pyproject.toml` is at `0.20.0` with the classifier `Development Status :: 4 - Beta` (set `1.0.0a1` and `3 - Alpha`); `CHANGELOG.md` is the generated legacy one and is replaced by a hand-kept changelog; `sql` and `server` extras still have no importer, so the install matrix stays as is until Phase 6.
- **Steps:** wheel build and clean-venv smoke job (absorbing the `extras` job's CSV profile); Windows leg for the `spawn` path; coverage reporting; the release workflow; changelog; version bump.
- **Acceptance:** The smoke job runs on every PR. `pip install dsbf==1.0.0a1` works from PyPI.

---

## Phase 6: Store, service and dashboard v2. Standalone local (placeholder)

- **Capability:** A run store (SQLite with Alembic) and an artifact store (fsspec), `dsbf runs ...`, API v2 with a local job runner, a generated TypeScript client, and the dashboard rebuilt on API v2 and served by `dsbf dashboard` from the wheel; release `1.0.0`.
- **Motivation:** Mode 2 (standalone, local): DSBF is the pipeline, with history and a bundled UI (REFACTOR_PLAN §8.2–§8.4).
- **Boundaries:** Local and single-user only: no Postgres, auth or queue (Phase 8). API v1 and the old store were deleted in U4-18, so this phase builds new and migrates nothing.
- **Sequence and dependencies:** Depends on Phase 5. Store, then API, then client, then the dashboard tab by tab, then packaging.
- **Success criteria:** `pip install "dsbf[server]" && dsbf check data.csv --sink store && dsbf dashboard` works on a clean machine. Opening a run transfers under 200 KB.

- **U6-00 · Plan Phase 6** (`todo`, depends on the Phase 5 units).  **Note from U5-08:** a `codes` field on the check spec would make the check reference's finding codes exact (scanned from source today). **Note from U5-06:** the HTML report omits `Run.policy` (the CLI prints the verdict); show it in the dashboard or report.
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
