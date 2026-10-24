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

Phase 4 is broken down in full. Phase 5 is broken into `outline` units that U5-00 confirms.
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

The four Python caps below protect legacy tasks. The core checks that replace those tasks are
written and tested against the newer releases (each unit runs its tests with `uv run --with`), and
U4-19 lifts the caps once U4-18 has deleted the legacy code.

| Held back (`pyproject.toml` cap) | Why | Revisit in |
|---|---|---|
| `pandas <3` (PR #47, closed 2026-10-16) | 3.0 fails the 3.12 and 3.13 test jobs (string dtype, copy-on-write); the legacy tasks run on pandas | U4-19 (pandas 3 frames already load as an input type, U2-01) |
| `statsmodels <0.15` | 0.15's `variance_inflation_factor` includes the intercept: VIFs drop and four columns on the `clean` archetype stop being flagged | U4-08 decided the centred VIF; U4-19 lifts the cap, with the legacy task gone |
| `scikit-learn <1.8` | 1.8 changes what `GaussianMixture` scores for integer input (titanic `pclass` BIC 1016 → 1763), so the legacy `detect_bimodal_distribution` reports other BIC values and finding titles. The core `bimodality` fits float64 and is the same on 1.7 and 1.8 (U4-06); the mis-scoring is not reported upstream | U4-19 lifts the cap, with the legacy task gone |
| `scipy <1.18` (found in U1-05, no PR) | 1.18 makes `kstest(x, "norm", args=(mu, sigma))` raise `TypeError: ndtr() takes from 1 to 2 positional arguments but 3 were given`, so `run_ks` and two normality tests fail on a fresh `pip install`. The core `normality` standardises first and doesn't make that call (U4-06) | U4-19 lifts the cap, with the legacy task gone |
| `plotly <7` (PR #62, closed 2026-10-16) | 7 removes the `mapbox` layout property that `custom_plotly_templates.py` sets, so it fails at import | U4-18 (plotly is deleted with the backend plotting; drop the cap and the ignore entry) |
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

### How a migration works

These rules apply to every migration unit (U4-02 to U4-15). They were decided in U4-00, and the
retired set and the legacy stack's end were chosen by the user on 2026-10-19.

- **Parallel tracks.** Core checks live in `src/dsbf/checks/`, in the `core` pack, beside the intact `legacy` pack. A core check names the legacy tasks it supersedes. Until U4-15, the default selection is `core` plus every legacy task that nothing supersedes. A superseded task still runs when a remaining legacy task hard-requires it (the planner pulls it in as `required_by`) or reads its result without declaring it (`UNDECLARED_READS` in `dsbf.legacy.adapter`: `data_quality_scorer` reads `detect_out_of_bounds`, `detect_collinear_features` and `summarize_numeric`). Its findings are dropped, so each finding is reported once, and the legacy scorers keep their inputs until U4-15 removes them. No shim feeds legacy scorers from core output.
- **Parity, then intended differences.** Each migration compares the core check with the tasks it supersedes on the ten golden datasets, through the harness from U4-02. A metric either matches within tolerance or is an accepted difference with a reason, as in `tests/test_profile/accepted.py`. A statistical fix is an accepted difference whose reason says why the new result is right. The unit removes its tasks' entries from `tests/golden/engine_differences.json`.
- **Golden findings.** Each unit regenerates the core findings snapshots with `--update-golden` and explains the diff in its PR. The golden suite and the perf suite run on a PR only with the `golden` or `perf` label, so every migration PR carries `golden`, and a PR that changes run time (a new heavy check, or a change to the executor) also carries `perf`.
- **Run time rises before it falls.** `ml_readiness_scorer` hard-requires 14 legacy tasks, most of them superseded in U4-02 to U4-08, so they keep running beside the core checks that replace them until U4-15 takes the scorers out of the default selection. Each unit reports its change to the `standard-*` scenarios; the drop is measured at U4-15.
- **Check conventions** (`src/dsbf/checks/__init__.py` states them): a check reads the columns `applies_to` selects and never picks columns by dtype; it doesn't call `warnings.catch_warnings()`, because the engine records its warnings as notes and the block would hide the warnings of the checks running beside it; it gives scipy numeric arrays, never object dtype (scipy 1.17 checks an object array for NaN element by element); it fits scikit-learn models on float64, because `GaussianMixture` mis-scores integer input (differently in 1.7 and 1.8); it skips a column whose arithmetic overflows, with the reason, and never puts a NaN or infinity in its output; and each finding's evidence carries `n_used` and the method.
- **Every check has** an `applies_to` (column selection by construction, F-11), a `CheckParams` model with bounds, an output model, `n_used` and the method on each finding's evidence, a cost class, its extras (`unavailable` without them), unit tests on small fixtures, and the adversarial-frame test. Recommendation tips are attached after the check returns (U4-04).
- **Retired, not migrated:** the 13 tasks that restate the profile (`summarize_nulls`, `summarize_numeric`, `summarize_unique`, `summarize_value_counts`, `summarize_modes`, `summarize_boolean_fields`, `summarize_text_fields`, `summarize_dataset_shape`, `categorical_length_stats`, `missingness_heatmap`, `infer_types`, `sample_head`, `sample_tail`), whose guidance moves to rule checks; `validate_plugin_coverage`, `identify_bottleneck_tasks` and `log_resource_usage`, which registry notes and `Timing` replace; and the two plot tasks (N-A2).
- **Names** are nouns for the domain behaviour (ADR 0010): `zeros`, not `detect_zeros`.

| Core check | Kind | Supersedes | Unit |
|---|---|---|---|
| `zeros` | rule | `detect_zeros` | U4-02 |
| `null_rate` | rule | `summarize_nulls` (findings) | U4-04 |
| `constant_columns` | rule | `detect_constant_columns` | U4-04 |
| `high_cardinality` | rule | `detect_high_cardinality` | U4-04 |
| `id_columns` | rule | `detect_id_columns` | U4-04 |
| `dominant_value` | rule | `detect_single_dominant_value`; `summarize_boolean_fields` (imbalance finding) | U4-04 |
| `near_zero_variance` | rule | `detect_near_zero_variance`; `summarize_numeric` (near-zero-variance findings) | U4-04 |
| `duplicate_rows` | rule | `detect_duplicates` | U4-04 |
| `duplicate_columns` | rule | `detect_duplicate_columns` | U4-04 |
| `text_encoding` | rule | new: U+FFFD and NUL in names and values | U4-04 |
| `skewness` | rule | `detect_skewness`; `summarize_numeric` (mean–median gap findings) | U4-05 |
| `kurtosis` | rule | `compute_kurtosis` | U4-05 |
| `entropy` | rule | `compute_entropy` | U4-05 |
| `dtype_optimizations` | rule | `suggest_dtype_optimizations` | U4-05 |
| `sample_size` | rule | `sample_size_adequacy` | U4-05 |
| `class_imbalance` | rule | `detect_class_imbalance` | U4-05 |
| `schema` | rule | `schema_validation` | U4-05 |
| `outliers` | analysis | `detect_outliers` (IQR, z-score, MAD) | U4-06 |
| `multivariate_outliers` | analysis (`ml`) | `detect_outliers` (isolation forest) | U4-06 |
| `normality` | analysis | `normality_tests`, `normality_qq_plots` | U4-06 |
| `bimodality` | analysis (`ml`) | `detect_bimodal_distribution` | U4-06 |
| `out_of_bounds` | analysis | `detect_out_of_bounds` | U4-06 |
| `associations` | analysis | `compute_pairwise_associations`; `detect_data_leakage` (feature–feature pairs, F-26) | U4-07 |
| `contingency` | analysis | `contingency_tables` | U4-07 |
| `group_differences` | analysis | `one_way_anova`, `kruskal_wallis`, `mann_whitney_u` | U4-08 |
| `collinearity` | analysis | `detect_collinear_features` | U4-08 |
| `missingness_mechanism` | analysis | `missingness_mechanism_analysis` | U4-09 |
| `mixed_types` | analysis | `detect_mixed_type_columns` | U4-09 |
| `encoded_columns` | analysis | `detect_encoded_columns` | U4-09 |
| `format_violations` | analysis | `detect_regex_format_violations` | U4-09 |
| `string_anomalies` | analysis | `detect_string_anomalies` | U4-09 |
| `fuzzy_duplicates` | analysis, heavy (`text`) | `fuzzy_duplicate_detection` | U4-10 |
| `temporal_gaps`, `datetime_consistency`, `datetime_features` | rule | `temporal_gap_detection`, `check_datetime_consistency`, `extract_datetime_features` | U4-11 |
| `stationarity`, `autocorrelation`, `seasonal_decomposition` | analysis (`timeseries`) | `detect_stationarity`, `compute_acf_pacf`, `decompose_time_series` | U4-11 |
| `reference_drift` | analysis | `compare_with_reference_dataset`, `detect_feature_drift`, `detect_target_drift` | U4-12 |
| `categorical_encoding`, `numeric_binning`, `transformations` | rule, analysis | `suggest_categorical_encoding`, `suggest_numerical_binning`, `transformation_preview` | U4-13 |
| `imputation` | synthesis | `imputation_strategy_suggestions` | U4-13 |
| `target_leakage` | analysis | new: feature→target leakage (F-26) | U4-14 |
| `mutual_information` | analysis (`ml`) | `compute_mutual_information` | U4-14 |
| `quality_dimensions`, `ml_readiness` | synthesis | `data_quality_scorer`, `ml_readiness_scorer` | U4-15 |

The 49 superseded and 18 retired tasks account for all 67 registered legacy tasks.

### U4-00 · Plan Phase 4
- **Status:** done (2026-10-19) · **Size:** M · **Depends on:** U3-12 · **PR:** #100 (`0ef2161`)
- **Notes:** Phases 0–3 are now rows above, with their summaries and a 48-unit PR table in `docs/HISTORY.md`. Every finding in `docs/REVIEW_FINDINGS.md` has a dated status, checked against the code and probe runs. Phase 4 has 19 units, Phase 5 has `outline` units, and Phases 6–9 are sketched. A review before merging (2026-10-19) fixed U4-01's warning design, kept soft-read legacy inputs running, resized U4-02 and moved its per-check schema export to U5-03, and gave the text-heavy golden dataset to U4-09.
- **Deviations:** scope was widened by the user before the unit started. The user also chose to delete the legacy stack, including the old store, API v1 and the dashboard's backend, in Phase 4, with a stopgap CLI, and to retire the 13 profile-restating summaries. API v1's removal moved from Phase 6, and some Phase 5 carry-ins moved to Phase 4 or became moot.
- **Goal:** Break the Phase 4 batches into full units using Phase 3 learnings and golden-run evidence. Scope widened by the user on 2026-10-19: condense Phases 0–3 into the table above and `docs/HISTORY.md`; record a status for every finding in `docs/REVIEW_FINDINGS.md`, verified against the code and real runs; plan Phase 5 as `outline` units and sketch Phases 6–9.
- **Commit:** `docs: plan the check migration and record finding status`

### U4-01 · Engine performance baseline and warnings as notes
- **Status:** done (2026-10-20) · **Size:** S · **Depends on:** U4-00 · **PR:** #101 (`91c064d`)
- **Notes:** A check's warnings are `python_warning` notes with a count (`dsbf.engine.capture`, `Note.count`, schema 0.7.0), and titanic's default run prints nothing on stderr. #73's versions are in the lock, and the baseline, including the `engine-*` and new `standard-*` scenarios, comes from a perf run of this branch on the runner. Running checks in parallel was 3% slower on titanic and 7% slower on kitchen5k, so `dsbf.profile()` takes one worker for a plan of legacy tasks only. `standard-titanic` is 21.3 s; `standard-kitchen50k` is 266.7 s with a 2,318 MB peak.
- **Deviations:** `ignore` and `error` filters still hold during a run (the goal said every warning reaches the hook), and filters a library adds during a run are kept. The engine child's peak memory now reads `VmHWM`: `ru_maxrss` survives `exec`, so every engine scenario had reported the harness's 959 MB. scipy 1.17.1 doubles `missingness_mechanism_analysis` on kitchen50k (7.5 → 17.1 s); the baseline records that instead of the task being patched.
- **Note from U3-12:** record the `engine-*` perf baselines (serial and parallel, titanic and kitchen5k) from the CI runner with `--update-perf-baseline` before the first migration, so each Phase 4 unit is measured against them. Locally the parallel executor is slower than serial for the adapted tasks (titanic 24.7 s against 22.4 s, kitchen5k 120.3 s against 104.1 s), so use the CI figures to decide whether the default `workers` should be lower for adapted checks, and whether `fuzzy_duplicate_detection` (18.4 s of titanic's serial time, 74.3 s of kitchen5k's) should be `runs_in="process"`.
- **Note from U3-12:** Python warnings from legacy tasks (`infer_types` date parsing, `scipy` constant input, `statsmodels` KPSS) reach the caller's stderr during `dsbf.profile()`, because they are not log records.
- **Note from U4-00:** the `engine-*` scenarios arrived with #99, after the 2026-10-19 nightly; the first nightly perf artifact that measures them is 2026-10-20's, so promote from a nightly run on `main` rather than dispatching one. Under the default warning filters, Python shows an identical warning once per code location and remembers it for the life of the process (checked on 2026-10-19), so a hook on `warnings.showwarning` alone sees a repeat zero times, and a second `dsbf.profile()` in the same notebook sees nothing.
- **Note from U4-00:** take Dependabot PR #73 (scipy 1.16.3 → 1.17.1, platformdirs, fastapi; open since 2026-10-17, fast gate green, golden suite never run) in this unit, so the baselines are recorded on the locked versions. Its golden run (2026-10-19, run 39309478411) failed only on `wide`: one `one_way_anova` p-value (`cont_clean_11` by `cat_bal_19`) moves from 0.643672 to 0.643673 in the sixth decimal, and nothing else changes, as a local run under both scipy versions confirmed. Regenerate `snapshots/wide.json` with `--update-golden` and explain that value in the PR; the engine parity test compares with the same snapshot. Don't merge #73 on its own, or the nightly golden run on `main` fails.
- **Goal:**
  - Record the `engine-*` baselines from the CI runner.
  - Add `standard-titanic` and `standard-kitchen50k` scenarios: the default selection at the standard depth, with the Phase 4 budgets (under 5 s; under 60 s and 1 GB) reported but not enforced, so every migration reports its change (which rises until U4-15; see "How a migration works").
  - A warning raised inside a check becomes one `python_warning` note on that check per category and message, with a count, instead of reaching stderr. For the length of a run, the executor makes every warning reach the hook (an `always` filter set from the calling thread and restored after) and routes each one to the check whose thread raised it; a warning from any other thread goes to the original handler.
  - Decide the default workers for adapted checks and whether fuzzy duplicates run in a process.
- **Assumptions to confirm:**
  - `warnings.showwarning` runs in the thread that raised, so a thread-local "current check" can route a warning to its check. `warnings.catch_warnings` is not thread-safe and can't be used per check, and Python 3.12 and 3.13 have no context-aware warning filters, so the run-wide filter change is visible to the caller's other threads while a run lasts; document that in `dsbf.profile()`.
  - A process check can collect its warnings in the worker and return them with its result.
  - A baseline is promoted from the perf workflow's artifact as `tests/perf/README.md` describes (workflow dispatch, then `--update-perf-baseline` on the downloaded results).
- **Acceptance:**
  - `tests/perf/baseline.json` holds the `engine-*` and `standard-*` scenarios, recorded on `ubuntu-latest`.
  - A toy check that warns twice gets one note with count 2, at 1 and N workers and in a process, and nothing reaches stderr.
  - A second run in the same process records the same note again, and the caller's warning filters are as they were after the run.
  - titanic's default run prints no warning on stderr.
- **Commit:** `perf: record engine baselines and turn check warnings into notes`

### U4-02 · Core pack, migration harness and the first rule check
- **Status:** done (2026-10-20) · **Size:** L · **Depends on:** U4-01 · **Resolves:** F-11 (mechanism and regression test) · **PR:** #102 (`aa2703a`)
- **Notes:** `dsbf.checks.core_pack()` holds `zeros`, registered explicitly by `dsbf.profile()` and `dsbf plan`; the default packs are `core` and `legacy`. `dsbf.legacy.supersedes` maps a task to its core check: a replaced task sits in the `legacy_superseded` pack, so only a requirement (or a planned legacy reader of it, through a second planning pass) runs it, and then with its output and no findings. The parity harness, accepted-differences registry, core findings snapshots (`tests/golden/snapshots/core/`) and the Hypothesis adversarial test are in place for every later migration. `execute()` now derives `finished_at` from the monotonic clock.
- **Deviations:** the adapter can't use `PlanEntry.selected` to drop findings (a check sees only its context and parameters), so `LegacyParams.superseded_by` does it and is part of the cache key. `zeros` divides by the non-null values, where `detect_zeros` divided by all rows; no golden dataset has a numeric column with nulls and zeros, so the registry lists no entry for it. `_default_workers` stays 1 for legacy tasks beside core rule checks. The parity differences are all columns the profile types categorical that the legacy task read anyway, with their findings.
- **Note from U0-03:** the legacy guards only check that some column matches; tasks still loop over every numeric or object dtype column, not `matched_cols`, so they also analyse columns whose semantic type was excluded. Verified on 2026-10-19: a 0/1 integer column profiled `categorical` appears in nine continuous-only legacy checks' output.
- **Note from U4-01:** `execute()` takes `Run.started_at` and `finished_at` from the wall clock, and `Run` rejects `started_at > finished_at`, so a clock that steps back during a run makes `dsbf.profile()` raise (seen as flaky test failures on a WSL machine whose clock steps back 0.5 s every 32 s). Derive `finished_at` from `started_at` plus the monotonic elapsed time.
- **Note from U4-01:** two check conventions. A core check doesn't call `warnings.catch_warnings()`: the engine turns its warnings into notes, and while that block runs it silences every other running check's warnings. And it gives scipy numeric arrays, never object dtype: scipy 1.17 checks an object array for NaN one element at a time. Write both into the check-authoring rules.
- **Goal:**
  - Create `src/dsbf/checks/` holding the `core` pack. `dsbf.profile()` and `dsbf plan` register it explicitly, never on import.
  - Add the supersedes mapping and the default selection rule from "How a migration works", including the superseded tasks a remaining legacy task reads through `UNDECLARED_READS`. The mapping lives in `dsbf.legacy`, so the engine stays free of legacy knowledge and the mapping is deleted with it.
  - Build the parity harness: the golden datasets, the legacy task through the adapter and the core check through the engine, a comparator per check, and an accepted-differences registry with reasons that fails on drift.
  - Add core findings snapshots to the golden tier, normalised to id, check, code, subject, severity and key metrics.
  - Add an adversarial test over every core check: the adversarial fixture plus Hypothesis-generated frames (odd names such as `a|b`, `[/x]`, empty and unicode names; all-null columns; one row; mixed types; timezone-aware datetimes). Hypothesis is added with `uv add --dev` (approved by the user on 2026-10-19).
  - Migrate the pilot check, `zeros`.
- **Assumptions to confirm:**
  - How `dsbf.profile()` builds its registry and selection today (`library.py`), and where `core` joins.
  - Whether the adapter can drop the findings of a check that is in the plan only as a requirement (`PlanEntry.selected` is false).
  - What `tests/golden/engine_one.py` needs to run the core pack.
- **Acceptance:**
  - `zeros` runs on titanic and the adversarial frame.
  - The default selection runs `zeros` and not `detect_zeros`, while `data_quality_scorer` and `ml_readiness_scorer` still get `detect_zeros` as a requirement, with its findings dropped.
  - The parity test compares `zeros` with `detect_zeros` on the ten golden datasets, with the differences listed.
  - An integer-coded boolean column never appears in `zeros` output.
  - A drift in the core findings fails the golden tier.
  - With `detect_out_of_bounds` marked superseded in a test mapping, `data_quality_scorer` still receives its result, and its findings are dropped.
- **Commit:** `feat(checks): add the core check pack with a migration harness`

### U4-03 · Profile additions for rule checks
- **Status:** done (2026-10-20) · **Size:** S · **Depends on:** U4-02 · **PR:** #103 (`97c9d6d`)
- **Notes:** `ColumnProfile` gains `content_hash` (a row-order digest, same in every process, None for Object columns), `name_encoding_damage`, `values_with_replacement_char` and `values_with_nul`; categorical, boolean and text profiles gain `entropy_bits`. Schema 0.8.0. On the ten golden datasets entropy and the hash-and-dtype duplicate pairs equal the legacy outputs with no accepted difference. The pass costs 0.2 s more on kitchen50k and 0.4 s on wide1m.
- **Deviations:** `load` refuses a non-UTF-8 CSV (`SourceReadError`) under polars 2.0.0, so the UTF-16 acceptance check reads the file with `utf8-lossy` first. No per-column memory field was added: `dtype_optimizations` estimates from dtype widths and string lengths.
- **Note from U2-03:** `load` refuses a non-UTF-8 CSV with `SourceReadError` (checked 2026-10-20 on polars 2.0.0). The earlier note here said it decoded such a file lossily; that was wrong. U4-04 decided to keep refusing: a caller who reads with `utf8-lossy` gets U+FFFD and NUL characters in names and values, which `text_encoding` reports.
- **Goal:** Add what the U4-04 and U4-05 rule checks need and the profile lacks:
  - Shannon entropy over the full value distribution for categorical, boolean and text columns;
  - a per-column content hash, so equal hashes mark duplicate-column candidates;
  - counts of values containing U+FFFD or NUL, and a flag on names that contain them.

  Bump `SCHEMA_VERSION` (minor) and update `docs/parity/profile.md`.
- **Assumptions to confirm:**
  - Every rule check in U4-04 and U4-05 maps to profile fields; list the mapping in `PLAN.md`. A check whose needs the profile can't meet becomes an analysis check.
  - Polars' column hash is deterministic within a process, which is all a duplicate-column comparison needs. The profile digest in the cache key changes with it, as it does for any profile change.
- **Acceptance:**
  - `profile-kitchen50k` and `profile-wide1m` stay within budget (under 2 s; memory ratio under 2.5).
  - The generated duplicate columns in kitchen5k have equal hashes.
  - A UTF-16 CSV shows non-zero U+FFFD and NUL counts.
- **Commit:** `feat(profile): add entropy, content hashes and encoding damage counts`

### U4-04 · Rule checks I: column health and declarative tips
- **Status:** done (2026-10-20) · **Size:** M · **Depends on:** U4-03 · **Resolves:** F-31 (`eval`), F-11 and F-07 for these checks · **PR:** #104 (`56a24d4`)
- **Notes:** the `core` pack holds ten checks. Nine read the profile only (about 24 ms together on the kitchen50k profile): `null_rate`, `constant_columns`, `high_cardinality`, `id_columns`, `dominant_value`, `near_zero_variance`, `duplicate_rows`, `duplicate_columns` and `text_encoding`. Tips are declarative `{field, op, value}` conditions (`any`/`all` for the one compound tip), in `engine/tips.py`: validated when the library loads, attached after a check returns as `Action`s in `Guidance(phase="ml")` (no schema change), a failed lookup is a note, and the library digest is in the cache key. No `eval` remains in `src/`. `SUPERSEDED_BY` holds nine more tasks; parity comparators are keyed by task (`tests/golden/migration/column_health.py`), and a test maps every legacy finding title to a core code. `content_hash` became a `ProfileFeature`.
- **Deviations:** `duplicate_columns` decides on the 64-bit digest plus `physical_dtype`, not a comparison of values (a rule check cannot read values), and reports one finding per group of equal columns where the legacy task reported pairs. `likely_identifier` is a warning, as the strongest legacy level. `constant_columns` also reports a column with no values. `summarize_numeric` stays unsuperseded until U4-05. The extra advice the legacy tasks gave (impute, indicator, stratified split, class weight, target encoding, zero handling) is now tips.
- **Note from U0-07:** `get_recommendation_tip` (`utils/reco_engine.py`) still runs `eval` on the YAML `condition` strings (F-31). Replace it with a declarative condition format (`{"field": ..., "op": ">", "value": ...}`) when the tasks that call it migrate.
- **Note from U2-05:** a categorical column with mostly unique strings lists first-seen one-off values as top values (text columns skip values seen once). Decide, with the checks that consume top values, whether categorical should also require a repeat.
- **Note from U4-00:** an integer row number (`0..n-1`) is profiled `numeric`, not `identifier` (checked on 2026-10-19). `id_columns` decides whether such a column is an identifier.
- **Note from U4-02:** a numeric column with two values, or at most 20 under 5% of the rows, is typed `categorical`, so `zeros` no longer reports sparse small-integer counts such as titanic's `sibsp` (68% zero) and `parch` (76%), which `detect_zeros` did. Decide whether `dominant_value` or `near_zero_variance` covers them. Also, `detect_zeros`'s advice (log1p, an `is_zero` indicator, replace with NaN) is not in the core findings; it comes back with the tips, so attach it to `zeros`' `high_zero_rate`.
- **Note from U4-03:** the profile now has `content_hash`, `entropy_bits` and the encoding counts. `duplicate_columns` treats equal `content_hash` plus equal `physical_dtype` as candidates and confirms them by comparing values (the hash is 64 bits). `load` refuses a non-UTF-8 CSV with `SourceReadError` (checked 2026-10-20, polars 2.0.0), where the notes above say it decodes lossily: decide whether to keep refusing or read with `utf8-lossy` so `text_encoding` can report the damage, and correct the U2-03 note either way. `dominant_value` skips a column with no listed top value (strings over 256 characters, or text values seen once) rather than reading it as 0%.
- **Note from U4-02:** the golden datasets hold one zero-heavy continuous column (kitchen5k's `sparse_column`), so `zeros`' parity rests mostly on unit tests; add a zero-inflated continuous column to a generator if this unit needs golden coverage for it. The migration harness is in `tests/golden/migration/` (a comparator per check, `accepted.py` for the differences, `SUPERSEDED_BY` in `dsbf.legacy.supersedes`); this unit's checks go through it.
- **Goal:** Profile-only rule checks (cost cheap): `null_rate`, `constant_columns`, `high_cardinality`, `id_columns`, `dominant_value`, `near_zero_variance`, `duplicate_rows`, `duplicate_columns` and `text_encoding`. A declarative recommendation library: conditions as `{field, op, value}`, evaluated after a check returns, with a failed lookup recorded as a note.
- **Assumptions to confirm:**
  - Every condition in `recommendation_library.yaml` fits `{field, op, value}`; list any that don't.
  - Where a tip lives on a `Finding`, since the contract has no tip field: an `Action`, or an additive schema change.
- **Acceptance:**
  - Each check matches its legacy task, or lists its differences.
  - The nine checks take under 50 ms together on the kitchen50k profile.
  - No `eval` remains in the code.
  - Every finding these tasks emitted on the golden datasets maps to a core finding code, or is dropped with a reason.
- **Commit:** `feat(checks): add column health rule checks with declarative tips`

### U4-05 · Rule checks II: distribution shape, types and data contracts
- **Status:** done (2026-10-21) · **Size:** M · **Depends on:** U4-04 · **Resolves:** F-07 (`schema_validation` keys) · **PR:** #105 (`9f5c5ea`)
- **Notes:** the `core` pack holds 17 checks. `skewness` (with `mean_median_gap`), `kurtosis`, `entropy`, `dtype_optimizations`, `sample_size` and `class_imbalance` read the profile only; `schema` is an `analysis` check (cheap) because it counts values out of range or outside an allowed set. `dsbf.profile(target=...)` now sets the `target` parameter of every selected check that declares one. `SUPERSEDED_BY` maps a task to a tuple of checks (`summarize_numeric` is replaced by `near_zero_variance` and `skewness`, and only when both are selected); `RETIRED_TASKS` are the ten profile-restating tasks that raised no finding, replaced by the profile whenever `core` is selected. 90 differences are accepted on the golden datasets, each with a reason; the profile's bias-corrected skewness is larger than the task's by sqrt(n(n−1))/(n−2) (0.2% on titanic's `age`, 6% on `tiny`).
- **Deviations:** `entropy` also reads boolean columns, as the task did. The float32 suggestion is dropped, since the profile cannot show float32 holds a column's precision. `dtype_optimizations` narrows integer codes the profile types categorical when it lists every value, so it finds savings on titanic's `pclass`, `sibsp` and `parch` that the task missed. `schema` replaces the keys `required_columns`, `dtypes`, `value_ranges` and `categories` with `columns`, a mapping of name to expectation (an unknown key is a planning error); `mixed_type_columns` is left to `mixed_types` (U4-09). `class_imbalance` without a target and `schema` without expectations end `not_applicable`, where the tasks ended `skipped`.
- **Goal:** Rule checks `skewness`, `kurtosis`, `entropy`, `dtype_optimizations`, `sample_size`, `class_imbalance` (on the `target` setting) and `schema`. `schema` takes the expected columns, types, nullability, ranges and allowed values as params, and is `not_applicable` without them. Supersede the 13 retired summary tasks.
- **Assumptions to confirm:**
  - How the profile's skewness and kurtosis estimators differ from scipy's defaults in the legacy tasks; that is either an accepted difference or a parameter.
  - That every guidance blurb of the retired tasks is covered by a core check, or dropped with a reason.
- **Acceptance:**
  - Parity, or listed differences.
  - A test maps every legacy finding code from the retired tasks on the golden datasets to a core code, or to "dropped" with a reason.
  - `dsbf.profile(target=...)` drives `class_imbalance`.
- **Commit:** `feat(checks): add distribution, type and schema rule checks`

### U4-06 · Analysis checks I: distributions
- **Status:** done (2026-10-21) · **Size:** M · **Depends on:** U4-05 · **Resolves:** N-A3 (data side) · **PR:** #108 (`63c2bbe`)
- **Notes:** the `core` pack holds 22 checks. Five `analysis` checks read `ctx.values`: `outliers` (IQR, z-score and MAD, with consensus at two rules), `multivariate_outliers` (isolation forest, `ml`), `normality` (Shapiro–Wilk to 5000 values, Kolmogorov–Smirnov above, Jarque–Bera always, and Q-Q quantiles for the columns that fail), `bimodality` (Gaussian mixtures, `ml`) and `out_of_bounds` (caller bounds). The forest is its own check because the engine marks `unavailable` per check; without `ml` it and `bimodality` are `unavailable` and `outliers` still runs. `SUPERSEDED_BY` gains five tasks (`detect_outliers` is replaced by two checks), and 169 differences from them are accepted on the golden datasets in `tests/golden/migration/accepted_analysis.py`, each by reason.
- **Deviations:**
  - The `scikit-learn <1.8` note was a different bug: scikit-learn mis-scores **integer** input to `GaussianMixture` (log-likelihood too high; 1.7.2 and 1.8.0 differently), so the legacy task's BICs for integer columns were wrong and moved between versions. A float64 fit is correct and the same on both. `bimodality` always fits float64.
  - `bimodality` decides on ΔBIC, Ashman's D between the components, each weight, and a dip in the histogram between the peaks; the task's relative test (ΔBIC / |BIC₁|) changes with the column's units and is dropped. The dip is needed because two Gaussians fit a uniform column better than one with the peaks far apart: kitchen5k's `sequential_id`, `normal_numeric_int`, `negative_counts` and five others were bimodal without it. On the golden datasets the check flags 2 columns (kitchen5k's `bimodal_column` and `subtle_leakage`) where the task flagged 38: it reads 6 of the others as integer codes, 22 are one skewed peak (20 are wide's `cont_skew_*`) and 8 are flat.
  - `out_of_bounds` takes no default bounds. The task's ranges by column name flagged nothing on any golden dataset (`test_the_default_bounds_of_the_legacy_task_flagged_nothing`), and a column's name doesn't say what range it should hold. It ends `not_applicable` without `bounds`, like `schema`.
  - `outliers` states its share over the finite values (the task used all rows) and drops the row indices from the output; the ten most distant flagged values stay.
  - A column whose arithmetic overflows (values near 1e308) is skipped with the reason in `skipped`, never a NaN in the output; the adversarial test found this on `normality`. The forest rescales each column to 0..1 (which changes nothing it does) so that values past single precision, as in kitchen50k, don't disable it. `bimodality` samples at 10 000 values and the forest at 100 000, recorded in the evidence. The forest's example rows are ranked by anomaly score rounded to nine places, then by position, because rows tie and the order differed between machines (the first CI golden run on kitchen5k).
  - The Kolmogorov–Smirnov p-value stays conservative, as in the task (the sample sets the normal, so no Lilliefors correction); the evidence says so.
- **Run time:** the five checks take 2.3 s together on kitchen50k and 14.9 s on 1M × 50 floats, run alone with one worker. `dsbf.profile()` now runs one check at a time while any legacy task is planned (it did so only beside rule checks): with the analysis checks it ran the legacy tasks in parallel threads, and each task's own wall time grew many times over on the CI runner (`kruskal_wallis` 0.05 s → 1.7 s on titanic) and failed the perf comparison. In the `standard-*` scenarios the checks then take 0.08, 0.59, 0.63 and 1.06 s (`outliers`, `normality`, `bimodality`, `multivariate_outliers`) on kitchen50k, beside the legacy tasks they replace, which stay until U4-15. On this WSL machine, not the CI runner, titanic takes 24.2 s and kitchen50k 430 s against the CI baselines of 21.3 s and 266.7 s; the drop is measured at U4-15.
- **Note from U1-05:** the `scipy <1.18` cap: write `normality` without the positional `kstest(..., args=...)` call that 1.18 breaks.
- **Note from U0-16:** when bimodality is migrated, decide how the BIC comparison is reported. The absolute BIC and `delta / abs(bic1)` both depend on scikit-learn's log-likelihood convention; the delta itself did not move.
- **Goal:** `outliers` (IQR, z-score and MAD, plus isolation forest behind `ml`), `normality` (the legacy tests, with Q-Q quantiles in its output), `bimodality` (Gaussian mixture BIC behind `ml`) and `out_of_bounds` (user bounds, counted from values). Each reads `ctx.values(col)` and records `n_used`, with row caps recorded in the evidence.
- **Assumptions to confirm:**
  - The new calls work on scipy 1.16 and 1.18 and on scikit-learn 1.7 and 1.8 (`uv run --with`).
  - The mixture fit is deterministic with a seed.
  - Isolation forest at 1M rows needs a sample cap.
- **Acceptance:**
  - Parity, or listed differences.
  - Without `ml`, `bimodality` and the isolation-forest method report `unavailable`.
  - The tests pass with scipy 1.18 and scikit-learn 1.8.
- **Commit:** `feat(checks): add outlier, normality, bimodality and bounds checks`

### U4-07 · Analysis checks II: pairwise associations
- **Status:** done (2026-10-22) · **Size:** M · **Depends on:** U4-06 · **Resolves:** F-13, F-10 (pair subjects), F-26 (redundancy part), REVIEW_FINDINGS part 2 C (intervals, Cramér's V, comparable scale, multiple comparisons) · **PR:** #109 (`a6641b7`)
- **Notes:** the `core` pack holds 24 checks. `associations` tests numeric-by-numeric and numeric-by-categorical pairs and `contingency` categorical-by-categorical ones, so the legacy `compute_pairwise_associations` is replaced only when both are selected; `detect_data_leakage` is replaced by `associations` and `contingency_tables` by `contingency`. Routing happens after the `min_n` gate (8), so a 20-row frame gets Kendall's τ-b (F-13). Every pair is a `Subject.column_pair` with `first` and `second` fields (F-10); η replaces η² and point-biserial r so r, ρ, τ, η and V̄ share a 0–1 scale; Fisher z intervals cover r, ρ and τ, a seeded percentile bootstrap covers Bergsma's V̄; each output and finding states `pairs_tested` next to `pairs_flagged`; `redundant_columns` (|Pearson r| ≥ 0.99, a warning) replaces the leakage scan's feature–feature half. 376 differences are accepted on the golden datasets in `tests/golden/migration/accepted_associations.py`, by eight reasons.
- **Deviations:**
  - The legacy `both` method is gone; every numeric pair carries `pearson_r` beside its statistic.
  - A categorical column over 50 levels is skipped (the task applied that limit only to categorical pairs, so η² against a 31 000-level column inflated toward 1); a group of one row is left out of η.
  - Chi-squared, V and p are Pearson's, not scipy's Yates-corrected values for 2×2 tables (the bulk of the accepted differences); the table is tested whole and only its top levels are printed, where the task cut it before testing, which skipped kitchen5k's `future_leakage` pairs.
  - `max_pairs` (10 000 for `associations`, 5 000 for `contingency`) ends a check `skipped` with a hint rather than test part of the pairs; `max_rows` (200 000) samples rows for `associations`. `compute_pairwise_associations`'s `__correlation_matrix__` is not carried over: the output is the tested pairs, and a matrix is a view Phase 6 builds.
- **Run time:** on kitchen50k `associations` takes 0.67 s (561 pairs) and `contingency` 0.20 s (120 pairs), against the task's 33.6 s on the CI runner (CI, `standard-kitchen50k`: 266.7 s to 235.7 s with the three tasks gone and the two checks added, `engine-titanic-parallel` 22.1 s to 21.0 s, no total above its baseline). The per-core `engine-*-parallel` scenarios failed on the first CI run because two superseded legacy tasks took longer by wall time while they ran beside different tasks (`detect_outliers` 2.0 s to 4.4 s on titanic, `detect_bimodal_distribution` 1.1 s to 2.9 s): the same four runs gave 3.8–5.0 s and 2.3–3.0 s, and a local A/B on 4 cores (`taskset`) showed no slowdown from the two checks, so the baseline was re-recorded from one of those CI runs, as `tests/perf/README.md` describes. The overlap explanation is not isolated; U4-15, which takes the legacy tasks out, measures the drop. On 1M × 50 floats `associations` took 1.7 s and 6.5 s in two runs on this WSL machine, sampled to 200 000 rows.
- **Note from U0-09:** the leakage correlation scan uses `select_dtypes` (F-11).
- **Note from U4-01:** on the runner, `compute_pairwise_associations` takes 33.6 s of `standard-kitchen50k`'s 266.7 s (one worker, `tests/perf/baseline.json`).
- **Goal:**
  - `associations`: Pearson, Spearman and Kendall routing, with the small-sample Kendall's τ reachable (route before the gate); η for numeric–categorical pairs, so strengths compare on one scale; Fisher z intervals for r and τ; `n_used` per pair; pairs as `Subject.column_pair`; pairs tested next to pairs flagged; and near-duplicate features (|r| ≥ 0.99) as `redundancy` findings, which supersedes `detect_data_leakage`'s scan.
  - `contingency`: bias-corrected Cramér's V (Bergsma 2013) with a bootstrap interval and a category cap.
- **Assumptions to confirm:**
  - The pair cap and cost class: the legacy task took 33.9 s on kitchen50k in the CI baseline.
  - Where the correlation matrix goes: in the output, or computed on demand in Phase 6.
- **Acceptance:**
  - A 20-row frame gets Kendall's τ.
  - Columns named `a|b` and `c|d` get correct subjects.
  - On small samples the corrected V is below the uncorrected one.
  - Intended differences are documented, and the time at kitchen50k is recorded.
- **Commit:** `feat(checks): add association and contingency checks with intervals`

### U4-08 · Analysis checks III: group differences and collinearity
- **Status:** done (2026-10-22) · **Size:** M · **Depends on:** U4-07 · **Resolves:** F-32 (VIF part) · **PR:** #110 (`5bb5a65`)
- **Notes:** the `core` pack holds 26 checks. `group_differences` replaces `one_way_anova`, `kruskal_wallis` and `mann_whitney_u` together (it reads a numeric column against each categorical one of 2 to 20 levels) and `collinearity` replaces `detect_collinear_features`; the legacy VIF task keeps running beside it because `data_quality_scorer` reads it (`UNDECLARED_READS`) until U4-15. Each pair gets the classic ANOVA F with ω² (η² beside it) and the tie-corrected Kruskal–Wallis H with ε² = H / (n − 1); each method's p-values are adjusted over every pair (`fdr_bh` by default), and the output and each finding state `pairs_tested`, `pairs_flagged` and `tests_run`. A pair is flagged when either adjusted p is under α *and* its effect reaches `min_effect` (0.01), so a huge sample's negligible difference is tested but not reported. A flagged pair gets stratified percentile bootstrap intervals (200 resamples, the rank effect on pooled ranks) and every pairwise Mann–Whitney contrast of its groups (up to 10 groups), Holm-adjusted, with a signed rank-biserial r (positive when the first level is higher) and an interval for a significant contrast. The finding says when the methods disagree, and when the group spreads differ by more than 2× (classic ANOVA assumes they don't). `collinearity` uses the centred VIF, the diagonal of the inverse correlation matrix from `numpy.linalg.eigh` (a floor on the eigenvalues, a VIF capped at 1e6 and flagged `capped`), so it needs no statsmodels. Missing data (F-32): a column over 40% missing is left out first, constant columns are left out, rows with a missing value in any remaining column are dropped, and while the complete rows number fewer than max(k + 1, `min_rows` 10) the sparsest column is dropped too; `n_used`, `skipped_columns` and `dropped_for_rows` are in the output, `n_used` and the method (`vif_centred`) on every finding, and each flagged column names the two columns it correlates with most. Tips attach drop, regularise and PCA advice to a VIF of 10 or more (20 for PCA) and investigate / encode advice to a group difference. The overflow-safe centring and the p-value adjustment moved to `checks/pairs.py`.
- **Deviations:**
  - The F, H, p, n and significance of every pair `one_way_anova` and `kruskal_wallis` tested match the check's on all ten datasets: their only accepted differences are the datasets with no numeric–categorical pair, which the check ends `not_applicable`.
  - The VIF differs from the legacy task's by definition, not rounding. The legacy fit had no intercept, so a column with a large mean scored high from its mean alone: on `clean`, the four independent columns scored 33.5, 26.7, 24.6 and 11.3 and now score 1.000 to 1.001 (the four columns the statsmodels note names stop being flagged, and are right to). On kitchen5k the legacy fit was singular (a constant column beside exact copies) and returned values below 1; the check reports the copies at VIF 900 and more. A VIF of 20 or more is a warning, where the task called it an error. 212 differences are accepted in `tests/golden/migration/accepted_group_differences.py`, by thirteen reasons.
  - `mann_whitney_u` is no longer a test of every first-ten level pair: it is a post-hoc contrast of a flagged pair, and its rank-biserial sign is corrected (the task's `1 − 2U/(n₁n₂)` was negative when the first group was higher, and its guidance said the reverse). A contrast is not made for a pair the omnibus tests do not flag, so none of `wide`'s 10 948 tests and 1 162 of kitchen5k's are repeated.
  - A categorical column over 20 levels is skipped with its reason (the legacy cardinality limits were 20 for the omnibus tests and 10 for Mann–Whitney); a group of fewer than `min_group_n` (5) rows is left out and counted.
- **Run time:** on a 52 500 × 60 frame `group_differences` takes 2.5 s (330 pairs, 664 tests, 5 pairs bootstrapped) against 14.6 s for the three tasks on the CI runner (U4-01), and `collinearity` under 0.1 s (22 columns, 52 500 complete rows). The `standard-*` scenarios still carry the legacy tasks, which U4-15 takes out. The whole process (CSV load, profile and the core pack) peaked at 550 MB locally.
- **Note from U4-01:** the three tasks took 14.6 s of `standard-kitchen50k`, and on kitchen5k `mann_whitney_u` went from 2.1 s serial to 64.2 s per core.
- **Goal:**
  - `group_differences`: one check with ANOVA, Kruskal–Wallis and Mann–Whitney as methods, effect sizes with intervals, and the number of tests run.
  - `collinearity`: VIF per column under its own missing-data strategy (drop sparse columns first, then rows), `n_used`, and a guard on rows below k + 1.
- **Acceptance:**
  - Parity, or listed differences, explaining the four `clean` columns the statsmodels note names.
  - The VIF definition and `n_used` are in the evidence.
- **Commit:** `feat(checks): add group difference and collinearity checks`

### U4-09 · Analysis checks IV: missingness and string structure
- **Status:** done (2026-10-22) · **Size:** M · **Depends on:** U4-08 · **PR:** #111 (`fb07cdb`)
- **Notes:** the `core` pack holds 31 checks. `missingness_mechanism` tests the null indicator of each column with at least 1% (and 8) nulls against every other column on the rows where that column has a value: numbers and datetimes by Mann–Whitney (one rank per predictor, each column's rank sum from a product with the indicators; signed rank-biserial r), levels by chi-squared with bias-corrected Cramér's V, each test stating `n_missing` and `n_observed`. All p-values are adjusted together (`fdr_bh`), and a test is flagged at an adjusted p under α with an effect of at least 0.1; the output and each finding state `tests_run` and `tests_flagged`, and each column lists the columns it is null with from the profile's pairs. A column is `depends_on_observed`, `no_dependence_found` or `untestable`, and the finding says that a dependence cannot tell MAR from MNAR. `mixed_types` classifies the distinct values of a string column as number, boolean, date, text or null token (`ignore_null_tokens` is a real switch) and reads `CheckContext.coerced` for pandas columns of mixed Python types. `encoded_columns` finds UUIDs, hex digests, base64 and random tokens by format, never a plain word. `format_violations` takes a list of `{column, pattern, mode}` rules (a pattern that does not compile is a planning error) and reports dates written in several formats, with their pattern signatures. `string_anomalies` finds case variants, edge whitespace, invisible characters and length outliers, counted in rows. `profiling/types.py` now types near-unique prose `text` (over 30 characters on average, mostly several words) before the distinct-share test, so a message column is no longer a key. `tests/golden/data/SMSSpamCollection` (UCI, CC BY 4.0) is the text-heavy golden dataset, `sms_spam`, with `GoldenDataset.core_params` giving `format_violations` its two rules.
- **Deviations:**
  - The task's "Little's MCAR test" is not carried over: it was neither Little's statistic nor a χ² (a diagonal-variance sum), so there is no dataset-level verdict. On `wide`, whose nulls are completely random, the task called all eight null columns MAR from 67 unadjusted tests each; the check runs 784 tests, adjusts them together and flags none. It also reads categorical predictors, which the task never did (`high_missingness`: `channel` V = 0.89, `plan_type` V = 0.85; kitchen5k: `control_character_text`).
  - The task called every plain word base64 (`sex`, `embarked`, `class` on titanic; each categorical column of five other datasets); `encoded_columns` finds none there, and finds kitchen5k's two UUID columns and base64 column, which the task skipped as identifiers. `mixed_types` finds kitchen5k's three mixed columns by content, where the task, handed a CSV of strings, finds nothing in the engine's run. 171 differences of the five tasks are accepted in `tests/golden/migration/accepted_missingness_strings.py` by fifteen reasons, and nine of earlier tasks on `sms_spam` in their own files.
  - The profile parity gained one difference (`prose-is-text-not-a-key`) and `sms_spam`'s stage.
- **Run time:** on a 52 500 × 60 frame the checks take 0.33 s (`missingness_mechanism`), 0.55, 0.59, 0.55 and 0.59 s (`mixed_types`, `encoded_columns`, `format_violations`, `string_anomalies`); on 1 000 000 × 37 with 10 columns of 20% nulls, 2.4, 1.4, 2.1, 1.2 and 1.2 s. The date formats took 3.2 s each until they were tried on a sample of the plausible values. The `standard-*` scenarios still carry the legacy tasks, which U4-15 takes out.
- **Goal:**
  - `missingness_mechanism` (built on the profile's missingness pairs, with n per test), `mixed_types` (its null handling can be switched off, which F-07 found impossible), `encoded_columns`, `format_violations` (regex rules plus mixed date formats from pattern signatures) and `string_anomalies`.
  - Add a text-heavy golden dataset whose license allows vendoring, with its legacy and core snapshots.
- **Acceptance:**
  - The text-heavy golden dataset exercises `format_violations` and `string_anomalies`.
  - Parity, or listed differences.
  - `inconsistent_datetime` is reported as mixed formats.
- **Commit:** `feat(checks): add missingness mechanism and string structure checks`

### U4-10 · Fuzzy duplicates redesigned
- **Status:** done (2026-10-22) · **Size:** M · **Depends on:** U4-09 · **Resolves:** F-35 · **PR:** #115 (`05de0dd`)
- **Notes:** the `core` pack holds 32 checks. `fuzzy_duplicates` (heavy, `text` extra, `runs_in="process"`, so it runs at the deep depth and the standard depth no longer reports fuzzy duplicates) compares distinct records of the string columns that have more than 20 values (labels are left out; identifier-typed columns are compared, `exclude_columns` drops keys) by the mean of the per-column Indel ratios weighted by each column's entropy in bits, threshold 0.92. Every pair is scored when `max_pairs` (2 000 000) holds them all; otherwise pairs that share a rare token (2 to 50 records), rarest first, deduplicated and sorted, so no hash order enters. The output states `pairs_possible`, `pairs_compared`, `coverage` and `stopped_by` (`rows`, `pairs`, `seconds`), lists the 20 closest pairs and counts the rest; a partial comparison adds a note. On titanic, which has no text column, it ends `not_applicable` and the `Run` is 0.57 MB (8.9 MB with the task); 50 of 50 planted copies are found at the default budgets and the closest unrelated pair scores 0.73 against 0.988 for the planted. `rapidfuzz` joined the `dev` group.
- **Deviations:** the score is a weighted column mean, not the whole row string (one shared 1 850-character field made unrelated rows 0.95 alike). Golden `kitchen5k` does not carry the planted rows, since that changes its row count and invalidates hundreds of accepted differences that hold exact numbers; `DSBFTestDataGenerator(near_duplicates=N)` provides them to `tests/test_checks/test_fuzzy_recall.py`. The hash-seed pin stays in the golden runner and perf harness, because the legacy task still runs there until U4-18. `engine_differences.json` needed no change. The first process-run check found that a worker renamed a column called `""`; fixed in `engine/process.py`. The task's 13 800 titanic pairs (30.5 s locally) and 5 on kitchen5k (178.6 s) were label coincidences and exact repeats; five reasons for the eleven datasets are in `accepted_fuzzy_duplicates.py`.
- **Run time:** `fuzzy_duplicates` takes 1.0 s of check time on kitchen5k with the planted rows (8.6 s for the whole deep run) and 1.1 s on titanic (including starting the worker); an exhaustive run of all 12.7 M pairs of kitchen5k takes 66 s.
- **Note from U3-11:** `fuzzy_duplicate_detection` puts its whole pair list in its output, 8.46 MB of an 8.93 MB titanic `Run` (2026-10-19).
- **Note from U0-13:** it builds candidate pairs from a token index over sets of strings and stops at `max_candidates`, so which pairs it keeps (and the count) follows `PYTHONHASHSEED`. Make the order deterministic, so the golden runner no longer needs the pinned seed.
- **Note from U4-01:** declare `fuzzy_duplicates` `runs_in="process"`. On the runner the legacy task is 84% of titanic's serial run (18.0 of 21.4 s) and 75% of kitchen5k's (68.4 of 90.6 s), and 71.5 s of `standard-kitchen50k`. Being pure-Python `difflib`, it holds the interpreter lock, so on threads it slows the checks beside it. An adapted task can't run in a process, because it's built at run time. Without a pinned seed, titanic's `Run` came out at 4.7, 5.3 and 9.8 MB in three processes.
- **Goal:**
  - rapidfuzz (`text` extra) on row strings extracted once;
  - blocking on rare tokens (IDF-weighted or MinHash/LSH) that leaves low-cardinality categoricals out of the row string;
  - a deterministic candidate order;
  - coverage (`pairs_compared / pairs_possible`) and budgets (rows, pairs, seconds) recorded in the evidence;
  - the top N pairs plus counts in the output, with the bulk behind `details_ref` once Phase 6 has an artifact store.

  Add planted near-duplicate rows to the kitchen-sink generator (it has none) and calibrate the threshold on them. Use `runs_in` as U4-01 decided.
- **Acceptance:**
  - The check's output on titanic is under 100 KB.
  - Results are equal across `PYTHONHASHSEED` values.
  - Recall on the planted near-duplicates is recorded.
  - Time at titanic and kitchen5k is within a budget set from U4-01's figures.
- **Commit:** `feat(checks): redesign fuzzy duplicate detection with blocking and budgets`

### U4-11 · Time-series checks redesigned
- **Status:** done (2026-10-22) · **Size:** L · **Depends on:** U4-10 · **Resolves:** F-39, N-A4 (data side) · **PR:** #116 (`cddf99d`)
- **Notes:** the `core` pack holds 38 checks. `datetime_consistency`, `temporal_gaps` and `datetime_features` are rule checks on `DatetimeProfile`, which gained `format`, `epoch_unit`, `unparsed`, `time_of_day` and `distinct_years`, `distinct_months`, `distinct_weekdays` and `distinct_hours` (result schema 0.9.0). `stationarity` (ADF and KPSS, `max_lag` 24), `autocorrelation` (ACF and PACF, `max_lags` 40, a lag flagged only outside its band and at 0.2 or more) and `seasonal_decomposition` (robust STL; the period is one cycle of the next longer natural unit, or `period`) share `checks/timeseries.py`: the series is resampled to the configured or profile-inferred frequency, repeated timestamps are aggregated (`mean` by default), up to 20% empty periods are interpolated, and only the most recent 2 000 periods are used. The frequency and source, aggregate, series length, periods filled and cut, and lag cap are in every finding's evidence. `dsbf.profile(time_index=...)` reaches every core check with a `time_index` parameter, as `target` does. The six tasks are in `SUPERSEDED_BY`, with comparators and 66 accepted differences in `accepted_time_series.py`.
- **Deviations:** the rule checks needed profile fields the profile lacked (the unit assumed it had them), so the schema went up. `seasonal_decomposition` keeps strengths, not the component series (the task wrote five arrays a column); the Phase 6 store keeps them. A text column is typed datetime only if a format reads over 90% of it, so `datetime_consistency` can never see a share of 10% or more unparsed. `datetime_features` lists a part only if it varies; the task's test was always true. The test plugin of `test_cli_plan` renamed its `stationarity` and `date_gaps` checks, which the core pack now owns.
- **Run time:** on kitchen50k (22 numeric columns, 365 daily periods) `stationarity` takes 0.28 s, `autocorrelation` 0.21 s and `seasonal_decomposition` 0.43 s, and the rule checks 0.001 s each, against 89 s for `detect_stationarity` in the CI baseline. The worst case the caps allow, 52 500 rows over 3 000 days and 40 columns of 2 000 periods, takes 1.3 s, 0.5 s and 3.4 s. The `standard-*` scenarios still carry the legacy tasks, which U4-15 takes out.
- **Note from U0-13:** on kitchen5k, `decompose_time_series` cannot infer a seasonal period (frequency `None` for a daily index with repeated dates), and the golden baseline records that.
- **Note from U4-01:** `detect_stationarity` wraps calls in `warnings.catch_warnings()` with `simplefilter("ignore")`, which silences every other running check's warnings while it lasts; the core checks don't. It is the slowest check of `standard-kitchen50k` (83.6 s on the runner).
- **Goal:**
  - `temporal_gaps`, `datetime_consistency` and `datetime_features` as rule checks on the datetime profile.
  - `stationarity` (ADF and KPSS), `autocorrelation` (ACF and PACF) and `seasonal_decomposition` on the series resampled to the inferred or configured frequency, with repeated timestamps aggregated, and with capped `maxlag` and series length. All three are recorded in the evidence. These checks are behind `timeseries` and take the index from `dsbf.profile(time_index=...)`.
- **Acceptance:**
  - `stationarity` takes under 2 s at kitchen50k (89 s for the legacy task in the CI baseline).
  - The frequency, `maxlag` and series length are in the evidence.
  - Parity, or listed differences.
- **Commit:** `feat(checks): redesign time-series checks with resampling and capped lags`

### U4-12 · Drift against a reference
- **Status:** done (2026-10-23) · **Size:** M · **Depends on:** U4-11 · **Resolves:** F-07 (drift keys), F-30 (reference only CSV) · **PR:** #117 (`70fe926`)
- **Notes:** the `core` pack holds 39 checks. `dsbf.profile(reference=...)` takes any source `load` accepts, profiled with the same `sample`, or a `Run`, whose stored profile is used as it is; a `Reference` (profile, provenance, no values) reaches only checks that declare `Requires(reference=True)`, which are `not_applicable` (`no_reference`) without one, and the cache key covers it. `reference_drift` bins the current values on the reference's histogram merged into at most `bins` (10) bins that each hold a tenth of it, so a data reference and a `Run` take one path: PSI and JS divergence for numeric columns, TVD over the reference's listed values for categorical and boolean ones, plus added and dropped columns, a changed dtype family, null-share shifts and `target_drift` one severity step higher. The three tasks are in `SUPERSEDED_BY`; `reference_one.py` runs them and the check against a perturbed copy of each golden dataset, with 78 accepted differences, and 33 more cover the default runs without a reference.
- **Deviations:** the unit planned `min_bin_share`, `js_warn` and a KS statistic; PSI's limits assume about ten bins, so `bins` replaced the first and the other two were dropped (severity comes from PSI or TVD, JS divergence stays in the evidence). A changed type is a changed dtype family, not a changed semantic type, because the profile types a constant number `categorical` and an all-null column `unknown`. An unreadable reference raises `SourceError`. The legacy `detect_feature_drift` and `detect_target_drift` fail on the library's default pandas backend, and `detect_target_drift` raises on a categorical target whenever the null counts differ (`chisquare` needs equal totals); the parity run uses the Polars backend.
- **Goal:** `dsbf.profile(..., reference=...)` takes any source `load` accepts, or a `Run`. `reference_drift` compares the two: PSI or JS divergence on numeric distributions, TVD on categorical top values, type, schema and missingness changes, and target drift when `target` is set.
- **Assumptions to confirm:**
  - The two profiles' histograms have different edges, so decide between re-binning from values (the reference is data) and approximating from quantiles (the reference is a stored profile). Phase 7's history-based drift reuses the second.
  - The reference is profiled with the same settings and sampling as the data.
- **Acceptance:**
  - Identical data shows no drift, and a perturbed copy of kitchen5k flags the shifted columns.
  - Parity with the three legacy tasks, or listed differences.
- **Commit:** `feat(checks): compare a dataset with a reference`

### U4-13 · ML-prep guidance checks
- **Status:** done (2026-10-23) · **Size:** M · **Depends on:** U4-12 · **PR:** #118 (`2564726`)
- **Notes:** the `core` pack holds 43 checks. `categorical_encoding` (binary, one-hot or frequency from cardinality and rows per level) and `numeric_binning` (quantile or equal-width bins and a count, from skewness and the span of the middle 98% of values) are rule checks; `transformations` (analysis) tries log1p, sqrt, square, Box–Cox and Yeo–Johnson through scipy on a 20 000-row sample and recommends the simplest that brings |skewness| to 0.5, saying when none helps; `imputation` (synthesis) reads `null_rate`'s severity as the tier and `missingness_mechanism`'s assessment when it ran. Every finding is `phase: ml` with typed `Action`s (`encode`, `bin`, `transform`, `impute`, `add_indicator`, `drop`), and the four run without the `ml` extra. A synthesis check now sees every column of the profile (never values), because advice about a column needs the column's name and type. The four tasks are in `SUPERSEDED_BY`; 174 accepted differences with 16 reasons cover the eleven golden datasets.
- **Deviations:** the target-correlation upgrade ("+ target encoding") is dropped, since it correlated arbitrary integer codes with the target; no `hashing`, `group_rare` or `target_encoding` actions, because the profile lists 10 top values and cannot measure rare levels above that. The legacy `log-transform` branch of the binning task is `transformations`' advice. Text, identifier and unknown columns get no imputation advice. `ml_readiness_scorer` still hard-requires `suggest_categorical_encoding`, which runs beside `categorical_encoding` until U4-15.
- **Run time:** the four checks take 0.13 s together on kitchen50k (52 500 × 60), `transformations` reading 20 000-row samples. The `standard-*` scenarios still carry the legacy tasks, which U4-15 takes out.
- **Goal:** `categorical_encoding`, `numeric_binning`, `transformations` (Yeo–Johnson and Box–Cox through scipy, with the skew reduction measured) and `imputation`, a synthesis check reading `null_rate` and `missingness_mechanism`. Their findings are `phase: ml` with typed actions, which the Phase 9 pipeline export will read.
- **Assumptions to confirm:** the legacy tasks use scikit-learn only for `PowerTransformer` and `LabelEncoder`, and scipy and NumPy can replace both, so these checks need no extra.
- **Acceptance:**
  - Parity, or listed differences.
  - Every action validates against `Action`.
  - The checks run without the `ml` extra.
- **Commit:** `feat(checks): add encoding, binning, transformation and imputation guidance`

### U4-14 · Target-aware checks
- **Status:** done (2026-10-23) · **Size:** M · **Depends on:** U4-13 · **Resolves:** F-26 (target leakage), F-07 (one target setting) · **PR:** #119 (`7df24ea`)
- **Notes:** the `core` pack holds 45 checks. `target_leakage` measures each feature against the target by type pair (the larger of |Pearson r| and |Spearman ρ|; η; bias-corrected Cramér's V), and also flags a part of an identifier that spells the outcome (`derived_from_target`), a feature whose nulls follow the target (`missingness_tracks_target`) and a datetime later than `time_index` in 95% of rows (`timestamp_after_event`); warn from 0.95, error from 0.99, `drop` actions. `mutual_information` (`ml` extra, imported inside the estimator) uses one estimator call per feature on complete pairs, normalises to a share of the target's entropy or an information-based r², and keeps a shuffled-target noise floor. `categorical_encoding` 1.1.0 reads both through `after` and adds an out-of-fold `target` encode action for a frequency-encoded column that predicts a binary or numeric target and is not a flagged leak. `compute_mutual_information` is superseded. Shared code is `checks/target.py`; η moved to `pairs.correlation_ratio`.
- **Deviations:** the unit named kitchen5k's `leakage_*` columns, which are `perfect_leakage`, `future_leakage` and `subtle_leakage` with the target `target_for_leakage`, given to the core checks through `core_params` (the legacy tasks run without one). The kNN estimator reads copied rows and rounded values as dependence, so rows that repeat across two or more continuous columns are left out (a note says so) and ties are spread; one shuffle was no noise floor, so at least 12 are used. Class columns over 50 values get no MI and no target-encoding advice. The thresholds are uncalibrated.
- **Run time:** on 52 500 × 60 `target_leakage` takes 4.0 s (2 s cutting 14 identifier-like columns into parts) and `mutual_information` 1.1 s, alone with one worker. The `standard-*` scenarios run neither without a target, and still carry the legacy tasks, which U4-15 takes out.
- **Note from U4-13 (done):** `categorical_encoding` no longer suggests target encoding (the legacy upgrade correlated arbitrary integer codes with a numeric target). When `mutual_information` or `target_leakage` shows a high-cardinality categorical column predicts the target, let `categorical_encoding` read it through a soft `after` and add an out-of-fold `target` encoding action.
- **Goal:**
  - `target_leakage`: near-perfect feature→target association by type, identifiers derived from the target, and timestamps after the event when `time_index` is set. Without a target it is `not_applicable` with a note.
  - `mutual_information` (`ml`) with the target.
- **Acceptance:**
  - kitchen5k's planted `leakage_*` columns are flagged.
  - Without a target, both checks are `not_applicable` with a note.
- **Commit:** `feat(checks): detect target leakage and mutual information`

### U4-15 · Synthesis: quality dimensions and ML readiness
- **Status:** done (2026-10-23) · **Size:** M · **Depends on:** U4-14 · **Resolves:** F-26 (Leakage dimension), F-21 (inputs as soft edges) · **PR:** #121 (`9fe8873`)
- **Notes:** the `core` pack holds 47 checks, and the default pack is `core` alone, so `dsbf.profile()` plans no legacy check. `quality_dimensions` and `ml_readiness` (`checks/synthesis.py` holds what they share) are report-only synthesis checks: they raise no finding, name every input in a soft `after`, and read the findings of the core checks. `quality_dimensions` reports the five quality dimensions by each finding's own `dimension`, from the detection checks listed in `QUALITY_INPUTS`; the checks that advise or compare (`ADVICE_CHECKS`) are not inputs, since a column they mention is not flawed. Each dimension has a level (the legacy traffic light, with an error now red), the columns it flags, references to its findings and a coverage: the share of its input checks that ran or had no column of their kind, with each input's state (`assessed`, `nothing_to_check`, `needs_setting`, `not_run`, `failed`). A dimension no input could read, such as Leakage without a target, is `not assessed` with no level. `ml_readiness` reads `READINESS_INPUTS` into the five preparation dimensions with their `ml` actions, a gate over the assessed dimensions, and `complete: false` with `unassessed` when one was not. A test holds both lists to the check specs. Both legacy scorers are in `SUPERSEDED_BY`; `packs=["core", "legacy"]` still runs the other legacy tasks and a scorer asked for by id, and all of it goes in U4-18. Parity comparators and 64 accepted differences are in `tests/golden/migration/accepted_synthesis.py`; the core snapshots gain the two checks' levels.
- **Deviations:** the unit said `quality_dimensions` lists coverage per column. A `CheckResult` does not record the columns a check examined, so coverage is per input check; per column would add a field to the Run schema. The legacy `_level` gave an error amber below 15% of the columns although its docstring said red; the port follows the docstring. `zeros` and `dominant_value` are distribution findings in the core, so they leave the quality validity and usability (the legacy scorer counted them) and count in `ml_readiness` under Unusable features. The 14 legacy tasks that fed the scorers no longer run by default.
- **Run time:** at the standard depth, on the CI runner (the nightly run on `main`) against the baseline of U4-01: `standard-titanic` 2.4 s (was 20.8 s) and `standard-kitchen50k` 10.9 s and 745 MB (was 235.7 s and 2 214 MB), both inside the Phase 4 targets. The `engine-*` pairs now plan the 47 core checks alone, since the 14 legacy tasks ran only to feed the scorers. The two synthesis checks take under 10 ms.
- **Goal:**
  - `quality_dimensions` over core findings (completeness, validity, usability, redundancy, leakage), listing each dimension's inputs and their coverage. It is report-only, and Leakage is "not assessed" without a target.
  - `ml_readiness` over findings.

  Both declare their inputs with soft `after`, so a missing input lowers coverage instead of skipping the score. After this unit the default pack is `core`, and no legacy task is in the default selection.
- **Acceptance:**
  - The default `dsbf.profile()` plans no legacy check.
  - The `standard-*` perf scenarios are measured against U4-01's figures, and the drop is recorded.
  - The golden findings snapshots cover every core check.
  - Parity with the legacy scorers, or listed differences.
- **Commit:** `feat(checks): score quality dimensions and ML readiness from findings`

### U4-16 · Calibration suite
- **Status:** done (2026-10-24) · **Size:** M · **Depends on:** U4-15 · **Resolves:** REVIEW_FINDINGS part 2 C (calibration suite)
- **Notes:** `tests/calibration/` holds 70 scenarios in 13 modules: frames generated from a seed with a flaw planted, or none. A plant is detected when a finding of its check, at the severity it asks for, names its columns; a false alarm is a warning (or, for the checks that only say `info`, a finding) that no plant or allowance explains; a probe reads a number off a result (pairs of copies found, the traffic-light level). `baseline.json` commits the detection rate, false-alarm rate (with Wilson intervals) and probe means of every core check, overall and per scenario, and a scenario fails when a rate got worse than its baseline, a probe moved the wrong way, or a check's version differs. A `calibration` test tier (deselected by default, `--update-calibration-baseline`) and `.github/workflows/calibration.yml` (nightly, on demand, PR label `calibration`) run it in about nine minutes and upload the rates. `tests/validation/` is eight in-memory tests of the archetypes on `dsbf.profile()` with the helpers in `support.py`; the legacy assertion modules, `run_validation.py` and the README warning about it are gone, and the dashboard checklists moved to `tests/validation/DASHBOARD_CHECKLIST.md` (the `clean` one says 9 columns).
- **Calibrations:** the user widened the unit to every Note it carried. The first run found 18 checks to change, each bumped a version, with its rates before and after in `PLAN.md` at the unit's PR: `outliers` (false alarms on clean normal columns .83 to .01: cutoffs that rise with n), `multivariate_outliers` (.78 to .00: a robust Mahalanobis distance and a forest score against a normal sample's), `normality` (.17 to .02: two tests at half the level, a minimum Q-Q gap), `skewness` (the mean-median warn level was unreachable), `group_differences` (Welch's ANOVA on values and ranks: the classic F called equal means different 55% of the time), `seasonal_decomposition` and `autocorrelation` (STL strength floors for white noise of the length, a Ljung–Box gate: .80 and .93 to .07 and .10 on 48 periods), `stationarity` (a trend-stationary reading: a ramp is no longer drifting), `string_anomalies`, `encoded_columns`, `reference_drift` (chance PSI subtracted, a chi-squared gate, the target held to half the limits: .93 to .03 at 150 rows, target drift 0 to 1.00), `mutual_information` (a noise floor per class feature), `target_leakage` (`warn_at` 0.93), `fuzzy_duplicates` (the same text with another number is capped at 0.7 alike), `transformations`, `quality_dimensions` (62% to 100% of eight tables at the level a written rule gives), and `dominant_value` and `class_imbalance` (the profile records `mode_count`; the result schema goes to 0.10.0 and the types change). The defaults of `missingness_mechanism`, `associations`, `contingency`, `collinearity`, `bimodality`, `numeric_binning` and `categorical_encoding` held up and are unchanged; `dominant_value` and `zeros` stay out of the quality dimensions; the identifier and datetime pairs stay out of `associations`; the text parts of `target_leakage` need no polars version (0.65 s for 50 000 rows × 5 columns).
- **Deviations:** the parity harness compares each core check with its legacy task and accepts a difference one at a time; the calibrated checks differ in thousands of rows (Welch's p-value in every pair), so `tests/golden/migration/accepted_calibration.py` accepts the kinds they changed for all datasets (`AcceptedKind`), and the golden snapshots and the calibration suite guard them from here; U4-18 deletes the harness. Two scenarios record gaps the unit did not close: `fuzzy_two_sources_unaided` (the same people in two systems with different keys is found only with `exclude_columns`: detection 0.00) and the recall of `fuzzy_duplicates` on copies edited in numbers (.95 to .97). The perf baseline was not re-recorded: several checks cost more (the reference forest and Mahalanobis fit, 40 null draws, the trend tests), and it is promoted from the nightly run on `main` after the merge.
- **Run time:** the calibration job takes about nine minutes on this machine, run in one process; running the scenarios in a process pool hung on `fuzzy_duplicates`, which starts a process of its own. On 40 columns × 2 000 daily periods, `seasonal_decomposition` takes 5.0 s, `stationarity` 4.5 s (the two trend tests double it) and `autocorrelation` 1.8 s.
- **Goal:**
  - Planted-truth datasets, generated deterministically: leakage, MAR and MNAR missingness, bimodality, near-duplicate rows, drift, outliers and encoding damage.
  - False-positive and false-negative rates per check version, from a nightly workflow that uploads its report, with baselines committed.
  - `tests/validation/` rebuilt on `dsbf.profile()`, its archetype assertions written against findings, and the README warning dropped.
- **Acceptance:**
  - The nightly calibration job is green.
  - The baseline rates are committed.
  - The validation archetypes pass in memory.
- **Commit:** `test: add a calibration suite of planted-truth datasets`

### U4-17 · Stopgap CLI on the new engine
- **Status:** todo · **Size:** S · **Depends on:** U4-15
- **Note from U0-11 (was in Phase 5):** `dsbf profile` defaults `--depth` to `standard`, while `dsbf quickstart` and `dsbf run` use `full`. The stopgap uses `standard`.
- **Note from U2-01:** add a parquet file with a `datetime64` column to the end-to-end test (the legacy writer failed on it).
- **Goal:** `dsbf profile DATA [--out DIR] [--pack] [--depth] [--target] [--time-index] [--plugin] [--param check.key=value]` runs `dsbf.profile()`. It writes `run.json` to `--out`, resolved to an absolute path (default `$DSBF_HOME/outputs/<run_id>/`), and prints `Run.render_health()`. The exit code is 0 unless setup fails (2). The command attaches console logging. The legacy `dsbf run` and `dsbf quickstart` are removed.
- **Acceptance:**
  - An end-to-end test from a temporary directory (titanic and the datetime parquet) writes a `run.json` that validates against `schema/run.schema.json`.
  - The README's CLI section is updated.
- **Commit:** `feat(cli): run dsbf profile on the new engine`

### U4-18 · Remove the legacy stack
- **Status:** todo · **Size:** L · **Depends on:** U4-16, U4-17 · **Resolves:** N-A2, F-38; deletes the legacy copies behind F-06, F-08, F-09, F-15, F-17, F-18, F-20, F-22, F-23, F-24, F-25, F-34 and F-36
- **Note from U3-10 and U3-06:** the adapter goes with `set_plugin_logger` / `PLUGIN_LOG_FN` and the global `PLUGIN_WARNINGS` in `eda/task_registry.py`, the legacy `task_groups` config key, `legacy_checks(session, plugin_groups=...)`, `LegacyParams.shared` (the whole shared config repeated in every adapted check's parameters), and the legacy `resource_limits.max_memory_gb` and `max_runtime_seconds` keys.
- **Note from U2-02:** remove the legacy `data_sampling` (the F-09 crash and the `stratified` alias for head). **Note from U2-06:** remove `eda/stage_inference.py`.
- **Note from U3-12:** the two plot-writing checks left out of an in-memory run go with `plot_factory`.
- **Note from U4-09:** `docs/WALKTHROUGH.md` describes the legacy missingness task's output (`consistent_with`, Little's test) and `frontend/src/components/quality/MissingnessMechanismCard.vue` reads it; both go with the legacy engine, and Phase 6 draws the check's `columns` instead.
- **Note from U1-06 (was in Phase 6):** `tests/perf/baseline.json` records the HTTP 500s of API v1 on the adversarial run; the legacy scenarios and their API timings go with API v1.
- **Note from U0-08:** the time-series test configs carry an inert nested `time_series.tasks.<name>` block; it goes with those tests.
- **Moot with this unit (were in Phase 5):** `dsbf run -c` using the YAML as the whole config (F-22), the dead config keys `metadata.output_format`, `engine.severity_thresholds` and `metadata.layout_name`, `message_verbosity`'s comment, and a relative configured `output_dir` (U1-04). All of these belong to the legacy config.
- **Goal:**
  - Delete `dsbf.legacy`, `dsbf.eda`, `dsbf.core`, `interfaces/api.py` (`EDA`), `dsbf.storage`, `dsbf.api` (API v1), `utils/plot_factory.py`, `utils/reco_engine.py`, `utils/data_utils.py`, the legacy config (`default_config.yaml` and its helpers, `custom_plotly_templates.py`), `run_profile.py`, `migrate_runs.py`, and `dsbf tasks export` with `static_metadata/task_metadata.*`.
  - Delete the legacy golden snapshots, `test_engine_parity.py` and `engine_differences.json`.
  - Replace the legacy perf scenarios with the `standard-*` and `profile-*` scenarios.
  - Remove the legacy entries from the pyrefly baseline by regenerating it.
  - Update `CLAUDE.md` practice 9, `docs/architecture.md`, and the README. Mark `docs/WALKTHROUGH.md` as describing the removed legacy run.
  - Keep the frontend source; it has no API until Phase 6.
- **Assumptions to confirm:** nothing outside the deleted packages imports them (search `src`, `tests` and `docs/practice_run`); the frontend's lint, build and tests don't need the API.
- **Acceptance:**
  - No import of the deleted packages remains in `src` or `tests`.
  - Every gate passes, and the golden tier holds only core snapshots.
  - The wheel builds, and `dsbf profile` works from a clean venv.
- **Commit:** `refactor!: remove the legacy engine, tasks, store and API`

### U4-19 · Lean base install and Phase 4 exit
- **Status:** todo · **Size:** M · **Depends on:** U4-18
- **Note from U4-16:** (1) Promote the perf baseline from the nightly run on `main` once U4-16 is merged, since the calibrated checks cost more (see its Run time). (2) The calibration rates depend on `numpy`, `scipy`, `statsmodels` and `scikit-learn`: when the caps lift, run `uv run pytest -m calibration` and review each moved rate before `--update-calibration-baseline`. (3) Gaps the suite records: `fuzzy_duplicates` finds the same people in two systems only with `exclude_columns` for the key that differs (a leave-one-column-out score would find them unaided), has no symmetric-delete blocking for single-character typos in every token, and flags look-alike records that differ in a letter and a digit (15% of runs); `bimodality` misses peaks 4 spreads (Ashman's D) apart in most samples below 300 values; `stationarity` misses a random walk in 3 runs of 20. Decide at Phase 4 exit which to close.
- **Note from U4-12:** `profile_dataset` raises a Polars `DuplicateError` ("duplicate output name '1:nulls'") for a frame with a column named `*` beside an all-null column, on `main` too; the generated-frame test for `reference_drift` skips such frames (`assume("*" not in frame.columns)` in `test_adversarial.py`). Fix it in the profiler and drop the `assume`.
- **Note from U4-04:** `LegacySession.frame()` panics (polars `NulError`) converting a frame with a NUL in a column name to pandas, so `dsbf.profile()` on a frame read from UTF-16 with `utf8-lossy` fails in the first legacy task, while the core `text_encoding` check runs. It goes with the legacy session; add a guard sooner only if it bites.
- **Note from U0-15 (ADR 0009):** shrink the base install to the core dependency set, moving the rest behind extras and deleting the dependencies nothing imports (for example matplotlib, plotly, seaborn, networkx, graphviz, pydot and missingno, verified by grep).
- **Note from U1-05:**
  1. Each extra needs a test that the core runs without it and reports `unavailable`.
  2. CI runs the extras as an install matrix, one leg per extra.
  3. Decide pyarrow's home: the design keeps it in the base.
  4. Declare `numpy` (never declared; imported by 34 files under `src/` on 2026-10-19) with a tested range, and declare every other directly imported package.
- **Note from U0-16 and the held-back table:** lift `pandas <3` (the golden suite and the pandas extra's tests must pass on 3.x), `statsmodels <0.15`, `scikit-learn <1.8` and `scipy <1.18`. Regenerate the golden findings and explain the differences.
- **Note from U3-12:** the README's library section and the golden child both rely on the pandas backend default; with no legacy task left, make Polars the default.
- **Goal:** The base install is the core set. The extras each have an `unavailable` test, and CI runs an extras matrix. The caps are lifted, and the default backend is Polars. The perf budgets are enforced, and the Phase 4 exit is measured.
- **Acceptance:**
  - In a clean venv, `pip install dsbf` without extras runs `dsbf profile` on titanic, with the `ml`, `timeseries` and `text` checks `unavailable`.
  - The install size is measured before and after.
  - On the CI runner, `standard-titanic` takes under 5 s, and `standard-kitchen50k` under 60 s with under 1 GB peak, as enforced budgets.
  - Only the Phase 6 rows remain in the held-back table.
- **Commit:** `build: shrink the base install to the core dependency set`

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
- **Status:** todo · **Size:** S · **Depends on:** U4-19
- **Goal:** Confirm the outline units below against what Phase 4 delivered, and fill in each one's assumptions and steps. Ask the user how the package will be published (the PyPI name `dsbf` and trusted publishing) and confirm the HTML report's scope.

### U5-01 · Policy evaluator
- **Status:** outline · **Size:** M · **Depends on:** U5-00
- **Note from U4-06:** `out_of_bounds` (plausibility range, `warn`) and `schema` (contract, with min and max per column) both take bounds for a column. Decide whether a policy rule reads one as the other.
- **Note from U1-10:** (1) the policy loader turns `yaml.safe_load` and `Policy.model_validate` failures into a `ConfigError`-style report that names the rule; (2) decide what `new_since: previous_run` does without a previous run; (3) add a range on one setting (`>0.1` and `<0.5`) only if users ask. **Note from U1-11:** publish `schema/policy.schema.json` with the loose spellings documented, together with its loader.
- **Goal:** `dsbf.profile(..., policy=..., previous=...)` evaluates the rules over findings and health and fills `Run.policy`. `fail` only sets the exit code.
- **Acceptance:** A REFACTOR_PLAN §7.2 example policy produces the documented `Run.policy`. A run with no rules matched exits 0. The policy schema export is checked in CI.

### U5-02 · Findings query API and notebook display
- **Status:** outline · **Size:** S · **Depends on:** U5-01 · **Resolves:** REVIEW_FINDINGS part 2 D (notebook experience)
- **Goal:** `run.findings.filter(severity=">=warn", dimension=..., column=...)`, `.to_polars()` and `.to_pandas()` (with the `pandas` extra). `Run._repr_html_` shows the health summary and the top findings in a notebook.
- **Acceptance:** The REFACTOR_PLAN §9 library example runs as a test. The HTML display renders the adversarial run without breaking on odd column names.

### U5-03 · JSON and NDJSON sinks, and reading a run back
- **Status:** outline · **Size:** M · **Depends on:** U5-02 · **Resolves:** N-B2 (output schemas)
- **Note from U4-00 (was on U4-02):** every core check declares a params and output model from U4-02 on, but nothing in Phase 4 reads them as JSON Schema, so their export lands here, with the published run schema.
- **Goal:** `run.write(dir, sinks=["json", "ndjson"])` writes `run.json` and `findings.ndjson`, and `dsbf.Run.read(path)` round-trips. The stopgap CLI's writer moves onto it. `dsbf schema export` also writes each core check's params and output schema (`schema/checks/<id>.schema.json`), and `--check` covers them.
- **Acceptance:** A round-trip test passes. The files validate against the published schema with `jsonschema`, and each check's output in a titanic run validates against its exported schema. `dsbf schema export --check` covers the check schemas.

### U5-04 · Parquet sinks and the consumer schema guide
- **Status:** outline · **Size:** M · **Depends on:** U5-03
- **Goal:** `findings.parquet`, `profile.parquet` and `results.parquet` with documented, versioned columns, plus a guide for loading them into a warehouse or BI tool.
- **Acceptance:** The Parquet schemas have their own breaking-change check. DuckDB reads the example queries in the guide back.

### U5-05 · Static HTML report
- **Status:** outline · **Size:** M · **Depends on:** U5-04 · **Resolves:** REVIEW_FINDINGS part 2 D (shareable export)
- **Goal:** One self-contained HTML file per run (no server, no CDN) with the health summary, findings by dimension and severity, and per-column profiles drawn from the profile's equal-width bins.
- **Acceptance:** The titanic report is under a size budget set in U5-00 and opens offline. User text is escaped (the F-14 lesson).

### U5-06 · CLI v2
- **Status:** outline · **Size:** M · **Depends on:** U5-05 · **Resolves:** N-B1 (generated CLI help and config reference), REVIEW_FINDINGS part 2 G (pipeline hooks)
- **Note from U3-09:** give the Typer app one callback that attaches the console handler, with a `--verbosity` option, so every command prints log records. **Note from U1-09:** CLI v2 builds the `cli` layer for `dsbf.config.load_config` and needs a `--set a.b=c` parser.
- **Note from U4-05:** `dsbf.profile()` takes no per-check parameters beyond the `target` setting, so a core check's params (such as `schema`'s expectations) reach it only through `make_plan(params=)` and `dsbf plan --param`; add `params=` to `profile()` and a section to `ProfileConfig`. Eligibility ignores parameters, so `dsbf plan` says `class_imbalance` and `schema` "will run" where they end `not_applicable` without a target or schema: let a check declare the parameters it needs.
- **Goal:** `dsbf profile` and `dsbf check` (with `--config`, `--set`, `--policy`, `--previous`, `--out` and `--sink`), plus `plan` and `schema export`. Every run prints the health summary. The exit code is 0, or 1 when the policy fails the run, or 2 on a setup error.
- **Acceptance:** The commands are tested end to end from a temporary directory, and each exit code is tested.

### U5-07 · Execution settings and cache management
- **Status:** outline · **Size:** S · **Depends on:** U5-06
- **Note from U3-07:** expose the worker count in the CLI and config, with per-check `runs_in` and timeout overrides (U3-06 deferred those too). **Note from U3-08:** add `dsbf cache clear` and a size or age eviction for `<DSBF_HOME>/cache/results/`, and expose cache on/off in the CLI and config.
- **Acceptance:** A per-check timeout set in config applies. Eviction keeps the cache under its cap, which a test checks.

### U5-08 · Docs for embedded use
- **Status:** outline · **Size:** M · **Depends on:** U5-07
- **Note from U3-09:** a library call prints nothing unless the host configures logging; say so, and list the accepted levels.
- **Goal:** README rewrite. Guides: "Use DSBF in your pipeline", "Consume DSBF findings in your own dashboard", "Write a check" and "Write a policy", plus a generated check reference (params, outputs, findings codes). An example GitHub Actions workflow writes Parquet findings.
- **Acceptance:** Every code block in the guides is executed by a test, and the example workflow runs in CI.

### U5-09 · Release readiness and 1.0.0a1
- **Status:** outline · **Size:** M · **Depends on:** U5-08 · **Resolves:** F-31 (install smoke test), REVIEW_FINDINGS part 2 E (PyPI) and F (coverage)
- **Note from U3-07:** worker processes use `spawn` where there is no `forkserver` (Windows), about 0.44 s per check as measured on Linux, and no CI leg covers that path; add one before the first release.
- **Goal:** A CI job that builds the wheel, installs it in a clean venv and runs `dsbf check` from a temporary directory. A Windows leg. Coverage reporting. A release workflow on tags with trusted publishing, the CHANGELOG, and the `1.0.0a1` release. Publishing is the user's step.
- **Acceptance:** The smoke job runs on every PR. `pip install dsbf==1.0.0a1` works from PyPI.

---

## Phase 6: Store, service and dashboard v2. Standalone local (placeholder)

- **Capability:** A run store (SQLite with Alembic) and an artifact store (fsspec), `dsbf runs ...`, API v2 with a local job runner, a generated TypeScript client, and the dashboard rebuilt on API v2 and served by `dsbf dashboard` from the wheel; release `1.0.0`.
- **Motivation:** Mode 2 (standalone, local): DSBF is the pipeline, with history and a bundled UI (REFACTOR_PLAN §8.2–§8.4).
- **Boundaries:** Local and single-user only: no Postgres, auth or queue (Phase 8). API v1 and the old store were deleted in U4-18, so this phase builds new and migrates nothing.
- **Sequence and dependencies:** Depends on Phase 5. Store, then API, then client, then the dashboard tab by tab, then packaging.
- **Success criteria:** `pip install "dsbf[server]" && dsbf check data.csv --sink store && dsbf dashboard` works on a clean machine. Opening a run transfers under 200 KB.

- **U6-00 · Plan Phase 6** (`todo`, depends on the Phase 5 units).
  - **Carry in from U4-00:** the dashboard does not run from U4-18 until this phase's API v2 and shell land (decided 2026-10-19). F-15 (dataset identity, the snapshot of the data as profiled) and F-25 (pagination, error mapping, migrations, garbage collection, encoded path parameters) are design inputs for the store and API, not fixes to old code.
  - **Carry in from U1-11:** (1) the dashboard is JavaScript, so it can use `frontend/src/generated/run.d.ts` only through JSDoc `import()` types or after a move to TypeScript; (2) a TypeScript consumer needs its own rank table for `Severity` and `Confidence`, which the schema cannot carry; (3) revisit the `vue-router <5` and `plotly.js-dist-min <4` caps in the held-back table.
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
