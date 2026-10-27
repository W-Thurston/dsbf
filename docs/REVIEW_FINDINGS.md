# DSBF Review Findings

> **Reference document.** Upcoming work is tracked as units in [`ROADMAP.md`](../ROADMAP.md);
> this file records the findings (F-xx) and ideas (N-xx) those units cite.

These findings come from a full review of the codebase on **2026-10-15**. The review looked at
branch `feature/viz_refactor` at `ca999ba`, plus the uncommitted working tree. It has two parts:

1. **Fixes**: defects, risks and cleanup in the current code, ordered by priority.
2. **Next steps**: where the project goes after that, grouped by theme and sequenced into milestones.

Items marked **(verified)** were reproduced during the review, either with a script, the test
suite or CI logs. The others come from reading the code. File references are `path:line` at the
reviewed commit.

---

## Status after Phase 5 (reviewed 2026-10-26)

Phase 5 built the policy evaluator, the sinks, CLI v2 and the release machinery (U5-00 to U5-11). It
resolves F-31 (the `package` job) and part 2 D (notebook experience, shareable export), E (release
workflow; publishing to PyPI is the user's step) and F (coverage reporting), and N-B1 and N-B2
(generated references and output schemas). The other findings keep their Phase 4 status.

---

## Status after Phase 4 (reviewed 2026-10-25)

Phase 4 migrated every analysis to a typed check, deleted the legacy engine, tasks, store and
API v1 (U4-18, #126) and shrank the base install (U4-19). Each finding that Phase 4 changed has a
second **Status (2026-10-25)** line under the earlier one; the earlier lines are kept as written,
and where they say a legacy copy "is deleted in U4-18" it now is.

| Status | Findings |
|---|---|
| Resolved | F-01 to F-11, F-13, F-14, F-16 to F-19, F-21 to F-28, F-31 (U5-09), F-32 to F-39 |
| Partial, owned by a later phase | F-12 (Phase 6, the dashboard draws the profile's bins), F-15 (Phase 6, dataset identity), F-20 (Phase 6, the Run health panel), F-29 (Phase 6, the frontend build), F-30 (Phase 7, read-time sampling) |

---

## Status after Phases 0–3 (reviewed 2026-10-19)

Each finding now opens with a **Status** line, checked on 2026-10-19 against `main` at `ec202ed`
by reading the code and by running `dsbf.profile()` on titanic, the adversarial frame and small
probe frames. The line gives the unit and PR that changed it, and the unit that owns what is left.
Unit IDs refer to [`ROADMAP.md`](../ROADMAP.md); finished units are listed with their PRs in
[`HISTORY.md`](HISTORY.md).

- **A ticked box** is fixed in the code that remains after Phase 4. The legacy engine, tasks, store and API v1 are deleted in U4-18 (decided on 2026-10-19), so a defect that survives only in that code is ticked, and its status line names U4-18.
- **Resolved in the new …** means the replacement is built and the legacy copy still runs until U4-18.
- **Partial** means some boxes are ticked and the status line names the owner of the rest. **Open** means not started; the line names the unit that owns it.

| Status | Findings |
|---|---|
| Resolved | F-01, F-02, F-03, F-04, F-05, F-14, F-16, F-17, F-19, F-21, F-27, F-33 |
| Resolved in the new engine, profile, data layer or contracts (legacy copy deleted in U4-18) | F-06, F-08, F-09, F-18, F-23, F-24, F-26, F-34, F-35, F-36, F-37, F-39 |
| Mostly resolved or partial | F-07, F-10 (U4-18), F-11 (U4-08 to U4-15), F-12, F-15, F-20, F-22, F-28, F-29, F-30, F-31, F-32 |
| Open, owned by Phase 4 | F-38 (U4-18) |
| Open, resolved by deleting API v1 (U4-18) and replacing it in Phase 6 | F-25 |

---

## Baseline at time of review

| Check | Result |
|---|---|
| `pytest` | **5 failing** (see [F-03](#f-03-five-failing-tests)) |
| `ruff check` / `ruff format --check` (local, ruff 0.11) | Clean |
| GitHub CI | **Red on every recent PR**: Lint fails, so tests never run (see [F-01](#f-01-ci-is-red-on-every-pr)) |
| Open PRs | 5 Dependabot PRs (#24–#28) blocked by red CI |
| Frontend `vite build` | Builds, but ships one **5.5 MB** JS chunk (1.7 MB gzip) |
| Registered tasks | 67, all `domain="core"` |
| Test coverage gaps | No tests for `dsbf/api/`, `dsbf/storage/`, `dsbf/interfaces/`, or the frontend |

---

# Part 1: Fixes

## P0: Correctness and data safety (fix before the next release)

### F-01 CI is red on every PR
**Status (2026-10-19): Resolved.** U0-02 (#31) installs ruff from `uv.lock` and moved the actions to Node 24; `.pre-commit-config.yaml` (`v0.17.0`) is kept in step with the lock. The backlog was cleared by U0-04 (#40), U0-16 (#59) and U0-17 (#64).

**(verified)** `.github/workflows/ci.yml:22` runs `pip install ruff` without a version, so CI gets
the latest ruff. The project pins `ruff = "^0.11.10"`. Formatter output differs between versions,
so `ruff format --check` fails in CI and the test job never starts. Every Dependabot PR is stuck
behind this.
- [x] Install ruff at the version in `poetry.lock`, either via `poetry install --only dev` or `pip install "ruff==<locked>"`.
- [x] Keep `.pre-commit-config.yaml` (`rev: v0.11.10`) in sync with that version, or bump both together.
- [x] Bump `actions/checkout`/`actions/setup-python` to releases on Node 24. CI already prints Node 20 deprecation warnings.
- [x] Once CI is green, work through the Dependabot backlog (#24–#28).

### F-02 The test suite writes into the developer's real database and output folder
**Status (2026-10-19): Resolved.** U0-07 (#50): a session-wide autouse fixture in `tests/conftest.py` points `DSBF_HOME` and `DSBF_DB_PATH` at a temporary root and changes into a temporary directory. The metadata export has been opt-in since U1-04 (#71), so it needs no switch.

**(verified)** While the suite was running, `dsbf/dsbf.db` (479 MB) was being modified. The
`clean_engine_run` fixture sets `config["engine"]["output_path"]` (`tests/conftest.py:27`), but the
engine never reads that key. It reads `output_dir` or `metadata.output_dir`. Nothing sets
`DSBF_DB_PATH`, so every engine-level test persists a "titanic" run into the dashboard DB and
writes to `dsbf/outputs/`. Tests also rewrite the tracked `dsbf/static_metadata/task_metadata.json`
(see [F-17](#f-17-every-run-rewrites-tracked-files-and-grows-dsbf_runjson)).
- [x] Add an autouse session fixture that sets `DSBF_DB_PATH` to a tmp file and `DSBF_AUTO_EXPORT_METADATA=0`, and passes `output_dir=tmp_path`.
- [x] Have the fixture `chdir` to a tmp dir so CWD-relative writes ([F-16](#f-16-paths-depend-on-the-current-working-directory)) can't leak.

### F-03 Five failing tests
**Status (2026-10-19): Resolved.** U0-03 (#32). The five tests pass (re-run on 2026-10-19).

**(verified)**
- `test_detect_bimodal_distribution::test_skips_all_null_column` / `test_skips_constant_column` fail with `KeyError: 'bimodal_flags'`.
- `test_suggest_categorical_encoding::test_high_cardinality_gets_frequency_encoding` fails with `KeyError: 'encoding_suggestions'`.
- `test_normality_tests::test_custom_alpha_respected` fails with `KeyError: 'x'`.
- `test_memory_tracking::test_memory_limit_exceeded_warning` fails because `MemoryTestTask` isn't registered, and `tests/helpers/context_utils.py:51` requires that.

The first two come from this branch's BaseTask migration: `BaseTask.make_empty_result()`
(`dsbf/core/base_task.py:392`) returns `data={}`, which drops the top-level keys each task used to
return (for example `{"bimodal_flags": {}}`). That changes the data contract the frontend and
scorers read. The encoding and normality failures look related but haven't been root-caused yet.
- [x] Let `make_empty_result()` accept the task's empty data shape (for example `make_empty_result(msg, excluded, data={"bimodal_flags": {}})`), and use it in every migrated task.
- [x] Confirm the cause of the normality and encoding failures.
- [x] Register the memory-test task in a fixture, or let `make_ctx_and_task` accept unregistered classes.

### F-04 The default config fails its own validation
**Status (2026-10-19): Resolved.** U0-08 (#51) moved the shared block to a top-level `time_series:` section.

**(verified)** `validate_config_and_graph(load_default_config())` returns
`["Unknown task name in config['tasks']: 'time_series'"]`. The shared `time_series` block
(`default_config.yaml:285`) lives under `tasks:` but isn't a task
(`dsbf/utils/config_validation.py:27`). Every run logs a config warning, and with
`safety.strict_mode: true` **every run aborts**.
- [x] Move shared blocks to a top-level `shared:` or `time_series:` section and update `BaseTask.get_shared_param()`, or whitelist known shared blocks in the validator.

### F-05 Uncommitted config change hard-codes one dataset
**Status (2026-10-19): Resolved.** U0-01 (#29) reverted it: the default config has `time_series.enabled: false` and `datetime_index_column: null`.

The working-tree diff of `dsbf/config/default_config.yaml` sets `time_series.enabled: true`,
`datetime_index_column: GAME_DATE` and `seasonal_period: 7`. Those are NFL-dataset specifics. If
this is committed, every user's run turns on time-series analysis against a column that doesn't
exist. It also contradicts the documented "DSBF will not guess" principle.
- [x] Revert before committing, and keep dataset-specific settings in a local config passed with `dsbf run -c`.

### F-06 The engine treats a `status="failed"` result as success
**Status (2026-10-19): Resolved in the new engine.** U3-05 (#92) trusts the status a check returns and never infers success from the absence of an exception; U3-10 (#97) added `ctx.stop(...)`, so an adapted task's `failed` is `error` and its `skipped` is `skipped`. The executor owns isolation, so a check has no try/except boilerplate. The legacy `ExecutionGraph` (`eda/graph.py:38`) still marks a task `success` when nothing escapes; it is deleted with the legacy engine (U4-18).

**Status (2026-10-25): Resolved.** The legacy `ExecutionGraph` is deleted (U4-18, #126).

**(verified)** `Task.run` (`dsbf/eda/graph.py:34`) marks a task `"success"` whenever no exception
escapes, even if the task returned `TaskResult(status="failed")`. Dependents then run on bad
upstream output (`graph.py:105` only checks `Task.status`), and `task_outcomes` overcounts
successes. This affects `schema_validation` in `fail_or_warn: "fail"` mode, `detect_target_drift`,
and `identify_bottleneck_tasks`, which return failed results instead of raising.
- [x] Set `Task.status` from `result.status`.
- [x] **(verified in the practice run)** The reverse happens too. Five tasks that return `status="skipped"` themselves (`schema_validation`, `compare_with_reference_dataset`, `detect_class_imbalance`, `detect_feature_drift`, `detect_target_drift`) are listed under `task_outcomes["success"]`.
- [x] Separately, about 60 tasks repeat the same `try/except: if self.context: raise / else make_failure_result` block. Move it into a template method on `BaseTask` (see [N-B3](#b-engine-architecture)).

### F-07 Documented config keys are silently ignored
**Status (2026-10-22): Partial.** Checks declare their parameters, and the planner rejects an unknown key or a value out of bounds (U1-09 #77, U3-04 #91). Adapted legacy tasks still take their config as a free dict (`LegacyParams.task`), so the ignored keys in the table, `ignore_null_type ... or True` (`detect_mixed_type_columns.py:72`) and 43 `get_task_param(x) or default` calls remain until each task migrates (U4-06 to U4-15; the nine column health checks of U4-04 and the seven rule checks of U4-05 take their bounded params from `CheckParams`, and `schema` replaces `schema_validation`'s four keys with one `columns` mapping in which an unknown key is a planning error; since U4-09 `mixed_types` has a real `ignore_null_tokens` switch and `format_violations` takes its rules as a list in which a pattern that does not compile is a planning error, so `match_threshold` is gone). `dsbf.profile(target=...)` (U3-12 #99) is one setting: it fills the `target_column` keys of the legacy tasks and, since U4-05, the `target` parameter of every core check that declares one (`class_imbalance`), and, since U4-12, of `reference_drift`, which replaces `detect_target_drift` (whose `target` key never saw it) and `detect_feature_drift` with typed `psi_warn`, `psi_high`, `tvd_warn`, `tvd_high`, `bins`, `missing_shift` and `target` parameters; the ignored `js_threshold`, `ks_pvalue_threshold`, `chi2_pvalue_threshold`, `compare_*` and `enabled` keys of the three drift tasks are gone from it. Since U4-14, `target_leakage` and `mutual_information` take it too, and `categorical_encoding` reads their results; no check reads a separate target key.

**Status (2026-10-25): Resolved.** Every check in the `core` pack declares bounded parameters and the planner rejects an unknown key; the legacy free-dict config is deleted (U4-18, #126).

**(verified)** A script cross-checked `default_config.yaml` against the keys each task reads:

| Task | Keys in config that are never read | What the code actually reads |
|---|---|---|
| `detect_feature_drift` | `psi_threshold`, `tvd_threshold`, `js_threshold`, `ks_pvalue_threshold`, `chi2_pvalue_threshold`, `enabled` | `psi`, `tvd` |
| `detect_target_drift` | `psi_threshold`, `ks_pvalue_threshold`, `chi2_pvalue_threshold`, `enabled` | `psi`, `ks_pvalue`, `chi2_pvalue`, `entropy_delta`, `target` |
| `compare_with_reference_dataset` | `compare_types`, `compare_missingness`, `compare_summary_stats`, `enabled` | three thresholds only |
| `detect_regex_format_violations` | `match_threshold` | n/a |
| `extract_datetime_features` | `skew_threshold` | n/a |
| `data_quality_scorer` | `weights` (old dimension names; the scorer is no longer weighted) | n/a |

Related problems:
- [x] `detect_mixed_type_columns.py:72`: `bool(get_task_param("ignore_null_type") or True)` is always `True`, so users can't turn it off. (`mixed_types`, U4-09: `ignore_null_tokens` is a bool parameter, true by default, and off it counts null-like words as a kind of value.)
- [ ] Several params use the `get_task_param(x) or default` pattern, so a legitimate `0` or `False` is replaced by the default. Use `get_task_param(x, default)`.
- [ ] The target column is configured four different ways: `target_column: target` in two tasks, `target_column: null` in one, and `target` (not in the config) for target drift. Replace them with one top-level `target_column`.
- [x] The long-term fix is task-declared parameter schemas ([N-B1](#b-engine-architecture)), which make this class of drift impossible.

### F-08 Stage inference reads the wrong config path and depends on the backend
**Status (2026-10-19): Resolved in the new profile.** U2-06 (#85): `dsbf.profiling.infer_stage` reads `StageThresholds.from_config` and the profile, so it does not depend on the backend. The legacy `eda/stage_inference.py` is still called by `ProfileEngine` until U4-18.

**Status (2026-10-25): Resolved.** The legacy `eda/stage_inference.py` is deleted (U4-18, #126).

**(verified)**
- `dsbf/eda/stage_inference.py:8` reads `config["stage_inference"]`, but the default config puts it under `engine.stage_inference`, so the thresholds are ignored.
- The Polars branch (`stage_inference.py:33`) compares dtypes to the strings `"i64"`/`"f64"`, which never match. The same 4×2 numeric frame is inferred as `model_ready` with pandas and `cleaned` with Polars. Built-in datasets default to Polars.
- The inferred labels (`raw`/`cleaned`/`model_ready`/`exploratory`) don't match the task stage vocabulary (`raw`/`cleaned`/`modeling`/`report`/`any`), so the inferred stage can't drive task selection.
- [x] Fix the config path, use `df.schema` numeric checks for Polars, and unify the stage vocabulary.

### F-09 Polars and `sample_strategy: random` crash
**Status (2026-10-19): Resolved in the new data layer.** U2-02 (#81): `Dataset.sample` implements seeded random and stratified sampling and records them in `SamplingInfo`. The legacy `utils/data_utils.py:35` still passes `random_state=` to Polars; it is deleted in U4-18.

**Status (2026-10-25): Resolved.** The legacy `utils/data_utils.py` is deleted (U4-18, #126).

**(verified)** `dsbf/utils/data_utils.py:24` calls `d.sample(n=..., random_state=42)`. Polars uses
`seed=`, so this raises `TypeError`. `"stratified"` is a silent alias for `"head"`, which biases
any ordered file.
- [x] Branch on backend, and either implement stratified sampling or reject it with a clear error.

### F-10 Column names containing `|` crash pair-based tasks
**Status (2026-10-19): Partial.** The contracts key pairs by `Subject.column_pair` (U1-06 #74), and U0-09 (#52) stopped the leakage crash: a frame with columns `a|b` and `c|d` runs to a complete `Run` (checked on 2026-10-19). Seven legacy tasks still key pairs as `"a|b"`, and API v1 splits on the first `|`. `associations` and `contingency` (U4-07) report every pair as a `Subject.column_pair` and a record with `first` and `second` (a frame with columns `a|b` and `c|d` gets the right subjects, tested). `group_differences` (U4-08) does the same for a numeric and a categorical column, so every pair-based core check is done and the seven legacy tasks go with U4-18, with API v1.

**Status (2026-10-25): Resolved.** The seven legacy tasks that keyed pairs as `"a|b"` and API v1 are deleted (U4-18, #126); every pair-based core check reports a `Subject.column_pair`.

**(verified)** Pair results are keyed as `"col1|col2"` and split with `split("|")`. A column named
`a|b` makes `detect_data_leakage` raise `ValueError: too many values to unpack`
(`detect_data_leakage.py:110`). The API uses `split("|", 1)` (`dsbf/api/main.py:224`, `:432`), which
silently mis-attributes the pair instead.
- [x] Store pairs as `{"a": ..., "b": ..., ...}` records, or use a nested `{col1: {col2: ...}}` map, instead of encoded string keys.

### F-11 Semantic-type filtering is still bypassed in places
**Status (2026-10-22): Partial.** A check written to the new API sees only the columns its `applies_to` selects (`ctx.columns`, U3-01 #88), and the `core` pack's checks do: `zeros` (U4-02) and the nine column health checks (U4-04) and the seven rule checks of U4-05, which declare `applies_to` and read `ctx.profile` only (`schema` also reads the values it names), and the five distribution analysis checks of U4-06 the two pairwise checks of U4-07 (`associations`, `contingency`), the group difference and collinearity checks of U4-08 and the five missingness and string checks of U4-09, which read `ctx.values` for the columns they apply to (the four string checks take the columns the profile types `categorical`, `text` or `identifier` and skip one that is not stored as strings, saying why); the adversarial test runs all thirty-one. The regression test is in: on the adversarial frame the integer-coded booleans (`flag_as_int`, `flag_copy`) and `[/x]` never appear in `zeros`, and on titanic `zeros` reads `age` and `fare` where `detect_zeros` also read `survived`, `pclass`, `sibsp` and `parch` and raised guidance for three of them (`tests/golden/migration/accepted.py` lists each difference). The legacy tasks still pick their own columns: on a 200-row frame, a 0/1 integer column that the profile types `categorical` appeared in the output of nine continuous-only checks on 2026-10-19 (`detect_zeros` among them, now replaced), `detect_data_leakage` and `detect_collinear_features` among the rest. 21 task files still call `select_dtypes`. Each migration removes its copy (U4-04 to U4-15). The same frame shows that an integer row number (`0..n-1`) is profiled `numeric`, not `identifier`; `id_columns` (U4-04) reads a numeric integer column with no repeat and no gap as a row number.

**Status (2026-10-25): Resolved.** All 47 `core` checks pick their columns through `applies_to`, and the adversarial test runs them all; the legacy tasks that picked their own are deleted (U4-18, #126).

v0.20.0 fixed tasks iterating over `df.columns` instead of `matched_cols`. The same bug class
remains: `detect_data_leakage.py:73` computes correlations over
`df.select_dtypes(include="number")`, so integer-encoded booleans and IDs enter the leakage scan
and could produce false "Leakage" findings. 21 task files use `select_dtypes`.
- [ ] Audit each one and intersect with `matched_cols`, or replace it with `matched_cols`.
- [x] Add a regression test: an int-coded boolean column must not appear in any continuous-only task output (U4-02, for `zeros`; each migration adds its check to `tests/test_checks/test_adversarial.py`).

### F-12 The new histogram misrepresents distributions
**Status (2026-10-19): Partial.** The profile carries equal-width bins with the rule recorded (U2-05 #84). The dashboard still draws the percentile bins (`DistributionHistogramChart.vue:31`); the dashboard is off from U4-18, and its Phase 6 column view draws the profile's bins.

`DistributionHistogramChart.vue:37` builds bins between the min, 1/5/25/50/75/95/99th percentiles
and the max, then plots each bin's *count* as an equal-width bar. The bins have very different
widths but similar counts by construction, so the shape you see mostly reflects the percentile
choice, not the data. A uniform distribution would look peaked. Given the project's
"epistemic honesty" principle, this should not ship.
- [x] Emit real equal-width bin edges and counts from `summarize_numeric` (Freedman–Diaconis or Sturges, via `np.histogram`), and render those.
- [ ] As a stopgap, plot density (`count / width`) on a value axis instead of a category axis.

### F-13 The "Kendall's τ for n < 30" routing is unreachable with defaults
**Status (2026-10-22): Resolved in the new engine (legacy copy deleted in U4-18).** On a 20-row frame `compute_pairwise_associations` computed 0 associations (verified on 2026-10-19 and again on 2026-10-22). `associations` (U4-07) gates only on `min_n` (8) and routes after it, so the same frame gets Kendall's τ-b for every numeric pair (`test_a_20_row_frame_gets_kendalls_tau`).

**(verified)** `compute_pairwise_associations.py:517` skips pairs with
`valid_n < min_sample_size` (default 30) *before* method routing at `:298`, so the
`auto → kendalls_tau` branch never runs with the shipped config. README and CHANGELOG both
advertise it.
- [ ] Lower the default gate (for example to 10) or route before gating, and add a test using a 20-row frame.

### F-14 Stored XSS via dataset name
**Status (2026-10-19): Resolved.** U0-10 (#53): `highlightHtml` in `frontend/src/utils.js` escapes the text before it inserts `<mark>`.

`DatasetListView.vue:34` and `RunListView.vue:48-49` render `highlight(text, query)` with `v-html`.
`highlight()` (`DatasetListView.vue:86`) regex-escapes the *query* but never HTML-escapes the
*text*. A dataset named `<img src=x onerror=...>` (for example via `dsbf profile f.csv --name ...`)
executes in the dashboard. The risk is low for a local tool but real once a dashboard is shared.
- [x] HTML-escape `text` before inserting `<mark>`, or render matches as text nodes. Deduplicate the helper into `utils.js`.

## P1: Robustness and operational correctness

### F-15 Run identity and dataset identity
**Status (2026-10-19): Partial.** `Run.run_id` is a UUID, `started_at` is a timezone-aware time, and `DatasetManifest` carries the content fingerprint (U1-08 #76, U2-02 #81). The legacy store still uses a second-resolution `run_key` and datasets keyed by name, and API v1 re-reads the source; both are deleted in U4-18. Dataset identity and the per-run data snapshot are Phase 6 work.

**Status (2026-10-25): Partial.** The legacy store and API v1 are deleted (U4-18, #126). Dataset identity and the per-run data snapshot are still Phase 6 work.

- [x] **run_key collisions:** `run_key` is a second-resolution timestamp (`dsbf/core/base_engine.py:93`). Two runs in the same second share an output dir (`exist_ok=True` overwrites), and `_insert_run` silently reuses the first run's row, merging task results. Add a short random suffix or a UUID.
- [x] **`ran_at` can be NULL:** with a custom `output_dir`, `run_key` isn't a timestamp, `ran_at` becomes `None` (`writer.py:313`), and the explicit NULL overrides the column default. Those runs then sort last and break "latest run" queries. Use the actual start time.
- [ ] **Datasets are keyed by name only.** Two different files profiled under the same name are merged. `source_path` is set once and never updated (`writer.py:42`), so later runs point at the first file. The `EDA()` facade and default config both fall back to `"default_dataset_name"`.
- [ ] **The dashboard can show different data than what was profiled.** `/sample` and `/column-data` re-read whatever is at `source_path` *now*. That file may have changed, may be a different file (above), and ignores any sampling applied during the run. Store `source_path`, file size, mtime and a content hash per run, and warn in the UI when they no longer match.

### F-16 Paths depend on the current working directory
**Status (2026-10-19): Resolved.** U1-04 (#71): runtime paths resolve under `DSBF_HOME`, and stored paths are absolute. `dsbf.profile()` writes nothing (U3-12 #99, with a test).

- `dsbf/outputs/<ts>` (`base_engine.py:93`), `dsbf_run.json` (`:99`) and `dsbf/static_metadata/task_metadata.json` (`task_registry.py:279`) are all relative to CWD.
- The API resolves figure and source paths relative to the **repo root** (`dsbf/api/main.py:507`, `dsbf/api/db.py:150`).
- The default DB lives inside the package (`dsbf/storage/schema.py:26`).

Running `dsbf profile` from any other directory leaves the dashboard with broken figures. Once the
package is pip-installed, it would write into `site-packages`.
- [x] Introduce one `DSBF_HOME` (default: a `platformdirs` user data dir, or `./.dsbf` per project), store absolute paths in the DB, and resolve everything from it.

### F-17 Every run rewrites tracked files and grows `dsbf_run.json`
**Status (2026-10-19): Resolved.** U1-04 (#71) made the export an explicit `dsbf tasks export` and removed `record_run()`. The generated `task_metadata.json` is still tracked; it goes with the legacy registry in U4-18.

**Status (2026-10-25): Resolved.** `dsbf tasks export` and the generated `task_metadata.*` are deleted (U4-18, #126).

- `load_task_group()` calls `write_task_metadata()` on **every run** (`dsbf/eda/task_registry.py:276`, enabled by default). It rewrites the tracked `task_metadata.json` and strips its trailing newline, which fights the `end-of-file-fixer` pre-commit hook. That's why the file is always dirty in `git status`.
- `record_run()` appends to `dsbf_run.json` with no bound. It's currently 442 entries and 240 KB, and the SQLite `runs` table already holds the same information.
- [x] Make metadata export an explicit command (for example `dsbf tasks export`) and stop tracking the generated JSON, or generate it in CI.
- [x] Remove `record_run()`, or cap it.

### F-18 Version and git SHA describe the wrong repository
**Status (2026-10-19): Resolved for the library.** `dsbf.__version__`, the run manifest and `dsbf version` read `importlib.metadata` (U3-12 #99). API v1 still reports `version="0.1.0"` (`api/main.py:48`), and the legacy engine still runs `git rev-parse` (`core/base_engine.py:102`); both are deleted in U4-18.

**Status (2026-10-25): Resolved.** API v1 and the legacy engine are deleted (U4-18, #126).

`get_dsbf_version()` runs `git describe` and `_get_git_sha()` runs `git rev-parse` in the
**current directory** (`dsbf/utils/versioning.py:16`, `base_engine.py:125`). When DSBF is used
from inside another project's repo, run metadata records *that* project's tag and SHA. The API also
reports `version="0.1.0"` (`dsbf/api/main.py:45`).
- [x] Use `importlib.metadata.version("dsbf")` everywhere, and only record a SHA when the package itself is installed from a git checkout.

### F-19 Logging
**Status (2026-10-19): Resolved.** U3-09 (#96): `dsbf.*` loggers with only a `NullHandler`, an `error` level, and handlers that are attached and closed by context managers. The markup fix landed in U0-09 (#52).

- Task logs go through `AnalysisContext._log` to the `dsbf.context` logger, which has **no file handler** and `propagate=False`. As a result, `run.log` misses task-level messages, including memory/runtime limit warnings and TaskResult validation failures.
- `setup_logger` calls `handlers.clear()` without closing them (`logging_utils.py:64`), which leaks one open `run.log` file descriptor per run in long-lived processes such as notebooks or a future API-triggered runner.
- `logging.setLoggerClass(DSBFLogger)` at import time (`logging_utils.py:30`) changes logger creation for every library imported afterwards.
- `RichHandler(markup=True)` interprets brackets in messages. **(verified)** A column named `[bold]` disappears from the log. A message containing `[/x]` raises a `MarkupError` inside the handler, which `logging` catches, so the message is lost and a "Logging error" traceback is printed. Some call sites escape with `\\[` (and that escape then leaks into `run.log` as a literal `\[`); most don't.
- `level="error"` isn't mapped in `get_log_fn` and falls back to INFO2.
- [x] Use one `dsbf` logger hierarchy with child loggers that propagate, a file handler attached once per run and removed and closed at the end, `markup=False` (or escape user data), and an `error` level.

### F-20 Failure visibility end to end
**Status (2026-10-19): Partial.** The status model (U1-07 #75) and the executor (U3-05 #92) give every check a result, `skipped` with `blocked_by`, and an `error` note with its traceback in `Note.detail`. `Run.render_health()` prints the summary (U3-12 #99). Persisting full results and a Run health panel in the dashboard are Phase 6 work. The legacy `TaskResult` defects listed here are deleted in U4-18.

**Status (2026-10-25): Partial.** The legacy `TaskResult` is deleted (U4-18, #126). `dsbf profile` prints `Run.render_health()`; the dashboard's Run health panel is Phase 6.

- Skipped tasks (failed dependency) get **no TaskResult** (`graph.py:105-117`), so they're missing from `report.json`, the DB and the dashboard. Users can't tell "not applicable" from "skipped because X failed".
- The DB stores only `status/summary/data/guidance` (`dsbf/storage/writer.py`). `error_metadata`, `reliability_warnings`, `recommendations` and `metadata` (for example `memory_exceeded` and `excluded_columns`) are dropped, so the dashboard can't explain failures or show reliability caveats.
- `TaskResult.to_dict()` (`task_result.py:74`) omits `recommendation_tags`.
- `error_to_metadata()` (`task_result.py:116`) keeps only `str(e)` and no traceback.
- [ ] Emit `status="skipped"` results with the blocking dependency, persist the full TaskResult (one JSON column or extra columns plus a schema migration), and add a "Run health" panel in the dashboard listing failed and skipped tasks with reasons.

### F-21 Ordering relies on implicit DAG levels
**Status (2026-10-23): Resolved.** Soft `after` edges (U3-03 #90). The adapter gives the three tasks that read undeclared results `after` edges, and a test scans the task sources for new ones (U3-10 #97). The scorers that motivated the finding are now `quality_dimensions` and `ml_readiness` (U4-15), which name every input in `after`: a check that failed, was not planned or needs a setting the caller did not give lowers the dimension's coverage and never skips the report, and the old scorers leave the default selection.

`data_quality_scorer` deliberately leaves `detect_collinear_features` and `detect_out_of_bounds`
out of `depends_on` (`data_quality_scorer.py:127`) so their failures don't skip the scorer. It
still reads their results, and only gets them because both happen to sit at a lower topological
level. Any change to their dependencies would silently empty the Redundancy and Validity
dimensions.
- [x] Add soft dependencies (`runs_after=[...]`): ordering edges that don't propagate failure.

### F-22 Configuration loading
**Status (2026-10-19): Partial.** `dsbf.profile()` takes data and settings as arguments, validates the enums in `ProfileConfig` and never mutates its inputs (U1-09 #77, U3-12 #99). `dsbf run -c` (which still uses the YAML as the whole config), the `EDA` facade and the dead config keys are deleted with the legacy CLI and config in U4-18. CLI v2 gets layered config in U5-06.

**Status (2026-10-25): Resolved.** Resolved by deletion: `dsbf run -c`, the `EDA` facade, the legacy config and its dead keys are gone (U4-18, #126). Layered config for CLI v2 is U5-06.

- [ ] `dsbf run -c my.yaml` uses the YAML **as the whole config** (`dsbf/interfaces/cli.py:53`). A partial user config loses every default task parameter. Deep-merge user config over defaults. The `default_config.yaml` comment already promises a `user_config.yaml` override that doesn't exist.
- [x] `EDA(...)` (`dsbf/interfaces/api.py`) mutates the caller's config dict, raises `KeyError` when the config has no `metadata` block (`:36`), and supports in-memory data by monkeypatching `engine._load_data` (`:48`). Add a first-class `ProfileEngine(config, data=df)` parameter.
- [x] Validate enum-like values (`profiling_depth`, `backend`, `runtime_estimate`, `sample_strategy`) up front. An invalid depth currently raises a bare `KeyError` in `build_graph`.
- [ ] Dead config: `metadata.output_format`, `engine.severity_thresholds`, `layout_name` (`dsbf/layouts/default.json` is unused), and the comment in `default_config.yaml` about a "dashboard Settings → Time Series panel" that doesn't exist yet.

### F-23 Plugin loading
**Status (2026-10-19): Resolved in the new registry.** U3-02 (#89): an explicit `Registry`, directory plugins imported under names unique to their path, nothing registered on import, and plugin problems recorded as notes. The legacy registry (registration on import in `utils/task_utils.py:14`, the global `PLUGIN_WARNINGS`, modules named by file stem) is deleted in U4-18.

**Status (2026-10-25): Resolved.** The legacy registry and `PLUGIN_WARNINGS` are deleted (U4-18, #126).

- [x] The `core` task group imports `dsbf.custom_plugins.core`, which is **empty**. Core tasks are actually registered as an import side effect of `dsbf/utils/task_utils.py:14` (`load_all_tasks()` at module import). Make registration explicit.
- [x] Loading a plugin directory twice in one process (a notebook or the API) re-executes the module, hits "already registered", and logs a traceback. `PLUGIN_WARNINGS` is a global that accumulates across runs and is never cleared.
- [x] Plugin modules are registered in `sys.modules` by file stem (`task_registry.py:296`), so two plugin dirs that each have a `utils.py` overwrite each other.
- [x] A non-directory, non-module `task_groups` entry fails with a printed traceback instead of a clear message. `if not group_path.exists()` inside `if group_path.is_dir()` is dead code.

### F-24 Resource tracking is misleading
**Status (2026-10-19): Resolved in the new engine.** U3-06 (#93) and U3-07 (#94): wall time, CPU time and sampled peak memory per check, labelled approximate, with enforced timeouts and a run budget. The legacy `graph.py:126` is deleted in U4-18.

**Status (2026-10-25): Resolved.** The legacy `graph.py` is deleted (U4-18, #126).

`peak_mem = max(rss_before, rss_after)` (`graph.py:126`) is neither a peak nor per-task, because
RSS is process-wide and rarely shrinks. It also mixes units (`/1e6` vs `* 1024`).
`max_runtime_seconds` only warns.
- [x] Measure per-task deltas (or use `tracemalloc` / a sampling thread for peaks), label them honestly, and optionally enforce timeouts ([N-B5](#b-engine-architecture)).

### F-25 API and data-layer issues
**Status (2026-10-19): Open; resolved by deletion and replacement.** Still true of API v1 (for example `/sample` reads the whole file, `api/db.py:194`). Decided on 2026-10-19: API v1 and the old store are deleted in U4-18, and the Phase 6 store and API v2 are built against these items. The `encodeURIComponent` item goes with the generated client in Phase 6.

**Status (2026-10-25): Resolved.** Resolved by deletion: API v1 and the old store are gone (U4-18, #126). The list stays a design input for the Phase 6 store and API v2.

- [ ] `/sample` reads the **entire source file** to return 10 rows (`dsbf/api/db.py:193`). Use `nrows=`. For built-in datasets, each request re-downloads from seaborn or sklearn. Cache, or snapshot a sample at profile time.
- [ ] `_load_dataframe` returns `None | DataFrame | {"error": ...}`. `/column-data` passes the error dict back as an HTTP 200 payload. Raise and map to 4xx/5xx instead.
- [ ] `/associations/{column}` and `/correlations/{column}` sort by `abs(metric)` across metric types. η² (where 0.14 is "strong") is ranked against r (where 0.7 is "strong"), so the ordering is meaningless across types. Sort by strength rank, then by metric within type.
- [ ] The endpoint list in the module docstring is stale. Several handlers ignore `_DB_PATH`, which works only because `schema.py` re-reads the env var.
- [ ] Add a delete/GC path (`DELETE /api/runs/{key}` and `dsbf runs delete|gc`). `clean_dsbf_db.txt` currently documents manual SQL. DB rows outlive deleted output dirs (only 1 of many runs still has files on disk), which leaves orphaned figure rows.
- [ ] `runs.quality_score` is always NULL (`writer.py`), but `DatasetListView` still renders a "Quality" column from it, so it always shows "-". Drop the column (with a migration) or replace it with the five-dimension summary.
- [ ] Add a `schema_version` table and real migrations instead of `ALTER TABLE ... except OperationalError`.
- [ ] `frontend/src/api.js`: `getDataset`, `listRuns` and `getTask` don't `encodeURIComponent` their path params, so names containing spaces, `/` or `#` break routing.

### F-26 "Leakage" is mislabeled
**Status (2026-10-23): Resolved in the new engine (legacy copy deleted in U4-18).** The feature–feature check is now Redundancy: `associations` (U4-07) raises `redundant_columns` (warning, redundancy dimension) for |Pearson r| ≥ 0.99 and says it is no statement about a target. The target-based check is `target_leakage` (U4-14): it measures every feature against the target by type pair, finds identifiers built from the outcome, missingness that follows it and datetimes later than the time index, and is `not_applicable` with a note without a target. `quality_dimensions` and `ml_readiness` (U4-15) read only `target_leakage` for Leakage, and report the dimension as `not assessed`, with no level, when it could not run.

The Leakage dimension is driven by `detect_data_leakage`, which flags *feature–feature* pairs with
|r| ≥ 0.99. That's redundancy or duplication. Target leakage means a feature encodes information
about the target that wouldn't be available at prediction time, and detecting it requires a target.
Calling feature collinearity "leakage" conflicts with the hedged-language principle.
- [x] Fold the current check into Redundancy. Redefine Leakage around a configured `target_column` (near-perfect feature→target association, post-event timestamps, target-derived IDs). If no target is set, show "not assessed" rather than green.

## P2: Hygiene, docs and developer experience

### F-27 README accuracy
**Status (2026-10-19): Resolved.** U0-11 (#54) corrected the README, and U0-12 (#55) made the pyrefly badge true. U3-12 (#99) rewrote the library section, and the README's Roadmap section links to `ROADMAP.md`.

- [x] It says "Python 3.10+", but `pyproject.toml` requires `>=3.12`.
- [x] `from dsbf.api import profile_file` and `from dsbf.api import ProfileEngine` don't exist (`dsbf/api/__init__.py` is empty and `dsbf.api` is the FastAPI package). The in-memory example (`engine.df = ...`) has no effect. The real facade is `dsbf.interfaces.api.EDA`.
- [x] The default depth is documented as `standard`, but `default_config.yaml` ships `full`, along with `message_verbosity: debug` and `visualize_dag: true`.
- [x] It says "D3-native chart components", but the refactor in progress uses ECharts.
- [x] It shows a "Type checked: Pyrefly" badge, but no type checker is configured or run.
- [x] Fold the README's "Roadmap" section into a link to this file.

### F-28 Repository clutter
**Status (2026-10-19): Mostly resolved.** U0-11 (#54) deleted the stray files, `.flake8` and the dead code, and moved the handoff doc to `docs/architecture.md`. `poetry.lock` went with the move to uv (U0-05 #41). `run_profile.py` and `migrate_runs.py` were kept on purpose while the legacy engine runs; they are deleted in U4-18.

**Status (2026-10-25): Resolved.** `run_profile.py` and `migrate_runs.py` are deleted (U4-18, #126).

- [x] Untracked or stray files: root `package-lock.json` (empty), `clean_dsbf_db.txt`, `dsbf_reference_dataset.csv` (5.3 MB), `dsbf_test_schema.json`, `tests/generators/dsbf_test_generator_backup.py`, `dsbf_directory_structure.md`. Delete them, or move them under `docs/`, `examples/` or `tests/fixtures/`.
- [x] Move `dsbf_handoff_doc.md` into `docs/architecture.md`. It's the best architecture doc in the repo.
- [ ] `.flake8` is obsolete now that ruff is used. `run_profile.py` and `migrate_runs.py` at the root should become CLI subcommands.
- [x] Remove dead code: `dsbf/utils/check_recommendation_contexts.py`, `dsbf/utils/display_registered_tasks.py`, `dsbf/layouts/default.json`, and the commented-out `render_dashboard` in `cli.py`.
- [x] Commit the regenerated `poetry.lock`, which removes `panel` and its transitive deps to match `pyproject.toml`.

### F-29 Frontend build and code health
**Status (2026-10-19): Partial.** Vite is on a stable release (8.3), and ESLint, Prettier, Vitest and the build run in CI (U1-03 #68). The bundle split and the tab monoliths (now 1,491 to 1,772 lines) are Phase 6 work.

- [x] Unpin `vite ^8.0.0-beta.13` and its `overrides` block and move to a stable release.
- [ ] Bundle size: Plotly (via `PlotCard.vue:35` static import, which also defeats `PairPlotCard`'s dynamic import) plus ECharts ship together in one 5.5 MB chunk. Finishing the ECharts migration ([N-A1](#a-finish-in-flight-work)) and route-level code splitting should cut this substantially.
- [ ] Tab components are monoliths: `QualityTab.vue` (1.5k lines), `TimeSeriesTab.vue` (1.2k) and `MlReadinessTab.vue` (1.1k). Extract cards and composables (`useRunTasks`, `useTheme`).
- [x] Add ESLint, Prettier and a `vite build` job to CI.

### F-30 Engine performance
**Status (2026-10-23): Partial.** One profile pass, a per-column value cache for checks (U2-02 #81), and file loading through Polars (U2-01 #80). Each adapted legacy check still gets its own copy of the frame (a full copy with the default pandas backend) until U4-18 and U4-19. Read-time sampling is Phase 7 (T2). The reference dataset takes any source `dsbf.data.load` accepts, or a `Run`, through `dsbf.profile(reference=...)` (U4-12); only its profile reaches `reference_drift`.

**Status (2026-10-25): Partial.** No adapted legacy check copies the frame any more (U4-18, #126). Read-time sampling is Phase 7.

- [x] Each pandas-dependent task calls `to_pandas()` on the Polars frame independently (`BaseTask.setup_run` / `get_dataframe_pandas`), which can mean up to about 60 full copies per run. Convert once and cache on `AnalysisContext`.
- [ ] Data is fully loaded before sampling. For large files, sample at read time (Polars `scan_*` with `head` or `sample`).
- [ ] `_load_data` ignores `engine.backend` for files and always returns pandas. (The reference dataset no longer supports only CSV: resolved by U4-12.)

## Practice-run findings (2026-10-15)

These come from an instrumented, isolated run on four datasets: titanic, the repo's kitchen-sink
generator at 5k and 50k rows, and an adversarial frame. The stage-by-stage measurements are in
[`WALKTHROUGH.md`](WALKTHROUGH.md). Everything below is **verified**.

### F-31 Tasks crash when run outside the repo root
**Status (2026-10-19): Partial.** U0-07 (#50) loads package data with `importlib.resources` and makes a failed tip lookup return no tip. U4-04 replaced `eval` with declarative `{field, op, value}` conditions (`engine/tips.py`), validated when the library loads; core checks get tips after they return, and a failed lookup is a note. A CI job that installs the wheel and runs from a temporary directory is U5-09; U1-05 (#72) checked this by hand.
**Status (2026-10-26): Resolved.** The `package` CI job (U5-09) builds the wheel, installs it into a clean venv with no extras and runs `dsbf check` from a temporary directory, with a passing and a failing policy, on every pull request; the Windows job runs it too.

`dsbf/utils/reco_engine.py:7` opens `Path("dsbf/static_metadata/recommendation_library.yaml")`
relative to the CWD. Twelve tasks call `get_recommendation_tip()` inside their main `try` block, so
an optional "tip" lookup raises `FileNotFoundError` and fails the whole analysis. The failures
cascade: `data_quality_scorer` and `ml_readiness_scorer` are skipped, so the **Quality bar and ML
Readiness tab are empty**, as are `transformation_preview` and `normality_qq_plots`. This affects
every pip-installed user and anyone running from a different directory.
- [x] Load package data with `importlib.resources.files("dsbf.static_metadata")`. Audit every other `Path("dsbf/...")` / `open("dsbf/...")` call.
- [x] Make enrichment non-fatal: a missing tip should produce no tip, not a failed task.
- [x] `get_recommendation_tip()` ran `eval(condition, {}, context_vars)` on YAML strings (`reco_engine.py:28`). With empty globals, Python still injects builtins. Replace it with a small declarative condition format (`{"field": ..., "op": ">", "value": ...}`).
- [ ] Add a CI job that runs one profile from a temp directory against an installed wheel.

### F-32 Listwise deletion across all numeric columns
**Status (2026-10-22): Partial.** The profile computes statistics on each column's own non-null values (U2-04 #83), and U0-09 (#52) added the VIF crash guard. `collinearity` (U4-08) leaves out a column more than 40% missing before it drops any row, then drops the sparsest column until there are at least k + 1 (and 10) complete rows, states `n_used`, the columns left out and the columns dropped for rows, and uses the centred VIF (the diagonal of the inverse correlation matrix). On kitchen5k the legacy task had no row to fit; on `severe_multicollinearity` the check reads 2 000 rows where the task read 721. The legacy `compute_reliability_flags` still deletes rows listwise, until U4-15 takes the scorers out.

**Status (2026-10-25): Resolved.** The legacy `compute_reliability_flags` is deleted with the legacy tasks (U4-18, #126); `collinearity` is the only VIF.

`numeric_df = df.select_dtypes("number").dropna()` drops every row with *any* null in *any*
numeric column:
- `compute_reliability_flags` (`dsbf/utils/reliability_stats.py:18`) used **0 rows** on kitchen5k and adversarial, and 714 of 891 on titanic. Global reliability flags are computed on an empty or biased frame.
- `detect_collinear_features.py:85` crashes on `zero-size array` because the guard checks the column count, not the row count. This is the crash the `data_quality_scorer` comment works around (F-21).
- [x] Compute per-column statistics on per-column non-null values. For multivariate methods (VIF), drop columns above a missingness threshold first, then rows, and record `n_used` in the result. (`collinearity`, U4-08)
- [ ] Guard on `n_rows < k + 1`, not only on column count. Grep for the pattern in all tasks.

### F-33 Crashes on all-null columns and `None` values
**Status (2026-10-19): Resolved.** U0-09 (#52) fixed both crashes. The adversarial frame is a permanent fixture (`tests/fixtures/adversarial.py`) that every legacy task (`tests/eda/test_tasks/test_adversarial_data.py`) and the engine (U3-11 #98) run against. U4-02 runs every core check against it too, which enforces the logging standard for new code.

- `summarize_modes.py:59`: `col_modes[0]` on an all-null column raises `IndexError`. The task also iterates `df.columns` instead of `matched_cols`.
- `normality_tests.py:223`: a **debug log line** formats `p_value=None` with `:.4f`, raising `TypeError` and failing the task. The f-string is evaluated even when debug logging is off.
- [x] Fix both. Adopt the standard that log calls never do formatting that can raise: pre-format with a helper that renders `None` as `n/a`, or use `%`-style lazy logging.
- [x] The adversarial frame used in the practice run should become a permanent fixture that every task runs against (N-F, adversarial property tests).

### F-34 Wasted post-processing every run
**Status (2026-10-19): Resolved in the new engine.** `dsbf.profile()` builds no column-to-task map, draws no DAG image and serialises nothing (U3-12 #99). The legacy `ProfileEngine` still does all three (`eda/profile_engine.py:123`, `visualize_dag: true` by default) until U4-18.

**Status (2026-10-25): Resolved.** The legacy `ProfileEngine` is deleted (U4-18, #126).

- `build_column_task_mappings` re-serialises every result via `to_dict()` (0.67 s on titanic) and treats every `data` key as a column name. The map is polluted with `__dataset__`, `__correlation_matrix__`, `age|class|First_vs_Second`, and so on. Use `task_column_map` from the columns each task actually processed (`matched_cols`).
- `visualize_dag: true` by default draws a 400 KB `dag.png` on every run (about 1 s), and it's identical for an identical task set. Make it opt-in, or cache it by task-set hash.
- `TaskResult._sanitize()` and `json.dumps(indent=2)` run over the full payload several times (report, mapping, DB). Serialise once and reuse the bytes.

### F-35 Fuzzy duplicate detection dominates runtime and output, and its results are arbitrary
**Status (2026-10-22): Resolved in the `fuzzy_duplicates` check (U4-10).** The legacy task stays in the legacy pack until U4-18 deletes it, but the default selection no longer runs it, and `fuzzy_duplicates` is a heavy check, so it runs at the deep depth. On 2026-10-19 it was 8.46 MB of titanic's 8.93 MB `Run` and 16.7 s of the legacy engine's 26.7 s in the CI perf baseline. The check scores each column with `rapidfuzz`, weights the columns by their entropy, leaves labels (columns of 20 or fewer values) out, blocks on rare tokens only when all pairs do not fit the budget, states `pairs_compared / pairs_possible` and the budget that stopped it, lists the 20 closest pairs, and ends `not_applicable` on titanic, which has no text column.

**Status (2026-10-25): Resolved.** The legacy `fuzzy_duplicate_detection` task is deleted (U4-18, #126).

It took **58.8 s of 72.5 s** of task time on titanic (891 rows) and **172 s of 296 s** on kitchen5k.
It produced **8.6 MB** of data on titanic, which made `report.json` **18.5 MB** (5.3 s to write) and
the dashboard's on-open `/tasks` response **8.2 MB**.

Efficiency:
- [x] On titanic, ~42 s were ~395,000 `working_df.iloc[i][col]` lookups (`fuzzy_duplicate_detection.py:374-377`). Pre-extract the string columns once as a NumPy array of row strings.
- [x] On kitchen5k, ~148 s were pure-Python `difflib.SequenceMatcher` on long text rows. Use `rapidfuzz` (C++, typically 50–100× faster) and truncate or hash very long fields.

Correctness:
- [x] Blocking keeps any pair that shares *one* token. On data with categorical columns, every row shares tokens like `male` or `southampton`, so blocking does nothing and the step stops at `max_candidates=50,000`. That's an **arbitrary ~12.6% of titanic's ~396k pairs**, and nothing in the output says the result is partial. Block on rare tokens (IDF-weighted or MinHash/LSH), exclude low-cardinality categoricals from the row string, and report `pairs_compared / pairs_possible`.
- [x] 16,316 flagged pairs on titanic suggests the 0.85 threshold on concatenated row strings mostly measures "same categorical values", not near-duplicate records. Calibrate against the planted near-duplicates in the kitchen-sink generator.
- [x] Store at most the top N pairs plus counts. Fetch details on demand.

### F-36 `NaN` / `Infinity` in stored data make API endpoints return HTTP 500
**Status (2026-10-19): Resolved in the contracts.** Non-finite values become `None` and are named in the contracts (U1-06 #74), and `Run` JSON is strict. API v1 can still emit `NaN`; it is deleted in U4-18.

**Status (2026-10-25): Resolved.** API v1 is deleted (U4-18, #126).

The writer stores task data with `json.dumps(..., default=str)`, which emits the non-standard tokens
`NaN` and `Infinity`. FastAPI's encoder refuses them on the way out. On the adversarial run,
`/associations` (a `NaN` metric) and `/sample` (an `inf` in the source data) both returned **500**.
`TaskResult._sanitize()` converts NumPy scalars but leaves float `nan`/`inf` alone.
- [x] Map non-finite floats to `None` in `_sanitize()` (or emit a sentinel object), write with `allow_nan=False` so violations fail loudly at write time, and handle `inf` in `/sample` serialisation.

### F-37 `infer_types` parses entire columns as dates
**Status (2026-10-19): Resolved in the profile.** U2-03 (#82): sample-based type inference took 0.2–0.3 s on kitchen50k, against 25.95 s for the legacy `infer_types`. `infer_types` still runs as an adapted check until it is superseded (U4-05) and deleted (U4-18).

**Status (2026-10-25): Resolved.** The legacy `infer_types` is deleted (U4-18, #126).

`_is_datetime_like` (`infer_types.py:250`) took 8.6 of `infer_types`' 9.0 s on kitchen5k (25 object
columns). Every downstream task waits on `infer_types`.
- [x] Test a sample of a few hundred non-null values first, try known formats before falling back to dateutil, and only parse the full column for columns that pass.

### F-38 `generate_univariate_plots` cost and output
**Status (2026-10-19): Open.** Already left out of in-memory library runs (U3-12 #99). Deleted with `plot_factory` in U4-18.

**Status (2026-10-25): Resolved.** `plot_factory` and the plotting tasks are deleted (U4-18, #126).

It took 84 s on kitchen5k and wrote 361 files, including **two 17–18 MB PNGs** for a column of
extremely long strings, because bar-chart labels are full category values. It would be removed
entirely by N-A2. Until then:
- [ ] Truncate labels, cap categories to the top-k, render one theme and let the frontend restyle it, and skip text-intent columns.

### F-39 `detect_stationarity` scales super-linearly
**Status (2026-10-22): Resolved in the `stationarity`, `autocorrelation` and `seasonal_decomposition` checks (U4-11).** The legacy tasks stay in the legacy pack until U4-18 deletes them, but the default selection no longer runs them. The checks resample to the configured or profile-inferred frequency (repeated timestamps aggregated, short holes interpolated), use at most the most recent `max_periods` (2 000) periods, cap the ADF and KPSS lags at `max_lag` (24) and the ACF/PACF lags at `max_lags` (40), and state the frequency, the series length and the lag cap on every finding. `stationarity` takes 0.28 s on kitchen50k (22 numeric columns, 365 daily periods) against 89 s for the task in the CI baseline, and 1.3 s on 52 500 rows over 3 000 days and 40 columns (2 000 periods each, the worst case the cap allows).

**Status (2026-10-25): Resolved.** The legacy time-series tasks are deleted (U4-18, #126).

On 10× more rows (kitchen5k → kitchen50k) it went from **4.9 s to 186 s (38×)**.
`detect_stationarity.py:192` calls `adfuller(series.values, autolag="AIC")` with no `maxlag`.
statsmodels' default max lag grows with n (about 57 at 52k rows), and it fits one regression per
candidate lag. It also tests the raw rows in file order rather than the series resampled to the
configured frequency. With tens of thousands of points, ADF and KPSS reject on trivially small
deviations anyway.
- [x] Resample or aggregate to the configured or inferred frequency before testing, cap `maxlag` (configurable), and cap the series length used for testing (for example the most recent N periods). Record all three in the result.
- [x] Apply the same review to `compute_acf_pacf` and `decompose_time_series`.

### Overall run cost (for tracking)
kitchen50k (52,500 × 60) took **707 s** with a 2.1 GB peak. Five tasks (`detect_stationarity`,
`fuzzy_duplicate_detection`, `generate_univariate_plots`, `infer_types`,
`compute_pairwise_associations`) account for about 90% of that. The practice-run scripts are now the
`tests/perf/` benchmark, which fails on a regression of more than 25% against a committed
baseline and keeps this from creeping back.

**Status (2026-10-19):** in the CI perf baseline (4 CPUs) the legacy engine took 26.7 s on titanic,
160 s on kitchen5k and 377 s at 2.1 GB peak on kitchen50k. `dsbf.profile()` on titanic took 26.9 s
locally (60 checks, all adapted legacy tasks). The REFACTOR_PLAN §11 budgets (titanic standard plan
under 5 s, 52k × 60 under 60 s and 1 GB) are Phase 4's exit criteria, measured in U4-01 and enforced
in U4-19.

---

# Part 2: Next steps

## A. Finish in-flight work

| ID | Item | Notes |
|---|---|---|
| N-A1 | **Complete the ECharts migration** | Move the remaining Plotly cards (`PlotCard`, `PairPlotCard`, correlation heatmap, missingness) to `DsbfChart`, then remove `plotly.js-dist-min`. Charts should render from task *data*, not pre-rendered figure files. **Status:** Phase 6 rebuilds the dashboard on ECharts only (the column view and the remaining charts). |
| N-A2 | **Retire backend plot generation** | Once the frontend no longer reads figure files, remove `generate_univariate_plots`, `generate_dataset_summary_plots` and `dsbf/utils/plot_factory.py` (1.2k lines), along with the `matplotlib`/`seaborn`/`missingno`/`plotly`/`pydot`/`graphviz` deps (keep seaborn only if built-in datasets still need it). This shrinks install size and run time and removes the figure-path problems in [F-16](#f-16-paths-depend-on-the-current-working-directory). **Status:** open. U4-18 deletes the plotting tasks and `plot_factory`, and U4-19 drops the plotting dependencies. Both tasks are already left out of in-memory library runs (U3-12). |
| N-A3 | **Q-Q plot** | `normality_qq_plots` already computes the quantiles. The Distributions tab shows a "coming soon" placeholder. **Status:** the `normality` check (U4-06) keeps the Q-Q quantiles of every column that fails its tests (or of all, with `qq_all_columns`): theoretical and empirical quantiles in the column's units, the Q1–Q3 reference line, the deviation in standard deviations and a pattern (heavy tails, right tail, left tail). The Phase 6 column view draws them. |
| N-A4 | **Time Series tab** | Finish ACF/PACF, stationarity and decomposition views. Add a friendly empty state and a UI path to set `datetime_index_column`. **Status:** `dsbf.profile(time_index=...)` exists (U3-12) and reaches the `stationarity`, `autocorrelation` and `seasonal_decomposition` checks (U4-11), which keep the tests, the ACF/PACF values with their bands and the trend and seasonal strengths, but not the STL component series; the Phase 6 store keeps those as an artifact, and the tab is rebuilt then. |
| N-A5 | **Findings tab** | Cross-column finding explorer (already on the README roadmap). It can be built entirely on the `guidance` blobs every task emits. **Status:** findings exist as a contract (U1-07) and come from every adapted task (U3-11). The explorer is Phase 6. |

## B. Engine architecture

| ID | Item | Why |
|---|---|---|
| N-B1 | **Typed config with task-declared params** | Each task declares its parameters, defaults and docs (pydantic or dataclass). `default_config.yaml`, validation, CLI help and docs are generated from those declarations. This removes the whole class of bugs in F-04, F-07 and F-22, and gives the dashboard a schema for a Settings panel. **Status:** partial. Checks declare `CheckParams`, and the planner validates them (U1-09, U3-04). Legacy tasks stay untyped until they migrate (Phase 4). A generated config reference and CLI help come with CLI v2 (U5-06, U5-08). |
| N-B2 | **Versioned TaskResult data schemas** | Each task declares the shape of `data`, and contract tests assert it, including the empty-result shape (F-03). Export JSON Schema or TypeScript types for the frontend, which today reads deeply nested `data?.x?.[col]` paths with no contract. **Status:** partial. `@check(output=...)` takes an output model (U3-01). Every migrated check declares one from U4-02 on, and U5-03 exports each core check's params and output schema. |
| N-B3 | **Template-method task execution** | `BaseTask.execute()` handles setup, column matching, empty results, exception wrapping, timing and reliability flags. Subclasses implement `compute(df, cols)`. This removes about 60 copies of boilerplate. **Status:** resolved by the check API: the executor owns isolation, empty results, timing and column selection (U3-01, U3-05). The legacy copies are deleted in U4-18. |
| N-B4 | **Soft dependencies and real skip semantics** | `runs_after=` ordering edges (F-21) plus first-class `skipped` results (F-20). **Status:** resolved (U3-03 soft `after`; U3-05 `skipped` with `blocked_by`). |
| N-B5 | **Parallel execution and timeouts** | Run independent tasks in each DAG level concurrently, with a configurable worker pool. The DAG and `runtime_estimate` metadata already exist. Enforce `max_runtime_seconds` per task. **Status:** resolved (U3-06 timeouts and run budget; U3-07 parallel levels, threads or one process per check). |
| N-B6 | **Large-data path** | Lazy Polars loading, read-time sampling with recorded provenance, and reliability warnings that say explicitly when results come from a sample. **Status:** partial. Sampling provenance flows into every `Evidence` (U2-02). Lazy and read-time sampling are Phase 7 (T2). |

## C. Statistical rigor

This is the project's core differentiator, so these items deserve priority after the P0 fixes.

- [ ] **Real histograms** (F-12): equal-width bins computed server-side, with the binning rule recorded in metadata. **Status:** partial: the profile has them (U2-05); the dashboard draws them in Phase 6.
- [x] **Uncertainty on associations (2026-10-22):** confidence intervals (Fisher z for r and τ, bootstrap for η² and V) and the n behind each estimate. Display "weak (95% CI 0.05–0.41, n=48)" instead of a bare label. **Status:** `associations` has Fisher z intervals for r, ρ and τ with `n_used` per pair, and `contingency` a percentile bootstrap interval for V̄ (U4-07). `group_differences` (U4-08) adds percentile bootstrap intervals for ω² and ε² (the rank test's effect size) and for a flagged pair's rank-biserial contrasts; η of `associations` keeps no interval, because the group tests report the same pair with one.
- [x] **Bias-corrected Cramér's V** (Bergsma 2013). The uncorrected V currently inflates on small samples and high-cardinality pairs. **Status:** resolved in `contingency` (U4-07); the plain V is kept beside it.
- [x] **A comparable association scale** across metric types (F-25). For example, convert η² to η, or rank by strength band and never by raw value across types. **Status:** resolved in the checks (U4-07): η replaces η² and point-biserial r, and every pair has a band on its own metric's scale. The API v1 ordering is deleted in U4-18.
- [x] **Multiple-comparison awareness** in pairwise scans on wide data. Report how many pairs were tested next to the number flagged. **Status:** done: `associations` and `contingency` state `pairs_tested` and `pairs_flagged` in the output and in each finding (U4-07); `group_differences` (U4-08) states `pairs_tested`, `pairs_flagged` and `tests_run`, adjusts each method's p-values over every pair (Benjamini–Hochberg by default) and a flagged pair's contrasts by Holm, and flags only a pair whose effect reaches `min_effect`.
- [ ] **Target-aware leakage detection** (F-26) built on a single `target_column`. **Status:** `target_leakage` and `mutual_information` (U4-14), on the one `target` setting of `dsbf.profile()` (U3-12); `quality_dimensions` shows the Leakage dimension as not assessed without a target (U4-15).
- [x] **Calibration suite (2026-10-24):** datasets with *known* ground truth (planted leakage, planted MAR or MNAR, planted bimodality), and false-positive and false-negative rates per check across versions. **Status:** `tests/calibration/` generates 70 planted-truth scenarios from seeds, scores every core check, commits the rates in `baseline.json` and runs nightly in CI (U4-16). `tests/validation/` is rebuilt on `dsbf.profile()` as in-memory tests of the eight archetypes. The first run found and fixed false-alarm rates of 78% to 93% in `outliers`, `multivariate_outliers`, `autocorrelation` and `reference_drift`, and a detection rate of 0 in `skewness` (mean-median gap) and `reference_drift` (target); the remaining gaps are in the notes of U4-19.

## D. Product and UX

- [ ] **`dsbf dashboard`**: one command that starts FastAPI and serves the built SPA (`npm run build` output mounted as static files). Today it takes two terminals and a Node toolchain. **Status:** Phase 6.
- [ ] **Run management:** `dsbf runs list | show | delete | gc`, plus delete in the UI (F-25). **Status:** Phase 6 (`dsbf runs ...` with the store).
- [ ] **Run comparison UI:** the backend already has `/api/runs/compare`. Build a side-by-side view and a drift view between two runs of the same dataset. This is the first step toward the README's "Dataset comparison" item. **Status:** Phase 6.
- [ ] **Run health panel:** failed and skipped tasks with reasons, reliability caveats, and sampling notices (F-20). **Status:** partial: `Run.health` and `Run.render_health()` exist (U3-12); the stopgap CLI prints it (U4-17); the dashboard panel is Phase 6.
- [ ] **Settings panel:** target column, time index and thresholds, editable from the UI and saved as a per-dataset config overlay, then re-run. **Status:** Phase 6.
- [ ] **Shareable export:** a static HTML or PDF report of a run for people without the stack installed. **Status:** the static HTML report, U5-05.
- [x] **Notebook experience:** `EDA(df).run()` returning a rich object with `_repr_html_`, plus `.open_dashboard()`. **Status:** resolved (U5-02, `Run._repr_html_`, `Findings`); `EDA` is replaced by `dsbf.profile()` (U3-12).

## E. Packaging and distribution

- [x] Use a user data dir or `DSBF_HOME` for the DB and outputs (F-16), with absolute paths stored. **Status:** resolved (U1-04).
- [ ] Ship the built frontend inside the wheel, so `pip install dsbf && dsbf dashboard` works without Node. **Status:** Phase 6.
- [ ] Publish to PyPI, with a release workflow triggered by tags. **Status:** prepared in U5-09 (`release.yml` with trusted publishing, `1.0.0a1`, a hand-kept changelog replacing the generated one); the box is ticked when the user registers the publisher, pushes `v1.0.0a1` and the package is on PyPI.
- [ ] Provide an optional Docker image for team or server use. Document `DSBF_CORS_ORIGINS`, reverse proxying, and that the API has **no auth** and should not be exposed publicly as-is. **Status:** Phase 8.
- [x] Get the version from package metadata (F-18). **Status:** resolved (U3-12).

## F. Quality engineering

- [ ] **API and storage tests:** FastAPI `TestClient` against a fixture DB, plus `persist_run` round-trip tests. **Status:** U0-06 added API v1 tests; they go with API v1 in U4-18, and Phase 6 tests the store and API v2.
- [x] **Golden-run snapshot:** profile a fixed dataset and snapshot `report.json` (with volatile fields stripped), so unintended output changes show up in review. **Status:** resolved for the legacy report (U0-13) and the engine (U3-11); U4-02 adds findings snapshots for the core pack.
- [ ] **Adversarial-frame property tests** (Hypothesis): column names with `|`, brackets, unicode or empty strings; all-null columns; single rows; mixed types; tz-aware datetimes. Every task must return a valid result without crashing (F-10, F-19). **Status:** partial: the adversarial fixture runs against every task (U0-09) and the engine (U3-11); U4-02 adds property-based frames for the core checks.
- [ ] **Frontend tests:** Vitest for chart option builders and utils, plus one Playwright smoke test that loads a fixture run. **Status:** partial: Vitest runs in CI (U1-03); the Playwright smoke test is Phase 6.
- [x] **Type checking in CI** (pyrefly or mypy), which would also make the README badge true. **Status:** resolved (U0-12, pyrefly with a baseline).
- [x] **Coverage reporting**, plus a CI job that runs the validation archetypes. **Status:** the archetype tests are `integration` tests in `tests/validation/` and run with the full suite; the calibration suite runs nightly (U4-16); coverage reporting is in the `full` job's step summary (U5-09, no floor yet).

## G. Longer horizon (from the README vision)

- **Domain packs:** curated task sets and thresholds (healthcare, finance, NLP). These become practical once N-B1 gives tasks typed, overridable params. **Status:** Phase 9; checks already have typed params and packs (U3-01, U3-02).
- **Pipeline hooks:** a `dsbf check` CLI for CI, dbt, Airflow or Prefect. It's report-only by default, with exit codes only from user-defined policies (see `REFACTOR_PLAN.md` §7). **Status:** `dsbf check` and policy exit codes in Phase 5 (U5-01, U5-06); orchestrator adapters in Phase 8.
- **Agent/LLM consumption:** guidance blurbs are already written to be "self-contained for LLM/agent consumption" (`BaseTask.add_guidance`). Exposing runs over an MCP server or a stable JSON schema would let assistants answer "what should I fix before modeling?" directly from a run. **Status:** the run schema is published (U1-11); the MCP server is Phase 8.
- **Next workflow phases:** modeling prep (encoding, imputation and transformation *plans* that can be exported as sklearn pipelines from the existing `actions` blocks), then modeling diagnostics. **Status:** Phase 9; U4-13 keeps the ML guidance actions as findings.
