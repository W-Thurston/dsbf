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
**(verified)** `.github/workflows/ci.yml:22` runs `pip install ruff` without a version, so CI gets
the latest ruff. The project pins `ruff = "^0.11.10"`. Formatter output differs between versions,
so `ruff format --check` fails in CI and the test job never starts. Every Dependabot PR is stuck
behind this.
- [ ] Install ruff at the version in `poetry.lock`, either via `poetry install --only dev` or `pip install "ruff==<locked>"`.
- [ ] Keep `.pre-commit-config.yaml` (`rev: v0.11.10`) in sync with that version, or bump both together.
- [ ] Bump `actions/checkout`/`actions/setup-python` to releases on Node 24. CI already prints Node 20 deprecation warnings.
- [ ] Once CI is green, work through the Dependabot backlog (#24–#28).

### F-02 The test suite writes into the developer's real database and output folder
**(verified)** While the suite was running, `dsbf/dsbf.db` (479 MB) was being modified. The
`clean_engine_run` fixture sets `config["engine"]["output_path"]` (`tests/conftest.py:27`), but the
engine never reads that key. It reads `output_dir` or `metadata.output_dir`. Nothing sets
`DSBF_DB_PATH`, so every engine-level test persists a "titanic" run into the dashboard DB and
writes to `dsbf/outputs/`. Tests also rewrite the tracked `dsbf/static_metadata/task_metadata.json`
(see [F-17](#f-17-every-run-rewrites-tracked-files-and-grows-dsbf_runjson)).
- [ ] Add an autouse session fixture that sets `DSBF_DB_PATH` to a tmp file and `DSBF_AUTO_EXPORT_METADATA=0`, and passes `output_dir=tmp_path`.
- [ ] Have the fixture `chdir` to a tmp dir so CWD-relative writes ([F-16](#f-16-paths-depend-on-the-current-working-directory)) can't leak.

### F-03 Five failing tests
**(verified)**
- `test_detect_bimodal_distribution::test_skips_all_null_column` / `test_skips_constant_column` fail with `KeyError: 'bimodal_flags'`.
- `test_suggest_categorical_encoding::test_high_cardinality_gets_frequency_encoding` fails with `KeyError: 'encoding_suggestions'`.
- `test_normality_tests::test_custom_alpha_respected` fails with `KeyError: 'x'`.
- `test_memory_tracking::test_memory_limit_exceeded_warning` fails because `MemoryTestTask` isn't registered, and `tests/helpers/context_utils.py:51` requires that.

The first two come from this branch's BaseTask migration: `BaseTask.make_empty_result()`
(`dsbf/core/base_task.py:392`) returns `data={}`, which drops the top-level keys each task used to
return (for example `{"bimodal_flags": {}}`). That changes the data contract the frontend and
scorers read. The encoding and normality failures look related but haven't been root-caused yet.
- [ ] Let `make_empty_result()` accept the task's empty data shape (for example `make_empty_result(msg, excluded, data={"bimodal_flags": {}})`), and use it in every migrated task.
- [ ] Confirm the cause of the normality and encoding failures.
- [ ] Register the memory-test task in a fixture, or let `make_ctx_and_task` accept unregistered classes.

### F-04 The default config fails its own validation
**(verified)** `validate_config_and_graph(load_default_config())` returns
`["Unknown task name in config['tasks']: 'time_series'"]`. The shared `time_series` block
(`default_config.yaml:285`) lives under `tasks:` but isn't a task
(`dsbf/utils/config_validation.py:27`). Every run logs a config warning, and with
`safety.strict_mode: true` **every run aborts**.
- [ ] Move shared blocks to a top-level `shared:` or `time_series:` section and update `BaseTask.get_shared_param()`, or whitelist known shared blocks in the validator.

### F-05 Uncommitted config change hard-codes one dataset
The working-tree diff of `dsbf/config/default_config.yaml` sets `time_series.enabled: true`,
`datetime_index_column: GAME_DATE` and `seasonal_period: 7`. Those are NFL-dataset specifics. If
this is committed, every user's run turns on time-series analysis against a column that doesn't
exist. It also contradicts the documented "DSBF will not guess" principle.
- [ ] Revert before committing, and keep dataset-specific settings in a local config passed with `dsbf run -c`.

### F-06 The engine treats a `status="failed"` result as success
**(verified)** `Task.run` (`dsbf/eda/graph.py:34`) marks a task `"success"` whenever no exception
escapes, even if the task returned `TaskResult(status="failed")`. Dependents then run on bad
upstream output (`graph.py:105` only checks `Task.status`), and `task_outcomes` overcounts
successes. This affects `schema_validation` in `fail_or_warn: "fail"` mode, `detect_target_drift`,
and `identify_bottleneck_tasks`, which return failed results instead of raising.
- [ ] Set `Task.status` from `result.status`.
- [ ] **(verified in the practice run)** The reverse happens too. Five tasks that return `status="skipped"` themselves (`schema_validation`, `compare_with_reference_dataset`, `detect_class_imbalance`, `detect_feature_drift`, `detect_target_drift`) are listed under `task_outcomes["success"]`.
- [ ] Separately, about 60 tasks repeat the same `try/except: if self.context: raise / else make_failure_result` block. Move it into a template method on `BaseTask` (see [N-B3](#b-engine-architecture)).

### F-07 Documented config keys are silently ignored
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
- [ ] `detect_mixed_type_columns.py:72`: `bool(get_task_param("ignore_null_type") or True)` is always `True`, so users can't turn it off.
- [ ] Several params use the `get_task_param(x) or default` pattern, so a legitimate `0` or `False` is replaced by the default. Use `get_task_param(x, default)`.
- [ ] The target column is configured four different ways: `target_column: target` in two tasks, `target_column: null` in one, and `target` (not in the config) for target drift. Replace them with one top-level `target_column`.
- [ ] The long-term fix is task-declared parameter schemas ([N-B1](#b-engine-architecture)), which make this class of drift impossible.

### F-08 Stage inference reads the wrong config path and depends on the backend
**(verified)**
- `dsbf/eda/stage_inference.py:8` reads `config["stage_inference"]`, but the default config puts it under `engine.stage_inference`, so the thresholds are ignored.
- The Polars branch (`stage_inference.py:33`) compares dtypes to the strings `"i64"`/`"f64"`, which never match. The same 4×2 numeric frame is inferred as `model_ready` with pandas and `cleaned` with Polars. Built-in datasets default to Polars.
- The inferred labels (`raw`/`cleaned`/`model_ready`/`exploratory`) don't match the task stage vocabulary (`raw`/`cleaned`/`modeling`/`report`/`any`), so the inferred stage can't drive task selection.
- [ ] Fix the config path, use `df.schema` numeric checks for Polars, and unify the stage vocabulary.

### F-09 Polars and `sample_strategy: random` crash
**(verified)** `dsbf/utils/data_utils.py:24` calls `d.sample(n=..., random_state=42)`. Polars uses
`seed=`, so this raises `TypeError`. `"stratified"` is a silent alias for `"head"`, which biases
any ordered file.
- [ ] Branch on backend, and either implement stratified sampling or reject it with a clear error.

### F-10 Column names containing `|` crash pair-based tasks
**(verified)** Pair results are keyed as `"col1|col2"` and split with `split("|")`. A column named
`a|b` makes `detect_data_leakage` raise `ValueError: too many values to unpack`
(`detect_data_leakage.py:110`). The API uses `split("|", 1)` (`dsbf/api/main.py:224`, `:432`), which
silently mis-attributes the pair instead.
- [ ] Store pairs as `{"a": ..., "b": ..., ...}` records, or use a nested `{col1: {col2: ...}}` map, instead of encoded string keys.

### F-11 Semantic-type filtering is still bypassed in places
v0.20.0 fixed tasks iterating over `df.columns` instead of `matched_cols`. The same bug class
remains: `detect_data_leakage.py:73` computes correlations over
`df.select_dtypes(include="number")`, so integer-encoded booleans and IDs enter the leakage scan
and could produce false "Leakage" findings. 21 task files use `select_dtypes`.
- [ ] Audit each one and intersect with `matched_cols`, or replace it with `matched_cols`.
- [ ] Add a regression test: an int-coded boolean column must not appear in any continuous-only task output.

### F-12 The new histogram misrepresents distributions
`DistributionHistogramChart.vue:37` builds bins between the min, 1/5/25/50/75/95/99th percentiles
and the max, then plots each bin's *count* as an equal-width bar. The bins have very different
widths but similar counts by construction, so the shape you see mostly reflects the percentile
choice, not the data. A uniform distribution would look peaked. Given the project's
"epistemic honesty" principle, this should not ship.
- [ ] Emit real equal-width bin edges and counts from `summarize_numeric` (Freedman–Diaconis or Sturges, via `np.histogram`), and render those.
- [ ] As a stopgap, plot density (`count / width`) on a value axis instead of a category axis.

### F-13 The "Kendall's τ for n < 30" routing is unreachable with defaults
**(verified)** `compute_pairwise_associations.py:517` skips pairs with
`valid_n < min_sample_size` (default 30) *before* method routing at `:298`, so the
`auto → kendalls_tau` branch never runs with the shipped config. README and CHANGELOG both
advertise it.
- [ ] Lower the default gate (for example to 10) or route before gating, and add a test using a 20-row frame.

### F-14 Stored XSS via dataset name
`DatasetListView.vue:34` and `RunListView.vue:48-49` render `highlight(text, query)` with `v-html`.
`highlight()` (`DatasetListView.vue:86`) regex-escapes the *query* but never HTML-escapes the
*text*. A dataset named `<img src=x onerror=...>` (for example via `dsbf profile f.csv --name ...`)
executes in the dashboard. The risk is low for a local tool but real once a dashboard is shared.
- [ ] HTML-escape `text` before inserting `<mark>`, or render matches as text nodes. Deduplicate the helper into `utils.js`.

## P1: Robustness and operational correctness

### F-15 Run identity and dataset identity
- [ ] **run_key collisions:** `run_key` is a second-resolution timestamp (`dsbf/core/base_engine.py:93`). Two runs in the same second share an output dir (`exist_ok=True` overwrites), and `_insert_run` silently reuses the first run's row, merging task results. Add a short random suffix or a UUID.
- [ ] **`ran_at` can be NULL:** with a custom `output_dir`, `run_key` isn't a timestamp, `ran_at` becomes `None` (`writer.py:313`), and the explicit NULL overrides the column default. Those runs then sort last and break "latest run" queries. Use the actual start time.
- [ ] **Datasets are keyed by name only.** Two different files profiled under the same name are merged. `source_path` is set once and never updated (`writer.py:42`), so later runs point at the first file. The `EDA()` facade and default config both fall back to `"default_dataset_name"`.
- [ ] **The dashboard can show different data than what was profiled.** `/sample` and `/column-data` re-read whatever is at `source_path` *now*. That file may have changed, may be a different file (above), and ignores any sampling applied during the run. Store `source_path`, file size, mtime and a content hash per run, and warn in the UI when they no longer match.

### F-16 Paths depend on the current working directory
- `dsbf/outputs/<ts>` (`base_engine.py:93`), `dsbf_run.json` (`:99`) and `dsbf/static_metadata/task_metadata.json` (`task_registry.py:279`) are all relative to CWD.
- The API resolves figure and source paths relative to the **repo root** (`dsbf/api/main.py:507`, `dsbf/api/db.py:150`).
- The default DB lives inside the package (`dsbf/storage/schema.py:26`).

Running `dsbf profile` from any other directory leaves the dashboard with broken figures. Once the
package is pip-installed, it would write into `site-packages`.
- [ ] Introduce one `DSBF_HOME` (default: a `platformdirs` user data dir, or `./.dsbf` per project), store absolute paths in the DB, and resolve everything from it.

### F-17 Every run rewrites tracked files and grows `dsbf_run.json`
- `load_task_group()` calls `write_task_metadata()` on **every run** (`dsbf/eda/task_registry.py:276`, enabled by default). It rewrites the tracked `task_metadata.json` and strips its trailing newline, which fights the `end-of-file-fixer` pre-commit hook. That's why the file is always dirty in `git status`.
- `record_run()` appends to `dsbf_run.json` with no bound. It's currently 442 entries and 240 KB, and the SQLite `runs` table already holds the same information.
- [ ] Make metadata export an explicit command (for example `dsbf tasks export`) and stop tracking the generated JSON, or generate it in CI.
- [ ] Remove `record_run()`, or cap it.

### F-18 Version and git SHA describe the wrong repository
`get_dsbf_version()` runs `git describe` and `_get_git_sha()` runs `git rev-parse` in the
**current directory** (`dsbf/utils/versioning.py:16`, `base_engine.py:125`). When DSBF is used
from inside another project's repo, run metadata records *that* project's tag and SHA. The API also
reports `version="0.1.0"` (`dsbf/api/main.py:45`).
- [ ] Use `importlib.metadata.version("dsbf")` everywhere, and only record a SHA when the package itself is installed from a git checkout.

### F-19 Logging
- Task logs go through `AnalysisContext._log` to the `dsbf.context` logger, which has **no file handler** and `propagate=False`. As a result, `run.log` misses task-level messages, including memory/runtime limit warnings and TaskResult validation failures.
- `setup_logger` calls `handlers.clear()` without closing them (`logging_utils.py:64`), which leaks one open `run.log` file descriptor per run in long-lived processes such as notebooks or a future API-triggered runner.
- `logging.setLoggerClass(DSBFLogger)` at import time (`logging_utils.py:30`) changes logger creation for every library imported afterwards.
- `RichHandler(markup=True)` interprets brackets in messages. **(verified)** A column named `[bold]` disappears from the log. A message containing `[/x]` raises a `MarkupError` inside the handler, which `logging` catches, so the message is lost and a "Logging error" traceback is printed. Some call sites escape with `\\[` (and that escape then leaks into `run.log` as a literal `\[`); most don't.
- `level="error"` isn't mapped in `get_log_fn` and falls back to INFO2.
- [ ] Use one `dsbf` logger hierarchy with child loggers that propagate, a file handler attached once per run and removed and closed at the end, `markup=False` (or escape user data), and an `error` level.

### F-20 Failure visibility end to end
- Skipped tasks (failed dependency) get **no TaskResult** (`graph.py:105-117`), so they're missing from `report.json`, the DB and the dashboard. Users can't tell "not applicable" from "skipped because X failed".
- The DB stores only `status/summary/data/guidance` (`dsbf/storage/writer.py`). `error_metadata`, `reliability_warnings`, `recommendations` and `metadata` (for example `memory_exceeded` and `excluded_columns`) are dropped, so the dashboard can't explain failures or show reliability caveats.
- `TaskResult.to_dict()` (`task_result.py:74`) omits `recommendation_tags`.
- `error_to_metadata()` (`task_result.py:116`) keeps only `str(e)` and no traceback.
- [ ] Emit `status="skipped"` results with the blocking dependency, persist the full TaskResult (one JSON column or extra columns plus a schema migration), and add a "Run health" panel in the dashboard listing failed and skipped tasks with reasons.

### F-21 Ordering relies on implicit DAG levels
`data_quality_scorer` deliberately leaves `detect_collinear_features` and `detect_out_of_bounds`
out of `depends_on` (`data_quality_scorer.py:127`) so their failures don't skip the scorer. It
still reads their results, and only gets them because both happen to sit at a lower topological
level. Any change to their dependencies would silently empty the Redundancy and Validity
dimensions.
- [ ] Add soft dependencies (`runs_after=[...]`): ordering edges that don't propagate failure.

### F-22 Configuration loading
- [ ] `dsbf run -c my.yaml` uses the YAML **as the whole config** (`dsbf/interfaces/cli.py:53`). A partial user config loses every default task parameter. Deep-merge user config over defaults. The `default_config.yaml` comment already promises a `user_config.yaml` override that doesn't exist.
- [ ] `EDA(...)` (`dsbf/interfaces/api.py`) mutates the caller's config dict, raises `KeyError` when the config has no `metadata` block (`:36`), and supports in-memory data by monkeypatching `engine._load_data` (`:48`). Add a first-class `ProfileEngine(config, data=df)` parameter.
- [ ] Validate enum-like values (`profiling_depth`, `backend`, `runtime_estimate`, `sample_strategy`) up front. An invalid depth currently raises a bare `KeyError` in `build_graph`.
- [ ] Dead config: `metadata.output_format`, `engine.severity_thresholds`, `layout_name` (`dsbf/layouts/default.json` is unused), and the comment in `default_config.yaml` about a "dashboard Settings → Time Series panel" that doesn't exist yet.

### F-23 Plugin loading
- [ ] The `core` task group imports `dsbf.custom_plugins.core`, which is **empty**. Core tasks are actually registered as an import side effect of `dsbf/utils/task_utils.py:14` (`load_all_tasks()` at module import). Make registration explicit.
- [ ] Loading a plugin directory twice in one process (a notebook or the API) re-executes the module, hits "already registered", and logs a traceback. `PLUGIN_WARNINGS` is a global that accumulates across runs and is never cleared.
- [ ] Plugin modules are registered in `sys.modules` by file stem (`task_registry.py:296`), so two plugin dirs that each have a `utils.py` overwrite each other.
- [ ] A non-directory, non-module `task_groups` entry fails with a printed traceback instead of a clear message. `if not group_path.exists()` inside `if group_path.is_dir()` is dead code.

### F-24 Resource tracking is misleading
`peak_mem = max(rss_before, rss_after)` (`graph.py:126`) is neither a peak nor per-task, because
RSS is process-wide and rarely shrinks. It also mixes units (`/1e6` vs `* 1024`).
`max_runtime_seconds` only warns.
- [ ] Measure per-task deltas (or use `tracemalloc` / a sampling thread for peaks), label them honestly, and optionally enforce timeouts ([N-B5](#b-engine-architecture)).

### F-25 API and data-layer issues
- [ ] `/sample` reads the **entire source file** to return 10 rows (`dsbf/api/db.py:193`). Use `nrows=`. For built-in datasets, each request re-downloads from seaborn or sklearn. Cache, or snapshot a sample at profile time.
- [ ] `_load_dataframe` returns `None | DataFrame | {"error": ...}`. `/column-data` passes the error dict back as an HTTP 200 payload. Raise and map to 4xx/5xx instead.
- [ ] `/associations/{column}` and `/correlations/{column}` sort by `abs(metric)` across metric types. η² (where 0.14 is "strong") is ranked against r (where 0.7 is "strong"), so the ordering is meaningless across types. Sort by strength rank, then by metric within type.
- [ ] The endpoint list in the module docstring is stale. Several handlers ignore `_DB_PATH`, which works only because `schema.py` re-reads the env var.
- [ ] Add a delete/GC path (`DELETE /api/runs/{key}` and `dsbf runs delete|gc`). `clean_dsbf_db.txt` currently documents manual SQL. DB rows outlive deleted output dirs (only 1 of many runs still has files on disk), which leaves orphaned figure rows.
- [ ] `runs.quality_score` is always NULL (`writer.py`), but `DatasetListView` still renders a "Quality" column from it, so it always shows "-". Drop the column (with a migration) or replace it with the five-dimension summary.
- [ ] Add a `schema_version` table and real migrations instead of `ALTER TABLE ... except OperationalError`.
- [ ] `frontend/src/api.js`: `getDataset`, `listRuns` and `getTask` don't `encodeURIComponent` their path params, so names containing spaces, `/` or `#` break routing.

### F-26 "Leakage" is mislabeled
The Leakage dimension is driven by `detect_data_leakage`, which flags *feature–feature* pairs with
|r| ≥ 0.99. That's redundancy or duplication. Target leakage means a feature encodes information
about the target that wouldn't be available at prediction time, and detecting it requires a target.
Calling feature collinearity "leakage" conflicts with the hedged-language principle.
- [ ] Fold the current check into Redundancy. Redefine Leakage around a configured `target_column` (near-perfect feature→target association, post-event timestamps, target-derived IDs). If no target is set, show "not assessed" rather than green.

## P2: Hygiene, docs and developer experience

### F-27 README accuracy
- [ ] It says "Python 3.10+", but `pyproject.toml` requires `>=3.12`.
- [ ] `from dsbf.api import profile_file` and `from dsbf.api import ProfileEngine` don't exist (`dsbf/api/__init__.py` is empty and `dsbf.api` is the FastAPI package). The in-memory example (`engine.df = ...`) has no effect. The real facade is `dsbf.interfaces.api.EDA`.
- [ ] The default depth is documented as `standard`, but `default_config.yaml` ships `full`, along with `message_verbosity: debug` and `visualize_dag: true`.
- [ ] It says "D3-native chart components", but the refactor in progress uses ECharts.
- [ ] It shows a "Type checked: Pyrefly" badge, but no type checker is configured or run.
- [ ] Fold the README's "Roadmap" section into a link to this file.

### F-28 Repository clutter
- [ ] Untracked or stray files: root `package-lock.json` (empty), `clean_dsbf_db.txt`, `dsbf_reference_dataset.csv` (5.3 MB), `dsbf_test_schema.json`, `tests/generators/dsbf_test_generator_backup.py`, `dsbf_directory_structure.md`. Delete them, or move them under `docs/`, `examples/` or `tests/fixtures/`.
- [ ] Move `dsbf_handoff_doc.md` into `docs/architecture.md`. It's the best architecture doc in the repo.
- [ ] `.flake8` is obsolete now that ruff is used. `run_profile.py` and `migrate_runs.py` at the root should become CLI subcommands.
- [ ] Remove dead code: `dsbf/utils/check_recommendation_contexts.py`, `dsbf/utils/display_registered_tasks.py`, `dsbf/layouts/default.json`, and the commented-out `render_dashboard` in `cli.py`.
- [ ] Commit the regenerated `poetry.lock`, which removes `panel` and its transitive deps to match `pyproject.toml`.

### F-29 Frontend build and code health
- [ ] Unpin `vite ^8.0.0-beta.13` and its `overrides` block and move to a stable release.
- [ ] Bundle size: Plotly (via `PlotCard.vue:35` static import, which also defeats `PairPlotCard`'s dynamic import) plus ECharts ship together in one 5.5 MB chunk. Finishing the ECharts migration ([N-A1](#a-finish-in-flight-work)) and route-level code splitting should cut this substantially.
- [ ] Tab components are monoliths: `QualityTab.vue` (1.5k lines), `TimeSeriesTab.vue` (1.2k) and `MlReadinessTab.vue` (1.1k). Extract cards and composables (`useRunTasks`, `useTheme`).
- [ ] Add ESLint, Prettier and a `vite build` job to CI.

### F-30 Engine performance
- [ ] Each pandas-dependent task calls `to_pandas()` on the Polars frame independently (`BaseTask.setup_run` / `get_dataframe_pandas`), which can mean up to about 60 full copies per run. Convert once and cache on `AnalysisContext`.
- [ ] Data is fully loaded before sampling. For large files, sample at read time (Polars `scan_*` with `head` or `sample`).
- [ ] `_load_data` ignores `engine.backend` for files and always returns pandas. The reference dataset only supports CSV.

## Practice-run findings (2026-10-15)

These come from an instrumented, isolated run on four datasets: titanic, the repo's kitchen-sink
generator at 5k and 50k rows, and an adversarial frame. The stage-by-stage measurements are in
[`WALKTHROUGH.md`](WALKTHROUGH.md). Everything below is **verified**.

### F-31 Tasks crash when run outside the repo root
`dsbf/utils/reco_engine.py:7` opens `Path("dsbf/static_metadata/recommendation_library.yaml")`
relative to the CWD. Twelve tasks call `get_recommendation_tip()` inside their main `try` block, so
an optional "tip" lookup raises `FileNotFoundError` and fails the whole analysis. The failures
cascade: `data_quality_scorer` and `ml_readiness_scorer` are skipped, so the **Quality bar and ML
Readiness tab are empty**, as are `transformation_preview` and `normality_qq_plots`. This affects
every pip-installed user and anyone running from a different directory.
- [ ] Load package data with `importlib.resources.files("dsbf.static_metadata")`. Audit every other `Path("dsbf/...")` / `open("dsbf/...")` call.
- [ ] Make enrichment non-fatal: a missing tip should produce no tip, not a failed task.
- [ ] `get_recommendation_tip()` runs `eval(condition, {}, context_vars)` on YAML strings (`reco_engine.py:28`). With empty globals, Python still injects builtins. Replace it with a small declarative condition format (`{"field": ..., "op": ">", "value": ...}`).
- [ ] Add a CI job that runs one profile from a temp directory against an installed wheel.

### F-32 Listwise deletion across all numeric columns
`numeric_df = df.select_dtypes("number").dropna()` drops every row with *any* null in *any*
numeric column:
- `compute_reliability_flags` (`dsbf/utils/reliability_stats.py:18`) used **0 rows** on kitchen5k and adversarial, and 714 of 891 on titanic. Global reliability flags are computed on an empty or biased frame.
- `detect_collinear_features.py:85` crashes on `zero-size array` because the guard checks the column count, not the row count. This is the crash the `data_quality_scorer` comment works around (F-21).
- [ ] Compute per-column statistics on per-column non-null values. For multivariate methods (VIF), drop columns above a missingness threshold first, then rows, and record `n_used` in the result.
- [ ] Guard on `n_rows < k + 1`, not only on column count. Grep for the pattern in all tasks.

### F-33 Crashes on all-null columns and `None` values
- `summarize_modes.py:59`: `col_modes[0]` on an all-null column raises `IndexError`. The task also iterates `df.columns` instead of `matched_cols`.
- `normality_tests.py:223`: a **debug log line** formats `p_value=None` with `:.4f`, raising `TypeError` and failing the task. The f-string is evaluated even when debug logging is off.
- [ ] Fix both. Adopt the standard that log calls never do formatting that can raise: pre-format with a helper that renders `None` as `n/a`, or use `%`-style lazy logging.
- [ ] The adversarial frame used in the practice run should become a permanent fixture that every task runs against (N-F, adversarial property tests).

### F-34 Wasted post-processing every run
- `build_column_task_mappings` re-serialises every result via `to_dict()` (0.67 s on titanic) and treats every `data` key as a column name. The map is polluted with `__dataset__`, `__correlation_matrix__`, `age|class|First_vs_Second`, and so on. Use `task_column_map` from the columns each task actually processed (`matched_cols`).
- `visualize_dag: true` by default draws a 400 KB `dag.png` on every run (about 1 s), and it's identical for an identical task set. Make it opt-in, or cache it by task-set hash.
- `TaskResult._sanitize()` and `json.dumps(indent=2)` run over the full payload several times (report, mapping, DB). Serialise once and reuse the bytes.

### F-35 Fuzzy duplicate detection dominates runtime and output, and its results are arbitrary
It took **58.8 s of 72.5 s** of task time on titanic (891 rows) and **172 s of 296 s** on kitchen5k.
It produced **8.6 MB** of data on titanic, which made `report.json` **18.5 MB** (5.3 s to write) and
the dashboard's on-open `/tasks` response **8.2 MB**.

Efficiency:
- [ ] On titanic, ~42 s were ~395,000 `working_df.iloc[i][col]` lookups (`fuzzy_duplicate_detection.py:374-377`). Pre-extract the string columns once as a NumPy array of row strings.
- [ ] On kitchen5k, ~148 s were pure-Python `difflib.SequenceMatcher` on long text rows. Use `rapidfuzz` (C++, typically 50–100× faster) and truncate or hash very long fields.

Correctness:
- [ ] Blocking keeps any pair that shares *one* token. On data with categorical columns, every row shares tokens like `male` or `southampton`, so blocking does nothing and the step stops at `max_candidates=50,000`. That's an **arbitrary ~12.6% of titanic's ~396k pairs**, and nothing in the output says the result is partial. Block on rare tokens (IDF-weighted or MinHash/LSH), exclude low-cardinality categoricals from the row string, and report `pairs_compared / pairs_possible`.
- [ ] 16,316 flagged pairs on titanic suggests the 0.85 threshold on concatenated row strings mostly measures "same categorical values", not near-duplicate records. Calibrate against the planted near-duplicates in the kitchen-sink generator.
- [ ] Store at most the top N pairs plus counts. Fetch details on demand.

### F-36 `NaN` / `Infinity` in stored data make API endpoints return HTTP 500
The writer stores task data with `json.dumps(..., default=str)`, which emits the non-standard tokens
`NaN` and `Infinity`. FastAPI's encoder refuses them on the way out. On the adversarial run,
`/associations` (a `NaN` metric) and `/sample` (an `inf` in the source data) both returned **500**.
`TaskResult._sanitize()` converts NumPy scalars but leaves float `nan`/`inf` alone.
- [ ] Map non-finite floats to `None` in `_sanitize()` (or emit a sentinel object), write with `allow_nan=False` so violations fail loudly at write time, and handle `inf` in `/sample` serialisation.

### F-37 `infer_types` parses entire columns as dates
`_is_datetime_like` (`infer_types.py:250`) took 8.6 of `infer_types`' 9.0 s on kitchen5k (25 object
columns). Every downstream task waits on `infer_types`.
- [ ] Test a sample of a few hundred non-null values first, try known formats before falling back to dateutil, and only parse the full column for columns that pass.

### F-38 `generate_univariate_plots` cost and output
It took 84 s on kitchen5k and wrote 361 files, including **two 17–18 MB PNGs** for a column of
extremely long strings, because bar-chart labels are full category values. It would be removed
entirely by N-A2. Until then:
- [ ] Truncate labels, cap categories to the top-k, render one theme and let the frontend restyle it, and skip text-intent columns.

### F-39 `detect_stationarity` scales super-linearly
On 10× more rows (kitchen5k → kitchen50k) it went from **4.9 s to 186 s (38×)**.
`detect_stationarity.py:192` calls `adfuller(series.values, autolag="AIC")` with no `maxlag`.
statsmodels' default max lag grows with n (about 57 at 52k rows), and it fits one regression per
candidate lag. It also tests the raw rows in file order rather than the series resampled to the
configured frequency. With tens of thousands of points, ADF and KPSS reject on trivially small
deviations anyway.
- [ ] Resample or aggregate to the configured or inferred frequency before testing, cap `maxlag` (configurable), and cap the series length used for testing (for example the most recent N periods). Record all three in the result.
- [ ] Apply the same review to `compute_acf_pacf` and `decompose_time_series`.

### Overall run cost (for tracking)
kitchen50k (52,500 × 60) took **707 s** with a 2.1 GB peak. Five tasks (`detect_stationarity`,
`fuzzy_duplicate_detection`, `generate_univariate_plots`, `infer_types`,
`compute_pairwise_associations`) account for about 90% of that. Turning the practice-run scripts
into a `tests/perf/` benchmark that fails on large regressions would keep this from creeping back.

---

# Part 2: Next steps

## A. Finish in-flight work

| ID | Item | Notes |
|---|---|---|
| N-A1 | **Complete the ECharts migration** | Move the remaining Plotly cards (`PlotCard`, `PairPlotCard`, correlation heatmap, missingness) to `DsbfChart`, then remove `plotly.js-dist-min`. Charts should render from task *data*, not pre-rendered figure files. |
| N-A2 | **Retire backend plot generation** | Once the frontend no longer reads figure files, remove `generate_univariate_plots`, `generate_dataset_summary_plots` and `dsbf/utils/plot_factory.py` (1.2k lines), along with the `matplotlib`/`seaborn`/`missingno`/`plotly`/`pydot`/`graphviz` deps (keep seaborn only if built-in datasets still need it). This shrinks install size and run time and removes the figure-path problems in [F-16](#f-16-paths-depend-on-the-current-working-directory). |
| N-A3 | **Q-Q plot** | `normality_qq_plots` already computes the quantiles. The Distributions tab shows a "coming soon" placeholder. |
| N-A4 | **Time Series tab** | Finish ACF/PACF, stationarity and decomposition views. Add a friendly empty state and a UI path to set `datetime_index_column`. |
| N-A5 | **Findings tab** | Cross-column finding explorer (already on the README roadmap). It can be built entirely on the `guidance` blobs every task emits. |

## B. Engine architecture

| ID | Item | Why |
|---|---|---|
| N-B1 | **Typed config with task-declared params** | Each task declares its parameters, defaults and docs (pydantic or dataclass). `default_config.yaml`, validation, CLI help and docs are generated from those declarations. This removes the whole class of bugs in F-04, F-07 and F-22, and gives the dashboard a schema for a Settings panel. |
| N-B2 | **Versioned TaskResult data schemas** | Each task declares the shape of `data`, and contract tests assert it, including the empty-result shape (F-03). Export JSON Schema or TypeScript types for the frontend, which today reads deeply nested `data?.x?.[col]` paths with no contract. |
| N-B3 | **Template-method task execution** | `BaseTask.execute()` handles setup, column matching, empty results, exception wrapping, timing and reliability flags. Subclasses implement `compute(df, cols)`. This removes about 60 copies of boilerplate. |
| N-B4 | **Soft dependencies and real skip semantics** | `runs_after=` ordering edges (F-21) plus first-class `skipped` results (F-20). |
| N-B5 | **Parallel execution and timeouts** | Run independent tasks in each DAG level concurrently, with a configurable worker pool. The DAG and `runtime_estimate` metadata already exist. Enforce `max_runtime_seconds` per task. |
| N-B6 | **Large-data path** | Lazy Polars loading, read-time sampling with recorded provenance, and reliability warnings that say explicitly when results come from a sample. |

## C. Statistical rigor

This is the project's core differentiator, so these items deserve priority after the P0 fixes.

- [ ] **Real histograms** (F-12): equal-width bins computed server-side, with the binning rule recorded in metadata.
- [ ] **Uncertainty on associations:** confidence intervals (Fisher z for r and τ, bootstrap for η² and V) and the n behind each estimate. Display "weak (95% CI 0.05–0.41, n=48)" instead of a bare label.
- [ ] **Bias-corrected Cramér's V** (Bergsma 2013). The uncorrected V currently inflates on small samples and high-cardinality pairs.
- [ ] **A comparable association scale** across metric types (F-25). For example, convert η² to η, or rank by strength band and never by raw value across types.
- [ ] **Multiple-comparison awareness** in pairwise scans on wide data. Report how many pairs were tested next to the number flagged.
- [ ] **Target-aware leakage detection** (F-26) built on a single `target_column`.
- [ ] **Calibration suite:** extend `tests/validation/` with datasets that have *known* ground truth (planted leakage, planted MAR or MNAR, planted bimodality), and track false-positive and false-negative rates per task across releases.

## D. Product and UX

- [ ] **`dsbf dashboard`**: one command that starts FastAPI and serves the built SPA (`npm run build` output mounted as static files). Today it takes two terminals and a Node toolchain.
- [ ] **Run management:** `dsbf runs list | show | delete | gc`, plus delete in the UI (F-25).
- [ ] **Run comparison UI:** the backend already has `/api/runs/compare`. Build a side-by-side view and a drift view between two runs of the same dataset. This is the first step toward the README's "Dataset comparison" item.
- [ ] **Run health panel:** failed and skipped tasks with reasons, reliability caveats, and sampling notices (F-20).
- [ ] **Settings panel:** target column, time index and thresholds, editable from the UI and saved as a per-dataset config overlay, then re-run.
- [ ] **Shareable export:** a static HTML or PDF report of a run for people without the stack installed.
- [ ] **Notebook experience:** `EDA(df).run()` returning a rich object with `_repr_html_`, plus `.open_dashboard()`.

## E. Packaging and distribution

- [ ] Use a user data dir or `DSBF_HOME` for the DB and outputs (F-16), with absolute paths stored.
- [ ] Ship the built frontend inside the wheel, so `pip install dsbf && dsbf dashboard` works without Node.
- [ ] Publish to PyPI, with a release workflow triggered by tags. A changelog workflow already exists.
- [ ] Provide an optional Docker image for team or server use. Document `DSBF_CORS_ORIGINS`, reverse proxying, and that the API has **no auth** and should not be exposed publicly as-is.
- [ ] Get the version from package metadata (F-18).

## F. Quality engineering

- [ ] **API and storage tests:** FastAPI `TestClient` against a fixture DB, plus `persist_run` round-trip tests.
- [ ] **Golden-run snapshot:** profile a fixed dataset and snapshot `report.json` (with volatile fields stripped), so unintended output changes show up in review.
- [ ] **Adversarial-frame property tests** (Hypothesis): column names with `|`, brackets, unicode or empty strings; all-null columns; single rows; mixed types; tz-aware datetimes. Every task must return a valid result without crashing (F-10, F-19).
- [ ] **Frontend tests:** Vitest for chart option builders and utils, plus one Playwright smoke test that loads a fixture run.
- [ ] **Type checking in CI** (pyrefly or mypy), which would also make the README badge true.
- [ ] **Coverage reporting**, plus a CI job that runs `tests/validation/run_validation.py`.

## G. Longer horizon (from the README vision)

- **Domain packs:** curated task sets and thresholds (healthcare, finance, NLP). These become practical once N-B1 gives tasks typed, overridable params.
- **Pipeline hooks:** a `dsbf check` CLI for CI, dbt, Airflow or Prefect. It's report-only by default, with exit codes only from user-defined policies (see `REFACTOR_PLAN.md` §7).
- **Agent/LLM consumption:** guidance blurbs are already written to be "self-contained for LLM/agent consumption" (`BaseTask.add_guidance`). Exposing runs over an MCP server or a stable JSON schema would let assistants answer "what should I fix before modeling?" directly from a run.
- **Next workflow phases:** modeling prep (encoding, imputation and transformation *plans* that can be exported as sklearn pipelines from the existing `actions` blocks), then modeling diagnostics.
