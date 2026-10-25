# How a DSBF run works: step by step

> **This describes the legacy run, which was removed.** It traces `ProfileEngine`, the task DAG,
> the SQLite store and the FastAPI backend as they were on 2026-10-15, with measurements from
> that date, and is kept as the record the review findings and the refactor plan refer to. The
> code it names no longer exists. How a run works now is in
> [`architecture.md`](architecture.md) and [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md).

This document traces one profiling run from the command line to the dashboard. For each stage it
gives:

- what happens,
- where the code lives,
- what the stage reads and writes,
- what it **measured** in an instrumented practice run on 2026-10-15,
- known issues (IDs refer to [`REVIEW_FINDINGS.md`](REVIEW_FINDINGS.md)).

> Source paths below (`dsbf/eda/...`) were measured before the `src/` layout move on 2026-10-16;
> they now sit under `src/dsbf/`.

## The practice run

The instrumented run wrapped every stage that `ProfileEngine.run()` calls with a wall-time and RSS
timer, profiled the whole run with cProfile, and then probed every dashboard API endpoint. It was
fully isolated: a scratch `DSBF_DB_PATH`, a scratch output dir and working directory, and
`DSBF_AUTO_EXPORT_METADATA=0`. Time-series settings matched the committed defaults. Datasets:

| Run | Shape | Load path | Why |
|---|---|---|---|
| `titanic` | 891 × 15 | built-in (seaborn → Polars) | Realistic small dataset |
| `kitchen5k` | 5,250 × 60 | CSV file (pandas) | The repo's own pathological-column generator |
| `kitchen50k` | 52,500 × 60 | CSV file (pandas) | Scaling behaviour |
| `adversarial` | 200 × 12 | CSV file (pandas) | Column names with `\|`, `[bold]`, `[/x]`, unicode, `""`; all-null, `inf`, int-coded booleans, tz-aware datetimes |

Headline numbers:

| Run | Wall time | Task time | Peak RSS (OS) | Tasks failed / self-skipped / graph-skipped |
|---|---|---|---|---|
| titanic | 80 s | 72.5 s | — | 4 / 5 / 2 |
| kitchen5k | 298 s | 296 s | 1.8 GB | 12 / 5 / 4 |
| adversarial | 29 s | — | — | 8 / 5 / 3 |
| kitchen50k | 707 s | 704 s | 2.1 GB | 12 / 5 / 4 (see [Scaling](#scaling-kitchen50k)) |

The practice run ran outside the repo root, the way it would after `pip install`. Most of the
failures come from that ([F-31](REVIEW_FINDINGS.md#f-31-tasks-crash-when-run-outside-the-repo-root)).

---

## Overview

```
dsbf profile data.csv                                   (dsbf/interfaces/cli.py)
   │
   ├─ 0. import dsbf ──► registers all 67 tasks as an import side effect
   ├─ 1. load config  ──► default_config.yaml (+ CLI overrides)
   ├─ 2. ProfileEngine(config)  ──► output dir, logger, git metadata
   └─ 3. engine.run()                                   (dsbf/eda/profile_engine.py:50)
        ├─ 3a  load data            _load_data()
        ├─ 3b  sample               data_sampling()
        ├─ 3c  reference data       (optional CSV)
        ├─ 3d  build context        AnalysisContext(...)
        ├─ 3e  load task groups     load_task_group()  (+ plugins)
        ├─ 3f  validate             validate_config_and_graph()
        ├─ 3g  infer data stage     infer_stage()
        ├─ 3h  build DAG            build_graph()
        ├─ 3i  RUN TASKS            ExecutionGraph.run()        ◄── ~99% of runtime
        ├─ 3j  column↔task maps     build_column_task_mappings()
        ├─ 3k  DAG image            graph.visualize()
        ├─ 3l  report.json          render_user_report()
        ├─ 3m  metadata_report.json write_metadata_report()
        ├─ 3n  SQLite               persist_run()

uvicorn dsbf.api.main:app   ──►  4. FastAPI reads SQLite + files
npm run dev                 ──►  5. Vue dashboard calls /api/...
```

---

## 0. Import and task registration

**What happens:** importing `dsbf.eda.profile_engine` pulls in `dsbf.utils.task_utils`, which calls
`load_all_tasks()` **at module import time** (`dsbf/utils/task_utils.py:14`). That imports every
module in `dsbf/eda/tasks/`, and each `@register_task` decorator adds a `TaskSpec` to the global
`TASK_REGISTRY` (`dsbf/eda/task_registry.py:71`). Importing changes no logging state: every module
logs to a `dsbf.*` logger, and `dsbf` carries only a `NullHandler`.

**Measured:** about 2.0 s, and about 300 MB RSS before any data is loaded. Most of that is
eager imports of seaborn, matplotlib, sklearn, statsmodels and the plot factory.

**Issues:** registration depends on import side effects, while the `core` task group it's supposed
to come from is an empty module (F-23). Every run
pays for plotting libraries even after the ECharts migration makes them unnecessary (N-A2).

## 1. Load configuration

**What happens:** `load_default_config()` (`dsbf/config/__init__.py`) reads `default_config.yaml`.
`dsbf profile` patches `dataset_path`, `profiling_depth` and `dataset_name`. `dsbf run -c` uses the
user's YAML **instead of** the defaults (no merge).

**Measured:** negligible.

**Issues:** no deep merge with defaults (F-22). Several documented keys are ignored by their tasks
(F-07).

## 2. Construct the engine

**What happens:** `BaseEngine.__init__` (`dsbf/core/base_engine.py`):
- picks the output dir (`<DSBF_HOME>/outputs/<YYYYmmdd_HHMMSS>`, absolute; the home defaults to the per-user data directory) and creates it;
- attaches no log handler (the CLI adds a console handler around the run, and `ProfileEngine.run` adds the `run.log` file handler for the length of the run);
- runs `git describe` and `git rev-parse` subprocesses for version and SHA metadata.

**Measured:** 6 ms.

**Issues:** it has side effects on disk before `run()` is called. The run_key has one-second
resolution (F-15). Paths depend on CWD (F-16). Git metadata describes whatever repo you're standing
in (F-18).

## 3a. Load data

**What happens:** `_load_data()` (`profile_engine.py:159`). With `dataset_path` set it reads
CSV/Parquet/Excel **with pandas, always**. Otherwise it loads a built-in seaborn, sklearn or OpenML
dataset and converts it to Polars if `engine.backend: polars` (the default).

**Measured:** 18 ms (titanic) and 84 ms (kitchen5k).

**Issues:** the same config gives you pandas for files and Polars for built-ins, so behaviour
differs by source (F-08, F-30). The whole file is read before sampling (F-30).

## 3b. Sample

**What happens:** `data_sampling()` (`dsbf/utils/data_utils.py`) uses `head`, `random` or
`stratified` sampling if rows exceed `resource_limits.sample_threshold_rows` (default 1,000,000).

**Measured:** no-op for all practice datasets.

**Issues:** random sampling crashes on Polars, and `stratified` is really `head` (F-09). The
dashboard's `/sample` and `/column-data` later read the unsampled file (F-15).

## 3c–3d. Reference data and context

**What happens:** an optional reference CSV is loaded for the drift tasks. `AnalysisContext`
(`dsbf/core/context.py`) is created to hold the data, config, results, and a free-form `metadata`
dict that tasks use to pass information to each other (for example `semantic_types` from
`infer_types`).

**Measured:** negligible.

The context logs to `dsbf.core.context`, a child of `dsbf`, so its records reach the console and
`run.log` alike. (Before the logging change `run.log` held 11 lines of the titanic run while the
console had 452.)

## 3e. Load task groups and plugins

**What happens:** `load_task_group()` imports each group in `task_groups` (a module name or a
plugin directory). It no longer writes `task_metadata.json`; `dsbf tasks export` does, on request.

**Measured:** 1 ms, since the tasks were already imported in stage 0.

**Issues:** it rewrites a tracked file on every run (F-17). Plugins can't be reloaded in-process
(F-23).

## 3f. Validate config and DAG

**What happens:** `validate_config_and_graph()` (`dsbf/utils/config_validation.py`) checks for
unknown task names in `config["tasks"]`, unknown dependencies, cycles, and plugin warnings. With
`safety.strict_mode` it raises.

**Measured:** 1 ms. The shipped config validates cleanly: the shared `time_series` settings are a
top-level section, not a `tasks` entry.

**Issues:** It doesn't validate values such as depth, backend or thresholds (F-22).

## 3g. Infer data stage

**What happens:** `infer_stage()` (`dsbf/eda/stage_inference.py`) labels the data `raw`, `cleaned`,
`model_ready` or `exploratory` from null ratio, numeric share and cardinality. The label is stored
in metadata only; nothing uses it to select tasks.

**Measured:** 9–27 ms. All practice datasets came out as `exploratory`.

**Issues:** it reads the wrong config path and gives different answers for pandas and Polars
(F-08).

## 3h. Build the DAG

**What happens:** `build_graph()` (`profile_engine.py:197`):
1. Filters the registry by `task_selection` (domains, runtime estimate, excluded stages) and
   `profiling_depth`, and drops experimental tasks.
2. Adds dependency edges. Dependencies are pulled in even if the filters would have excluded them.
3. Topologically sorts the graph, instantiates each task with its `config["tasks"][name]` block, and
   wraps it in a `Task`.
4. `ExecutionGraph` recomputes levels and sorts tasks by level.

**Measured:** 2 ms. 65–67 tasks were selected at `full` depth.

**Issues:** failure-tolerant ordering relies on implicit levels (F-21).

## 3i. Run the tasks (≈99% of runtime)

**What happens:** `ExecutionGraph.run()` (`dsbf/eda/graph.py:69`) runs tasks **sequentially**. For
each task:
1. If any dependency's `Task.status != "success"`, it marks the task skipped and stores **no result**.
2. It records RSS, then `context.run_task(task)` injects `input_data` and `context` and calls `task.run()`.
3. Inside a typical task:
   - `setup_run()` converts Polars to pandas (a fresh copy for every task) and calls
     `get_columns_by_intent()`, which filters columns by the `semantic_types` that `infer_types` wrote.
   - It computes its statistics.
   - `add_guidance()` attaches per-column EDA and ML guidance blurbs.
   - Optionally, `get_recommendation_tip()` adds a tip from `recommendation_library.yaml`.
   - It builds a `TaskResult`. On exception it re-raises if a context exists.
4. `validate_task_result()` checks the result shape and logs a warning on failure.
5. The graph records the duration and "peak" memory, checks soft limits, and logs completion.
   If the task raised, it stores a `status="failed"` result.

**Measured: where the time went**

| Task | titanic (891 rows) | kitchen5k (5,250 rows) | Cause |
|---|---|---|---|
| `fuzzy_duplicate_detection` | **58.8 s** (81%) | **171.7 s** (58%) | ~395k per-pair `df.iloc` row lookups (titanic); pure-Python `difflib` on long text (kitchen) |
| `generate_univariate_plots` | 9.3 s | **84.2 s** | matplotlib text layout for long category labels; writes two **17–18 MB PNGs** |
| `compute_pairwise_associations` | 0.8 s | 10.2 s | Per-pair Python loop over up to 1,770 column pairs |
| `infer_types` | 0.1 s | 9.0 s | `_is_datetime_like` parses **entire** object columns as dates |
| `detect_stationarity` | — | 4.9 s | |
| everything else (~60 tasks) | < 3 s total | < 15 s total | |

**Measured: what broke**

| Failure | Runs | Root cause |
|---|---|---|
| 12 tasks fail with `No such file or directory: 'dsbf/static_metadata/recommendation_library.yaml'` | all | CWD-relative path in `dsbf/utils/reco_engine.py:7`; an optional tip lookup fails the whole task (F-31) |
| `data_quality_scorer` and `ml_readiness_scorer` **skipped**, so the Quality bar and ML Readiness tab are empty | kitchen5k, adversarial (ML Readiness also on titanic) | They depend on tasks above (F-31) |
| `detect_collinear_features`: `zero-size array to reduction operation maximum` | kitchen5k, adversarial | Listwise `dropna()` across all numeric columns leaves 0 rows (F-32) |
| `normality_tests`: `unsupported format string passed to NoneType.__format__` | kitchen5k | A **debug log f-string** formats `p_value=None` (F-33) |
| `summarize_modes`: `list index out of range` | kitchen5k, adversarial | An all-null column has an empty mode list (F-33) |
| `normality_qq_plots`, `transformation_preview` skipped | kitchen5k | Upstream failures above |
| 5 tasks return `status="skipped"` themselves, but are counted as **success** in `task_outcomes` | all | F-06 |

**Measured: other observations**
- `context.reliability_flags` were computed on **0 rows** for kitchen5k and adversarial, and on 714 of 891 rows for titanic. Listwise deletion again (F-32).
- On kitchen5k, DSBF's own memory metric reported 464 MB while the OS measured a **1.8 GB** peak (F-24).

## 3j. Build column↔task maps

**What happens:** `build_column_task_mappings()` calls `to_dict()` on every result and treats every
key of `data` as a column name.

**Measured:** 0.67 s (titanic), mostly re-serialising the 8.6 MB fuzzy result. The map contained
non-column keys such as `__dataset__`, `__correlation_matrix__`, `age|class|First_vs_Second`
(F-34).

## 3k. DAG image

**What happens:** if `metadata.visualize_dag` (default **true**), matplotlib draws the DAG to
`figs/dag.png`.

**Measured:** 0.9–1.1 s and a **400 KB** PNG on every run, identical whenever the task set is the
same (F-34).

## 3l. `report.json`

**What happens:** `render_user_report()` serialises every non-diagnostic result with `indent=2`,
plus `top_ml_risks`.

**Measured:** titanic **5.3 s** and an **18.5 MB** file, nearly all of it fuzzy-duplicate pairs
(8.6 MB of data, doubled by indentation). kitchen5k: 0.5 s and 2.5 MB.

**Issues:** output size is unbounded (F-35). `top_ml_risks` reads `summary["column"]`, which most
tasks don't set.

## 3m. `metadata_report.json`

**What happens:** merges `context.metadata` and `run_metadata` and adds diagnostic task results.

**Measured:** 10–40 ms.

## 3n. Persist to SQLite

**What happens:** `persist_run()` (`dsbf/storage/writer.py:246`) upserts the dataset (by name),
inserts the run, and inserts one row per task (`status, summary, data, guidance` as JSON) plus one
row per figure file.

**Measured:** 30–180 ms. With a custom output dir, `ran_at` was stored as **NULL**, which then
removed "latest run" from `/api/datasets` (F-15, verified).

**Issues:** it drops `error_metadata`, `reliability_warnings`, `recommendations` and `metadata`
(F-20). `json.dumps` writes `NaN`/`Infinity` tokens that the API later can't return (F-36).

## 3o. `dsbf_run.json` (removed)

`record_run()` used to append `run_metadata` to `./dsbf_run.json`, unbounded (F-17). It is gone: the `runs` table holds the same information.

---

## 4. Serve: FastAPI

`uvicorn dsbf.api.main:app` serves read-only endpoints from SQLite (`dsbf/api/db.py`) and figure
files from disk. It never runs profiling.

**Measured** (curl against the practice DB):

| Endpoint | titanic | kitchen5k | adversarial | Notes |
|---|---|---|---|---|
| `GET /runs/{k}` | 5 KB | 6 KB | 6 KB | |
| `GET /runs/{k}/tasks` (**loaded on run open**) | **8.2 MB** | 1.3 MB | 3.2 MB | Every task's full data in one response (F-35) |
| `GET /runs/{k}/dq-status` | ok | `available:false` | `available:false` | Scorer was skipped; the UI gets no reason (F-20) |
| `GET /runs/{k}/ml-readiness` | `available:false` | `available:false` | `available:false` | Same |
| `GET /runs/{k}/associations` | 15 KB | 124 KB | **HTTP 500** | `NaN` in stored data isn't JSON-compliant (F-36) |
| `GET /runs/{k}/sample?n=10` | 1.3 s first call, then 7 ms | 80 ms | **HTTP 500** | Built-in data re-downloaded on the first call; `inf` value isn't JSON-compliant (F-36) |
| `GET /runs/{k}/associations/price\|usd` | | | 200, **empty list** | Pipe in column name mis-splits the pair key (F-10) |
| `GET /runs/{k}/column-data?cols=does_not_exist` | | **200** `{"error": ...}` | | Errors returned as success (F-25) |

## 5. Dashboard: Vue

`RunDetailView.vue` loads `getRun`, `getRunTasks` and `getRunFigures` in parallel on open, then
each tab reads `tasks[<task_name>].data` directly. Charts in the in-progress refactor are built from
that task data with ECharts; older cards load figure files through `/api/figures/{id}/file`.

**Issues:** the full task payload (8 MB for titanic) is downloaded before any tab renders. The JS
bundle is 5.5 MB (F-29). A run whose scorers were skipped shows empty Quality and ML Readiness
views with no explanation.

---

## Scaling (kitchen50k)

Same generator and config as kitchen5k, with 10× the rows (52,500 × 60, a 53 MB CSV).

| | kitchen5k | kitchen50k | Growth |
|---|---|---|---|
| Wall time | 298 s | **707 s (≈12 min)** | 2.4× |
| Peak RSS (OS, `/usr/bin/time`) | 1.8 GB | **2.1 GB** (≈40× the file size) | |
| `_load_data` | 0.08 s | 0.88 s | 10× |
| Tasks failed | 12 | 12 (same set) | |

Per-task scaling (10× rows):

| Task | 5k | 50k | Growth | Why |
|---|---|---|---|---|
| `detect_stationarity` | 4.9 s | **186 s** | **38×** | `adfuller(autolag="AIC")` with no `maxlag` cap; the default max lag grows with n, so cost is super-linear (F-39) |
| `fuzzy_duplicate_detection` | 172 s | 168 s | 1× | Capped at 5,000 sampled rows and 50,000 candidate pairs, so cost is flat but large (F-35) |
| `generate_univariate_plots` | 84 s | 122 s | 1.5× | Bar-chart PNGs for long-string columns grew to **31 MB each** (F-38) |
| `infer_types` | 9 s | 92 s | 10× | Full-column datetime parsing (F-37) |
| `compute_pairwise_associations` | 10 s | 72 s | 7× | Per-pair Python loop |
| `missingness_mechanism_analysis` | 2.7 s | 18 s | 7× | |
| everything else | | < 60 s total | | |

The five tasks above account for about 90% of runtime. Fixing F-35, F-37, F-38 and F-39 (or
retiring backend plotting, N-A2) would bring this run from about 12 minutes to roughly 1–2 minutes.
That's an estimate from the profile, not a measurement.

## Reproducing the measurements

The practice-run scripts were promoted to the benchmark in [`tests/perf/`](../tests/perf/README.md),
which runs the same datasets in a fresh interpreter and records the stage, task, memory and
dashboard-payload numbers described above:

```bash
uv run pytest -m perf --perf-no-compare
```

It writes `perf-results/perf.json`. The committed `tests/perf/baseline.json` holds the current
numbers from CI. The tables in this document are the 2026-10-15 measurements: they were taken
under `cProfile` and before the crash fixes, so they run slower than a run today and are kept as
history, not as a reference.
