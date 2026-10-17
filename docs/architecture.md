# DSBF architecture today

This describes the current (pre-refactor) code: how the pieces fit together and where to start reading. The target design is in [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md), a measured walk through one run is in [`WALKTHROUGH.md`](WALKTHROUGH.md), and the backlog is in [`../ROADMAP.md`](../ROADMAP.md).

## 1. What this project is

**DSBF (Data Scientist's Best Friend)** is a **tabular EDA / data-profiling engine**. It runs a **DAG of registered tasks** over a dataset (file or built-in), writes **JSON reports and plots** to a timestamped output directory, and **persists run metadata and figure paths to SQLite** so a **Vue dashboard** can browse history and drill into results.

**Python:** `>= 3.12` (`pyproject.toml`).

---

## 2. Repository layout

| Area | Role |
|------|------|
| `src/dsbf/eda/` | Profiling engine, task graph, stage inference, task registry and loader |
| `src/dsbf/eda/tasks/` | Individual analysis tasks (time series, ML readiness, associations, quality, and more) |
| `src/dsbf/core/` | `BaseEngine`, `BaseTask`, `AnalysisContext` |
| `src/dsbf/config/` | Default config (`default_config.yaml`), constants, Plotly and dashboard theme helpers |
| `src/dsbf/utils/` | Data load, sampling, report rendering, task filtering, logging, recommendations |
| `src/dsbf/storage/` | SQLite schema (`schema.py`) and run persistence (`writer.py`) |
| `src/dsbf/api/` | **FastAPI** app (`main.py`) and read-only DB helpers (`db.py`) |
| `src/dsbf/interfaces/` | Typer **CLI** (`cli.py`) and the Python facade `EDA` (`api.py`) |
| `src/dsbf/custom_plugins/` | Example location for user task directories (see `task_groups` in the config) |
| `frontend/` | The Vue 3 + Vite dashboard app, outside the Python package |
| `tests/` | Pytest suite, split into tiers by marker (`unit`, `integration`, `slow`, `network`) |
| `dsbf/outputs/` | Generated run folders (gitignored) |

`run_profile.py` and `migrate_runs.py` at the repository root are standalone scripts that later phases replace with CLI subcommands.

---

## 3. Core execution flow

### `ProfileEngine` (`src/dsbf/eda/profile_engine.py`)

1. Load config (caller-provided or default).
2. Load data via `_load_data()`:
   - If `metadata.dataset_path` is set, it must exist. Supported: `.csv`, `.parquet`/`.pq`, `.xlsx`/`.xls`. Errors are explicit (including a WSL path hint).
   - Otherwise a built-in dataset is loaded through `utils.data_loader.load_dataset`, using `dataset_name`, `dataset_source` and `engine.backend`.
3. Optional reference CSV from `engine.reference_dataset_path`.
4. Build `AnalysisContext` (data, config, `output_dir`, run metadata, reference data).
5. Load task groups from config (`task_groups`, default `["core"]`); collect plugin warnings.
6. Validate config + DAG (`utils.config_validation`); **strict mode** (`safety.strict_mode`) can abort on errors.
7. Infer the data stage (`eda/stage_inference`) and store it on the context and metadata.
8. Build `ExecutionGraph` from the registry specs, filtered by depth, domain, runtime and excluded stages.
9. Run tasks; collect `TaskResult` objects in `engine.results` and `context.results`.
10. Build **column ↔ task** mappings (`utils.task_and_column_mapping_utils`) into metadata.
11. Optional DAG PNG if `metadata.visualize_dag`.
12. Write `report.json` (unless `metadata.disable_report`) and `metadata_report.json`.
13. **`storage.writer.persist_run(self)`**: SQLite upsert (failures are logged, not fatal).
14. `record_run()` and completion log.

### Tasks (`src/dsbf/eda/tasks/*.py`, `src/dsbf/core/base_task.py`)

- All tasks subclass **`BaseTask`**; outputs are **`TaskResult`** (`src/dsbf/eda/task_result.py`).
- Registration and metadata live in **`src/dsbf/eda/task_registry.py`** (domains, depth, dependencies, semantic types, and more).
- The codebase has grown beyond classic EDA: examples include the **time-series** tasks, **`compute_pairwise_associations`**, **`ml_readiness_scorer`** and **`data_quality_scorer`**. The API and frontend rely on some of these for the Relationships, Quality and ML Readiness views.

---

## 4. Artifacts each run produces

Under **`dsbf/outputs/<run_key>/`**. The folder name is a `%Y%m%d_%H%M%S` timestamp and is the **`run_key`** in the database.

| Artifact | Purpose |
|----------|---------|
| `report.json` | Task-keyed results (the API and dashboard mirror this shape) |
| `metadata_report.json` | Run diagnostics, timings, mappings |
| `run.log` | Text log |
| `figs/` | Plot files, PNG and Plotly JSON (paths are also indexed in the SQLite `figures` table) |
| `figs/dag.png` | Task graph image, when `metadata.visualize_dag` is on and Graphviz is installed |

---

## 5. SQLite storage

- **Schema and connection:** `src/dsbf/storage/schema.py`.
  - The database path is `DSBF_DB_PATH` if set, else the `db_path` argument, else **`<repo_root>/dsbf/dsbf.db`**.
- **Writes:** `src/dsbf/storage/writer.py`, **`persist_run(engine)`** after each successful profile:
  - datasets, runs, per-task rows and the figure index;
  - `dataset_path` is handled apart from built-in datasets (CLI `quickstart` clears the path so the database doesn't keep a stale file path).

The **FastAPI layer does not re-run profiling**. It **reads the database** and **serves figure files from disk** using the paths stored there.

---

## 6. HTTP API (FastAPI)

**App:** `src/dsbf/api/main.py` (`app`). The top-of-file docstring lists the routes.

**Run locally** (from the repository root):

```bash
uv run uvicorn dsbf.api.main:app --reload
```

**Environment:**

- **`DSBF_DB_PATH`**: optional SQLite file path (same semantics as storage).
- **`DSBF_CORS_ORIGINS`**: comma-separated origins; the default is `http://localhost:5173` (Vite). Only `GET` is allowed.

**Routes:**

- **`GET /health`**
- **Datasets:** `/api/datasets`, `/api/datasets/{name}`, `/api/datasets/{name}/runs`
- **Runs:** `/api/runs/{run_key}`, `/tasks`, `/tasks/{task_name}`, `/figures`, `/sample`
- **Compare:** `/api/runs/compare?run_keys=...&task_name=...`
- **Data health:** `/api/runs/{run_key}/dq-status` (from `data_quality_scorer`)
- **ML readiness:** `/api/runs/{run_key}/ml-readiness` (from `ml_readiness_scorer`)
- **Relationships:** `/associations`, `/associations/{column}`, `/correlations/{column}`, `/column-data`
- **Figures:** `/api/figures/{id}/file` is a `FileResponse`; figure paths in the database are relative to the repository root.

**DB access:** `src/dsbf/api/db.py` holds pure read helpers on top of `get_connection`, returning plain dicts and lists.

---

## 7. Dashboard

**Location:** `frontend/`

- **Vue 3 + Vite + Vue Router + Axios**, with **Plotly** (`plotly.js-dist-min`) and **ECharts** for charts (`package.json`).
- **API client:** `frontend/src/api.js`, with `baseURL: '/api'`.
- **Dev proxy:** `vite.config.js` proxies **`/api` → `http://localhost:8000`**, so you run **FastAPI on 8000** and **`pnpm dev`** on 5173 without CORS friction.
- **Views and tabs:** dataset and run lists, run detail, and the tabs **Overview, Distributions, Relationships, Quality, ML Readiness and Time Series** (`frontend/src/views/` and `components/`).

**Operational recipe:**

1. Run a profile (`dsbf profile ...` or `dsbf run --config ...`) so SQLite is populated.
2. Start the API: `uv run uvicorn dsbf.api.main:app --reload`
3. Start the UI: `cd frontend && pnpm install && pnpm dev`

---

## 8. CLI

**Entry:** `dsbf` → `dsbf.interfaces.cli:app` (`pyproject.toml`).

| Command | Role |
|---------|------|
| `dsbf run -c config.yaml [--strict] [--visualize-dag] [--no-report]` | Full YAML-driven run |
| `dsbf profile DATA.csv [-d depth] [-n name]` | Default config. `--name` sets `metadata.dataset_name` (else the file's stem) for database and dashboard labels |
| `dsbf quickstart [dataset]` | Built-in dataset; clears `dataset_path` |
| `dsbf version` | Version string |

There is no CLI command that renders the dashboard; the dashboard is the Vue app served by Vite, against the FastAPI backend.

---

## 9. Python programmatic use

**`dsbf.interfaces.api.EDA`** accepts a path or an in-memory DataFrame (pandas or Polars) plus an optional config (a YAML path or a dict). It wires up `ProfileEngine` and returns **`get_all_results()`** from `run()`.

```python
from dsbf.interfaces.api import EDA

results = EDA("data.csv").run()
```

---

## 10. Dependencies and tooling

- **Install:** `uv sync` (the lockfile is `uv.lock`; dev tools are in `[dependency-groups] dev`).
- **Tests:** `uv run pytest`. The fast gate is `uv run pytest -m "unit and not slow"`; every test carries exactly one of the `unit` and `integration` markers.
- **Format and lint:** Ruff (`uv run ruff check . && uv run ruff format .`) and pre-commit. CI pins Ruff to the version in `uv.lock`.

---

## 11. Quick start for a new maintainer

```bash
uv sync
uv run dsbf quickstart titanic
uv run uvicorn dsbf.api.main:app --reload
# other terminal:
cd frontend && pnpm install && pnpm dev
# Browser: the Vite URL (e.g. http://localhost:5173), API proxied to :8000
uv run pytest
```

This is the current stack: **profile → SQLite + files → FastAPI → Vue dashboard**.
