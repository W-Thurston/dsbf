# DSBF architecture today

This describes the code as it stands after the legacy engine was removed: how the pieces fit together and where to start reading. The target design is in [`REFACTOR_PLAN.md`](REFACTOR_PLAN.md), the removed legacy run is described in [`WALKTHROUGH.md`](WALKTHROUGH.md), and the backlog is in [`../ROADMAP.md`](../ROADMAP.md).

## 1. What this project is

**DSBF (Data Scientist's Best Friend)** is a **tabular EDA / data-profiling engine**. `dsbf.profile()` loads a table, profiles it once, plans which **checks** apply, runs them under limits and returns one `Run`: a validated model of the data's identity, the profile, every check's status, findings and output, and the notes explaining anything that did not complete. `dsbf profile` writes that `Run` to `run.json`.

**Python:** `>= 3.12` (`pyproject.toml`).

---

## 2. Repository layout

| Area | Role |
|------|------|
| `src/dsbf/data/` | `load()` (CSV, TSV, Parquet, Excel, JSON, NDJSON, pandas, Polars, Arrow), `Dataset`, sampling and the data fingerprint |
| `src/dsbf/profiling/` | `profile_dataset()`: the one pass over the data that checks read (counts, numeric stats, value counts, types, histograms, temporal profile) and stage inference |
| `src/dsbf/engine/` | The check contract (`@check`, `CheckSpec`), `Registry`, planner (`make_plan`), executor (`execute`), process workers, limits, cache, reference data, recommendation tips |
| `src/dsbf/checks/` | The `core` pack: one module per check (47 of them) |
| `src/dsbf/model/` | The `Run` and every model inside it: `Finding`, `Evidence`, `CheckResult`, `Profile`, `Policy` and the vocabularies. Published as `schema/run.schema.json` |
| `src/dsbf/config/` | `ProfileConfig`, the sections it holds and the layered settings loader (file, `DSBF__` environment variables, arguments) |
| `src/dsbf/library.py` | `dsbf.profile()` |
| `src/dsbf/interfaces/` | The Typer CLI (`cli.py`) and the schema export (`schema_export.py`) |
| `src/dsbf/utils/` | Console logging and the package version |
| `src/dsbf/static_metadata/` | `recommendation_library.yaml`, the tips attached to findings |
| `src/dsbf/paths.py` | The DSBF home (`$DSBF_HOME` or the per-user data directory) and its `outputs/` |
| `frontend/` | The Vue 3 + Vite dashboard source. It has no API to read until Phase 6 of the roadmap |
| `tests/` | Pytest suite, split into tiers by marker (`unit`, `integration`, `golden`, `perf`, `calibration`; `slow` and `network` modify) |

---

## 3. Core execution flow

### `dsbf.profile()` (`src/dsbf/library.py`)

1. Build the settings: the settings file you name, then `DSBF__` environment variables, then the arguments (`config.load_config`). Every problem is reported at once as a `ConfigError`.
2. `data.load()` reads the source into a `Dataset`; a `sample` is drawn if asked, and its provenance recorded.
3. `profile_dataset()` makes the `DatasetProfile`.
4. The `core` pack goes into a `Registry`, and `make_plan` selects checks by pack, depth, id and the columns each would read, adds what they require, orders them and validates the parameters (`PlanError` on a mistake).
5. `execute` runs the plan: checks in dependency order on threads or worker processes, each under the check and run limits; a result cache keyed on the data fingerprint, the check version, its parameters and its upstream results is used when asked for.
6. After a check returns, tips from the recommendation library are attached to its findings as actions.
7. The `Run` comes back in memory. Nothing is written unless a `cache` is passed.

### Checks (`src/dsbf/checks/*.py`, `src/dsbf/engine/check.py`)

- A check is a function declared with `@check`: id, version, kind, the semantic types it applies to, what it requires from the profile and from other checks, a parameter model, an output model, cost and packs. The module docstring of `src/dsbf/checks/__init__.py` lists the conventions.
- A check reports through `ctx.finding(...)` and returns its output. A check that raises ends with status `error` and a note; it never stops the run.
- `quality_dimensions` and `ml_readiness` are report-only synthesis checks that read the other checks' findings.

---

## 4. Artifacts

`dsbf profile` writes `run.json` under `--out`, or `<DSBF home>/outputs/<run_id>/`. `dsbf.profile(cache=True)` keeps check results under `<DSBF home>/cache/`. Nothing else is written.

---

## 5. Dashboard

**Location:** `frontend/`. The source is the one the removed FastAPI backend served: Vue 3 + Vite + Vue Router + Axios, with Plotly (`plotly.js-dist-min`) and ECharts. It does not run until Phase 6 rebuilds the store, the API and the dashboard shell on the `Run`. `frontend/src/generated/run.d.ts` holds the TypeScript types of the `Run`, generated from the models.

---

## 6. CLI

**Entry:** `dsbf` → `dsbf.interfaces.cli:app` (`pyproject.toml`).

| Command | Role |
|---------|------|
| `dsbf profile DATA [-o DIR] [-p pack] [-d depth] [--target COL] [--time-index COL] [--plugin DIR] [--param check.key=value]` | Runs `dsbf.profile()` and writes `run.json` to `DIR` (default `$DSBF_HOME/outputs/<run_id>/`); prints `Run.render_health()`. Exit 2 when setup fails |
| `dsbf plan DATA` | Shows the checks a run would plan |
| `dsbf schema export` / `dsbf schema check` | Writes and guards `schema/run.schema.json` and the TypeScript types |
| `dsbf version` | Version string |

---

## 7. Python programmatic use

```python
import dsbf

run = dsbf.profile("data.csv", depth="quick", target="survived")
print(run.render_health())
```

See the README for every argument.

---

## 8. Dependencies and tooling

- **Install:** `uv sync` (the lockfile is `uv.lock`; dev tools are in `[dependency-groups] dev`).
- **Tests:** `uv run pytest`. The fast gate is `uv run pytest -m "unit and not slow"`; every test carries exactly one tier marker.
- **Format and lint:** Ruff (`uv run ruff check . && uv run ruff format .`), Pyrefly with a committed baseline, and pre-commit. CI pins Ruff to the version in `uv.lock`.

---

## 9. Quick start for a new maintainer

```bash
uv sync
uv run dsbf profile tests/golden/data/titanic.csv
uv run dsbf plan tests/golden/data/titanic.csv --depth deep
uv run pytest -m "unit and not slow"
```
