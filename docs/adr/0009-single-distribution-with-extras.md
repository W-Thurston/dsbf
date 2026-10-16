# ADR 0009: One distribution with extras and a `src/` layout

**Status:** Accepted · **Date:** 2026-10-16 · **Design:** [`REFACTOR_PLAN.md`](../REFACTOR_PLAN.md) §3.1

## Context

Today every dependency is a core dependency: fastapi, uvicorn, matplotlib, plotly, seaborn,
statsmodels, scikit-learn, networkx, graphviz, missingno and more. An embedded user pays for the
whole server and plotting stack, and `import dsbf` takes about 2.0 s (REFACTOR_PLAN §11). The
package also lives at the repository root, so tests can import the working tree instead of the
installed package, and CWD-relative package-data reads went unnoticed (F-31).

## Decision

DSBF stays **one distribution** with extras, in a **`src/dsbf/`** layout.

| Install | Contents |
|---|---|
| `dsbf` | Layers L0–L6, the core check pack (rules and basic statistics), CLI, JSON, Parquet and HTML sinks |
| `dsbf[ml]` | Isolation forest, mutual information, ML-readiness extras (scikit-learn) |
| `dsbf[timeseries]` | ACF/PACF, stationarity, decomposition (statsmodels) |
| `dsbf[text]` | Fuzzy duplicates, string anomaly scoring (rapidfuzz) |
| `dsbf[pandas]` | pandas input and output convenience |
| `dsbf[sql]` | Database and warehouse sources (Ibis) |
| `dsbf[server]` | Store, service, bundled dashboard (FastAPI, SQLAlchemy, Alembic) |
| `dsbf[all]` | Everything |

- A selected check whose extra isn't installed is reported as **`unavailable`** with a hint such as
  "install `dsbf[timeseries]`". It never errors.
- Package data is read through `importlib.resources`, and runtime data lives under `DSBF_HOME`.
- Heavy extras are imported lazily.

## Alternatives considered

- **Several packages** (`dsbf-core`, `dsbf-server`). Cleaner dependency boundaries, but separate
  versioning, releases and CI for a solo maintainer, for a benefit extras already give.

## Consequences

- The extras are declared in Phase 1, with the `src/` move as a separate unit. The base install
  shrinks only as legacy tasks migrate (Phase 4), since they still need the heavy libraries, and
  dependencies nothing imports are removed then.
- Each extra needs a test that the core runs without it and reports `unavailable`.
- A wheel smoke test (build, install in a clean environment, run from a temporary directory) guards the layout.
- Extras are the unit of the install matrix in CI.
