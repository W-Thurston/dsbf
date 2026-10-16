# ADR 0001: Library-first layering

**Status:** Accepted · **Date:** 2026-10-16 · **Design:** [`REFACTOR_PLAN.md`](../REFACTOR_PLAN.md) §2, §3

## Context

DSBF has to serve three modes (REFACTOR_PLAN §1): embedded in someone else's pipeline, standalone
with its own store and dashboard, and anything between. Today the engine is coupled to its
surroundings. A run writes `dsbf/outputs/`, `run.log` and `dsbf_run.json`, opens a SQLite database,
installs a global logger class at import time, and reads package data relative to the working
directory. Embedding it in a notebook or an orchestrator task means inheriting all of that, and
testing it needed a hermetic-test workaround (F-02, F-16, F-17, F-19, F-31).

## Decision

The core is a pure function: data plus configuration in, a `Run` out. Layers L0–L4 (contracts, data,
profile, checks, engine) and the in-memory result do no file I/O, database access, global logging
configuration or network calls. Everything else is an optional layer that consumes core output:

- L5 policy and L6 sinks (files, Parquet, HTML) are optional consumers of a `Run`.
- L7 store, L8 service and L9 dashboard live behind the `server` extra ([ADR 0009](0009-single-distribution-with-extras.md)).
- The CLI and other integrations are thin wrappers over the library API.

Runtime paths live under a configurable `DSBF_HOME`, and package data is read through
`importlib.resources`.

## Alternatives considered

- **Keep the engine coupled to storage and files** (today). Cheapest in the short term, but it
  makes mode 1 impossible without side effects and keeps the hermetic-testing problem.

## Consequences

- Mode 1 users get a library with no required database, files or server dependencies.
- Each layer can be swapped (mode 3): your checks with DSBF's store, DSBF's checks with your dashboard.
- Logging follows library conventions: a `dsbf` logger hierarchy that propagates and changes no global state.
- The legacy engine, database and dashboard keep working unchanged until the Phase 6 cutover, so
  the two stacks coexist for several phases (REFACTOR_PLAN §14).
- Tests need no isolation fixtures for the new core.
