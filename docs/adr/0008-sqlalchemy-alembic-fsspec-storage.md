# ADR 0008: SQLAlchemy Core, Alembic and fsspec for storage

**Status:** Accepted · **Date:** 2026-10-16 · **Design:** [`REFACTOR_PLAN.md`](../REFACTOR_PLAN.md) §8.2

## Context

The current store uses raw `sqlite3` with ad-hoc `ALTER TABLE` migrations, drops most of a task
result, and has no run management (F-20, F-25). Requirement R4 asks for local first and team later:
storage, auth and job execution must sit behind interfaces, so the team version is new
implementations, not a rewrite.

## Decision

- A `RunRepository` interface (save, get, list, query findings, history by dataset, compare,
  delete, gc) is implemented with **SQLAlchemy Core**, with **Alembic** migrations.
  - SQLite in Phase 6, Postgres in Phase 8.
  - Tables: `datasets`, `runs`, `check_results`, `findings`, `profiles` and `artifacts`.
- An `ArtifactStore` interface is backed by **fsspec**: local `DSBF_HOME` in Phase 6, S3, GCS and
  Azure in Phase 8.
- Each run stores a Parquet snapshot of the data as profiled (sampled where applicable, size-capped
  and configurable), so dashboard views show what was analysed.
- Store, service and dashboard live in the `server` extra ([ADR 0009](0009-single-distribution-with-extras.md)). The core imports none of them ([ADR 0001](0001-library-first-layering.md)).
- No Postgres or auth code is written before Phase 8. Phase 6 builds the interfaces and SQLite only.

## Alternatives considered

- **Raw `sqlite3`** (today). No migration tooling, and SQLite-specific.
- **DuckDB as the store.** Attractive for analytical queries over findings, but a weaker fit for
  concurrent writers and the later move to Postgres.

## Consequences

- Schema changes become Alembic migrations rather than runtime `ALTER TABLE`.
- Existing local databases are not migrated: before 1.0 you re-profile (REFACTOR_PLAN assumption A2).
- Using SQLAlchemy Core (not the ORM) keeps queries explicit and Postgres a dialect change.
- `dsbf runs list | show | diff | delete | gc` becomes possible.
