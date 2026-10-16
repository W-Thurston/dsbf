# ADR 0004: Ibis for warehouse and database pushdown

**Status:** Accepted · **Date:** 2026-10-16 · **Design:** [`REFACTOR_PLAN.md`](../REFACTOR_PLAN.md) §5.1

## Context

The embedded mode (REFACTOR_PLAN §1, R3) includes data that lives in databases and warehouses
(Postgres, DuckDB, Snowflake, BigQuery and others). Pulling whole tables into memory doesn't scale
and is often not allowed. The profiling pass is mostly aggregations (counts, min and max, moments,
quantiles, histograms, top-k), which a database can compute itself.

## Decision

Source tier T3 uses **Ibis** behind the `Source` and profile interface. Profile aggregations are
pushed down as SQL. Analysis checks pull a bounded sample into the Polars-based `Dataset`. Checks
never see a connection ([ADR 0003](0003-polars-arrow-data-representation.md)): adding a tier means adding a source adapter.

- Ibis ships only in the `sql` extra, with backend extras on top ([ADR 0009](0009-single-distribution-with-extras.md)).
- Nothing is built before Phase 7. Until then this decision constrains interfaces only:
  `Source` and the profile interface must not assume an in-memory frame.
- A check that needs a capability the source lacks reports `unavailable` with a reason.

## Alternatives considered

- **Hand-written SQL per dialect.** Full control, but a maintenance load that grows with each backend.
- **DuckDB-only federation.** Simple, but it reaches warehouses only through DuckDB scanners, and
  can't push aggregation into them as reliably.

## Consequences

- Phase 2's profile interface is written so the same statistics can be expressed as an Ibis
  aggregation. Phase 7 validates this and may amend this ADR.
- Ibis API churn is isolated behind the adapter.
- Pushdown results are approximate for some statistics (quantiles, distinct counts). The profile
  records the method used, as the in-memory path does.
