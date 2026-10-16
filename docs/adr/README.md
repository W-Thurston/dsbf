# Architecture decision records

Each record states the context, the decision, the alternatives considered and the consequences.
The design they belong to is [`docs/REFACTOR_PLAN.md`](../REFACTOR_PLAN.md) (§13 summarises them).

**Statuses:** `Proposed`, `Accepted`, `Superseded by ADR NNNN`. An accepted record isn't edited to
change the decision. Write a new record that supersedes it, and mark the old one. Corrections of
fact and new links are fine.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-library-first-layering.md) | Library-first pure core; everything else optional | Accepted |
| [0002](0002-findings-as-primary-output.md) | `Finding` with typed evidence is the primary output | Accepted |
| [0003](0003-polars-arrow-data-representation.md) | Polars/Arrow internally, NumPy at the statistics boundary | Accepted |
| [0004](0004-ibis-warehouse-pushdown.md) | Ibis for warehouse and database pushdown | Accepted |
| [0005](0005-pydantic-contracts.md) | pydantic v2 contracts, JSON Schema, generated TypeScript | Accepted |
| [0006](0006-report-only-failure-semantics.md) | Report-only failure semantics, opt-in policies | Accepted |
| [0007](0007-entry-point-plugins.md) | Entry-point plugins with the same contract as core | Accepted |
| [0008](0008-sqlalchemy-alembic-fsspec-storage.md) | SQLAlchemy Core, Alembic and fsspec for storage | Accepted |
| [0009](0009-single-distribution-with-extras.md) | One distribution with extras and a `src/` layout | Accepted |
| [0010](0010-check-naming.md) | Rename "task" to "check" outright | Accepted |
| [0011](0011-toolchain.md) | uv, pytest tiers, pyrefly, pnpm | Accepted |
