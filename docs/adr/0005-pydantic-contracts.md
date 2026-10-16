# ADR 0005: Pydantic v2 contracts, JSON Schema and generated TypeScript

**Status:** Accepted · **Date:** 2026-10-16 · **Design:** [`REFACTOR_PLAN.md`](../REFACTOR_PLAN.md) §4, §4.3

## Context

Mode 1 consumers load DSBF results into their own warehouses, dashboards and orchestrators, so they
need a schema that doesn't depend on DSBF internals. Today the result shape is implicit in
dictionaries, and the dashboard's TypeScript expectations are hand-maintained (F-25, F-29, F-36).
Configuration has the same problem: parameters are untyped, unknown keys are silently ignored, and
the default config failed its own validation (F-04, F-07, F-22).

## Decision

- Results, configuration, profiles, policies and the plugin interface are **pydantic v2** models.
- The models are the single source. `dsbf schema export` writes **JSON Schema**, and **TypeScript
  types** for the dashboard are generated from it.
- Every serialised object carries a semver `schema_version`. Additive changes are minor, breaking
  changes are major, and compatibility guarantees start with the published 1.x schema.
- The exported schema is checked in, and CI fails on an unannounced breaking diff.
- Each check has its own `version`. Changing its statistics or output bumps it.
- Check parameters are declared models. Unknown or ignored keys are validation errors.
- `schema export` is the only way to regenerate the checked-in schema and types.

## Alternatives considered

- **Dataclasses plus hand-written TypeScript.** No new dependency, but validation, schema export
  and TypeScript generation would each be hand-built and would drift.

## Consequences

- pydantic becomes a core dependency, and the contracts package (L0) otherwise has no heavy dependencies.
- Nothing may reach a stored model unvalidated: non-finite floats become `None` with a flag, as the
  evidence model requires.
- Before 1.0, schema changes are free (no compatibility shims, per CLAUDE.md practice 11).
- The breaking-change gate and type generation are built in Phase 1.
