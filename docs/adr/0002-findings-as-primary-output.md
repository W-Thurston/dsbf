# ADR 0002: Findings with typed evidence as the primary output

**Status:** Accepted · **Date:** 2026-10-16 · **Design:** [`REFACTOR_PLAN.md`](../REFACTOR_PLAN.md) §4

## Context

Each legacy task returns a free-form `data` dict plus guidance text. Consumers (the dashboard, the
quality scorer, exports) must know every task's private shape. Subjects are encoded as strings such
as `col1|col2`, which breaks when a column name contains `|` (F-10). Empty results have no defined
shape (F-03), skipped tasks produce no result at all (F-20), non-finite floats reach storage and
make API endpoints return HTTP 500 (F-36), and sampling provenance is not attached to the numbers it
affects (F-09, F-15).

## Decision

A `Finding` is the unit of output: one statement about the data with a structured `Subject`, a
`dimension`, a `severity`, a self-contained message, `Evidence` and optional guidance.

- `Evidence` carries the numbers behind the finding: finite metrics, `n_used`, `n_total`, method,
  assumption checks, an interval where one is defined, and sampling provenance. A finding can't be
  created without its sample size, method and provenance (R5).
- `Finding.id` is a stable hash of check, subject and code, so findings can be compared across runs.
- Operational outcomes are separate: a `CheckResult` has a status and `Note`s, never a finding.
- Bulky outputs (fuzzy-duplicate pairs, contingency tables, pairwise matrices) are written as
  artifacts and referenced by `details_ref`. Findings stay small.
- Dashboards, policies, exports and agents all read findings.

## Alternatives considered

- **Free-form per-task `data` dicts** (today). No schema, so no stable contract for consumers, and
  honesty about samples and uncertainty depends on each author remembering it.

## Consequences

- Every legacy task's output must be mapped to findings. The legacy adapter does this from Phase 3,
  and each task is then migrated properly in Phase 4.
- Golden runs snapshot normalised findings, so changes in behaviour show up as reviewable diffs.
- The model definitions are the contract surface for [ADR 0005](0005-pydantic-contracts.md).
- Check authors do more up front (a code, a subject, evidence). The `CheckContext` helpers keep that cheap.
