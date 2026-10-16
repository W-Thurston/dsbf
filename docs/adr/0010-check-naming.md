# ADR 0010: Rename "task" to "check" outright

**Status:** Accepted · **Date:** 2026-10-16 · **Design:** [`REFACTOR_PLAN.md`](../REFACTOR_PLAN.md) §4.1

## Context

The unit of analysis is called a "task" in the legacy engine. "Check" is the word pipeline and
data-quality users already use, and it is clearer next to "finding", "pack" and "plan" (the
vocabulary in REFACTOR_PLAN §4.1). DSBF has no published release and no external users, so a rename
breaks nobody (REFACTOR_PLAN assumption A2, CLAUDE.md practice 11).

## Decision

- The public API, configuration, CLI, schemas and documentation say **check**, with no alias for "task".
- The legacy adapter (`LegacyTaskCheck`, which wraps a `BaseTask`) is the only place the old name
  survives, and it is deleted at the end of Phase 4.
- Legacy `tasks` modules keep their names until each is migrated, and are renamed as they move.
  There is no separate mass-rename PR.
- Identifiers in code, comments and tests describe domain behaviour and carry no unit or finding IDs.

## Alternatives considered

- **Keep "task".** No churn, but it keeps a workflow-engine word for a data-quality concept, and
  the new public API would be inconsistent with its own vocabulary.
- **Rename with a deprecation alias.** Pointless without external users.

## Consequences

- Until Phase 4 ends, the repository uses both words: "check" in new code, "task" in legacy code.
  Docs say which they mean.
- The generated config reference and the CLI use "check" from the start.
- Golden and performance baselines key on legacy task names, and the migration maps old ids to new ones.
