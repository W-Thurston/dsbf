# ADR 0007: Entry-point plugins with the same contract as core

**Status:** Accepted · **Date:** 2026-10-16 · **Design:** [`REFACTOR_PLAN.md`](../REFACTOR_PLAN.md) §6.5

## Context

The legacy registry registers core tasks as an import side effect, and the `core` group imports an
empty module (F-23). Loading a plugin directory twice re-executes it and logs a traceback.
Plugin modules are registered in `sys.modules` by file stem, so two plugin directories with a
`utils.py` overwrite each other, and plugin warnings accumulate in a process-wide global.

## Decision

- Third-party checks and packs are discovered through the `dsbf.checks` and `dsbf.packs`
  **entry-point groups** (for example `pip install dsbf-healthcare`).
- **Local directories** remain supported (`plugins: [./my_checks/]`). They are imported under unique
  module names and are safe to reload.
- Core checks and plugins use the **same contract**: declared parameter models, version, cost,
  extras, requirements, isolation and caching. Core has no privileged path.
- Registration is explicit, with no import side effects.
- A plugin that fails to load produces a visible note in the run, not a traceback in a log.

## Alternatives considered

- **Import side effects and global registries** (today). Order-dependent, not reload-safe, and
  invisible to tooling.

## Consequences

- The registry is built from declared metadata, so `dsbf plan` and the generated config reference
  can list checks without executing them.
- Packaging a pack is the standard Python entry-point route, with no DSBF-specific mechanism.
- The plugin API is a compatibility surface once 1.0 ships, so the check interface is designed
  carefully in Phase 3 (the decorator and class forms are weighed there).
- Loading arbitrary local directories executes code. That is the same trust model as any Python import, and it is documented.
