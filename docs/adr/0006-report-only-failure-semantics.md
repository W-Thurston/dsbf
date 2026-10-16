# ADR 0006: Report-only failure semantics with opt-in policies

**Status:** Accepted · **Date:** 2026-10-16 · **Design:** [`REFACTOR_PLAN.md`](../REFACTOR_PLAN.md) §6.4, §7

## Context

Requirement R2: a failing check must never stop the run or the pipeline, and every failure, skip or
unavailable check must be explained in the results, so nobody has to read logs. Today each task
repeats its own try/except (F-06), a skipped task produces no result (F-20), a failed task stops
dependents from running (F-21), and the database drops the error metadata that would explain it.

## Decision

- A run **always completes** and **always returns a `Run`**. Only unrecoverable setup errors (an
  unreadable source, invalid configuration) stop it, and the CLI still writes a `Run` that contains the error.
- The **executor owns isolation**. An exception becomes `status="error"` with a `Note` (error type,
  message, hint), and the traceback goes to an artifact. Hard dependents become `skipped` and name
  the blocker. Soft (`after`) dependents run. Nothing propagates.
- Statuses are explicit: `ok`, `not_applicable`, `skipped`, `unavailable`, `error`, `timeout`.
  Every non-`ok` status carries a reason.
- `Run.health` summarises outcomes by status, and the CLI prints it at the end of every run.
- The **exit code is 0 by default**, even when checks error.
- Users may opt in to **policies**: declarative YAML rules over findings and health (no `eval`) with
  actions `annotate`, `warn`, `fail` and `notify`. `fail` affects only the exit code and the recorded
  policy outcome. The run still completes and writes every sink.

## Alternatives considered

- **Fail-fast.** Simple, but one broken check hides every other result, which is wrong for a profiler.
- **A built-in default gate.** Surprising for embedded users, and thresholds belong to the user's pipeline.

## Consequences

- No built-in quality gate. A user who wants one writes a policy (Phase 5).
- Check authors don't write try/except. The executor, planner and notes replace about 60 copies of it.
- Every status needs tests and a visible explanation, in the CLI summary and the dashboard's run health panel.
- Setup-error handling remains a special case that needs its own tests.
