# ADR 0011: Toolchain: uv, pytest tiers, pyrefly, pnpm

**Status:** Accepted · **Date:** 2026-10-16 (decided 2026-10-15) · **Design:** [`REFACTOR_PLAN.md`](../REFACTOR_PLAN.md) §11, §12

## Context

The repository used Poetry, npm and no type checker, while the maintainer's other projects use uv,
pyrefly and pnpm. Aligning them lowers the cost of moving between projects, and a type checker
matters most when new code is written against typed contracts ([ADR 0005](0005-pydantic-contracts.md)).

## Decision

- **Python dependencies and builds: uv.** `pyproject.toml` uses `[project]` metadata,
  `[dependency-groups]` and the `uv_build` backend. `uv.lock` is the lockfile, CI installs with
  `astral-sh/setup-uv`, and ruff's version is taken from the lock. Dependabot covers `uv` and `github-actions`.
- **Tests: pytest tier markers** (`unit`, `integration`, `slow`, `network`, `golden`, `perf`) with a
  fast default gate (`pytest -m "unit and not slow"`). Golden and perf tiers run nightly, on demand,
  and on labelled PRs.
- **Types: pyrefly**, pinned as a dev dependency, with the `default` preset and a committed baseline
  (`.pyrefly-baseline.json`) that holds the errors in legacy code, so new code is checked in full
  without fixing the old first.
- **Frontend: pnpm with ESLint and Vitest**, plus `vite build`. This arrives with the frontend move
  in Phase 1.
- **Lint and format: ruff**, as before.
- Generated files are regenerated through their owning command, never by hand (`uv lock`,
  `pnpm install`, `dsbf schema export`, the golden update option, the perf harness).

## Alternatives considered

- **Poetry and npm, no type checker** (the starting point). Works, but misaligned with the
  maintainer's other projects, and with nothing checking the new typed code.

## Consequences

- The uv, markers and pyrefly parts have landed in Phase 0. The frontend part is not done yet: until
  Phase 1 the frontend lives in `dsbf/dashboard/frontend` with npm.
- Python 3.14 is out of the CI matrix until the locked dependencies publish cp314 wheels.
- The pyrefly baseline shrinks as legacy modules are migrated and deleted, and `--error-stale-baseline`
  makes the gate fail if it doesn't.
