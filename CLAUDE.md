# DSBF: working agreement

Work in this repo is planned at two levels:
- A **program** is a phase in `ROADMAP.md`. It records capability, motivation, boundaries, sequence and success criteria.
- A **unit** is one bounded piece of a program: one branch, one commit, one PR and one session.

The documents involved:
- `ROADMAP.md` is the backlog of programs and their units: status, dependencies, assumptions, steps and acceptance criteria.
- `PLAN.md` holds the **active unit**, at most one, and nothing else.
- `docs/REFACTOR_PLAN.md` is the design, and `docs/adr/` records the architecture decisions (D1–D11) it rests on. `docs/REVIEW_FINDINGS.md` defines the F-xx and N-xx IDs that units cite. `docs/WALKTHROUGH.md` explains how a run works today.

## Ways of working

Read this section before planning or modifying the repository.

1. **Confirm before building.** Inspect the current repository state before proposing a change. Code, schemas, artifacts, commands, tests and documentation may differ from what a plan or earlier context says. A small read-only audit beats implementing from stale context.
2. **Locate first, then read the owning boundary.** Use targeted searches to find the relevant files, functions, tests, commands, artifacts and generated contracts. Read the exact owning boundaries before designing or editing them.
3. **Design at two levels before implementation.** The program level is locked in the phase block in `ROADMAP.md`. The unit level is the unit in `PLAN.md`, with its goal, design decisions, tests and acceptance criteria, written before code changes.
4. **Keep one active bounded unit.** Start new work only after selecting it from `ROADMAP.md` and scoping it in `PLAN.md`. Completed units are recorded in `ROADMAP.md`. At program closeout, their detailed entries are condensed into a program summary.
5. **Use descriptive implementation language.** Program, unit and finding IDs (`U0-03`, `F-12`) belong in planning documents only: `ROADMAP.md`, `PLAN.md`, `docs/REFACTOR_PLAN.md`, `docs/REVIEW_FINDINGS.md`, ADRs and PR bodies. Source names, comments, docstrings, tests, artifacts, commands and commit subjects describe lasting domain behaviour.
6. **Commit small coherent units.** Each completed unit produces one focused commit:
   - a Conventional Commit subject;
   - a detailed bullet-list body covering implementation, tests and documentation.

   The `PLAN.md` and `ROADMAP.md` updates belong in the same commit as the implementation. The PR title is the commit subject.
7. **Run focused gates during implementation.** Run linting, type checking and focused tests after each meaningful change.

   At a Python contract boundary, run:

   ```bash
   uv run ruff check . --fix && \
   uv run ruff format . && \
   uv run pyrefly check --baseline .pyrefly-baseline.json --error-stale-baseline && \
   uv run pytest -m "unit and not slow"
   ```

   At a frontend contract boundary, run:

   ```bash
   cd frontend
   pnpm lint
   pnpm build
   pnpm test:run
   cd ..
   ```

   Run the `integration`, `slow`, `network`, `golden` or `perf` tiers when the affected boundary requires them.
8. **Verify against real artifacts and responses.** After engine, data, storage, API or frontend changes, run DSBF on real data in an isolated scratch directory (with its own `DSBF_DB_PATH`). Then inspect the output directly: the `Run` or report, database rows, API responses, and the rendered dashboard. Check:
   - row and column counts, check statuses and their reasons;
   - identities and uniqueness, coverage of eligible columns;
   - sampling provenance, representative metric values, timestamps;
   - non-finite values, and payload sizes.

   Green tests don't replace real-data verification.
9. **Preserve generated-file ownership.** Regenerate checked-in derived files through their owning command, never by hand:
   - lockfiles: `uv lock` / `pnpm install`;
   - JSON Schema and TypeScript types: `dsbf schema export --types frontend/src/generated/run.d.ts` (writes `schema/run.schema.json` too; add `--check` to verify);
   - golden snapshots: the golden suite's update option;
   - `src/dsbf/static_metadata/task_metadata.json`: `dsbf tasks export --output src/dsbf/static_metadata/task_metadata.json`;
   - performance baselines: the perf harness.
10. **Close each unit completely.** Follow `.claude/commands/closeout.md` (run it as `/closeout`).
11. **Don't preserve development-era compatibility without a current need.** DSBF has never had a published release or external users (it isn't on PyPI, and the repo has no forks, as of 2026-10-15). Development-era schemas, configs, commands, tests, generated contracts and historical behaviour can be replaced when the active design requires it, unless a current contract explicitly requires compatibility. Compatibility guarantees begin with the published 1.x result schema.
12. **Use explicit dates and repository history.** Trust explicit user-provided dates, commit timestamps and repository history. When dates conflict or are ambiguous, state the exact date being used (`YYYY-MM-DD`) rather than relative wording.

## Unit workflow

1. **Start.** Run `/start-unit` (optionally with a unit ID). It selects the unit, backfills the previous unit's PR and SHA, cuts the branch, and scopes the unit into `PLAN.md`.
2. **Confirm** (practices 1 and 2).
   - Test every "Assumptions to confirm" item against the code, and record each one in `PLAN.md` as confirmed or changed, with evidence.
   - When an item says **ask the user**, or a changed assumption alters scope, stop and ask before going further.
   - *Done when* every assumption has a recorded outcome.
3. **Design** (practice 3).
   - Write the unit's design decisions in `PLAN.md`: interfaces, data shapes, file placement, test plan.
   - *Done when* every step of the unit maps to a decision or is mechanical.
4. **Build and test** (practices 7–9).
   - Implement the steps within scope, running the gates as you go.
   - Record out-of-scope discoveries under Follow-ups.
   - Record the commands and results, plus the real-artifact checks, under Test evidence.
   - *Done when* every acceptance check passes.
5. **Close out.** Run `/closeout`.

### `PLAN.md` template

```markdown
# PLAN.md: <ID> · <title>
Branch: unit/<ID>-<slug> · Started: <YYYY-MM-DD>

## Unit
<copied verbatim from ROADMAP.md>

## Assumption outcomes
- <assumption>: confirmed | changed: <evidence>

## Design decisions
- <decision>: <reason>

## Follow-ups
- <finding> (suggested phase or unit)

## Test evidence
- `<command>`: <result>
- Real artifact: <what was inspected>: <result>
```

## Current gotchas

Each gotcha names the unit that removes it. Delete the line at that unit's close-out.

None at present.
