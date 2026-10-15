---
description: Start the next unit from ROADMAP.md (or the unit ID given) and scope it in PLAN.md
argument-hint: "[unit ID, e.g. U0-03]"
---

# Start a unit

Requested unit: `$ARGUMENTS` (empty means the next one).

Read `CLAUDE.md`'s Ways of working first. Then work through these steps in order. Each step ends on
its *done when* condition.

1. **Check that nothing is active.**
   - Read `PLAN.md`. If it holds a unit, stop and report which one. Finish it, or run `/closeout`, before starting another.
   - *Done when* `PLAN.md` is the stub.
2. **Select the unit.**
   - Use the requested ID if one was given. Otherwise take the first `todo` unit in `ROADMAP.md`, in document order, whose dependencies are all `done`.
   - If the requested unit has dependencies that aren't `done`, stop and list them.
   - State the selected ID and title, and today's date (`YYYY-MM-DD`).
   - *Done when* exactly one unit is selected.
3. **Backfill the previous unit.**
   - If the most recently `done` unit has no PR number and merge SHA, find them with `gh pr list --state merged --search "<commit subject>"` and `git log`, and add them to its entry.
   - *Done when* the previous `done` unit's entry carries its PR number and SHA, or there is no previous unit.
4. **Branch.**
   - `git switch main && git pull --ff-only`, then `git switch -c unit/<ID>-<slug>`.
   - If uncommitted changes block the switch, stop and ask. U0-01 is the exception: it parks the existing work as its first step.
   - *Done when* you're on the unit branch, cut from the current `origin/main`.
5. **Scope it in `PLAN.md`.**
   - Fill the `PLAN.md` template from `CLAUDE.md`, copying the unit verbatim from `ROADMAP.md`.
   - Set the unit's status in `ROADMAP.md` to `active`.
   - *Done when* `PLAN.md` holds exactly this unit.
6. **Confirm the assumptions** (`CLAUDE.md` unit workflow, step 2), then **design** (step 3). Stop to ask the user wherever the unit says **ask the user**, or where a changed assumption alters scope.
   - *Done when* every assumption has a recorded outcome and the design decisions are written. Then report a short summary of the outcomes and decisions, and continue to build.
