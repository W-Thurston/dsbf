---
description: Close out the active unit in PLAN.md (commit, records, PR)
---

# Close out the active unit

Run this when every acceptance check in `PLAN.md` passes. Work through the steps in order. Each
step ends on its *done when* condition.

1. **Gates on the final tree.**
   - Run the Python gate from `CLAUDE.md` practice 7. Run the frontend gate if frontend files changed.
   - Run every other tier the unit's boundary touches (`integration`, `slow`, `network`, `golden`, `perf`).
   - Record the results under Test evidence in `PLAN.md`.
   - *Done when* every gate is green.
2. **Real-artifact check.**
   - Confirm `PLAN.md` records what you inspected directly (practice 8) and what you found.
   - If the unit changed behaviour that produces artifacts or responses and nothing was inspected, inspect now.
   - *Done when* Test evidence names at least one real artifact or response for every behaviour change.
3. **Save the PR body.**
   - Copy the final `PLAN.md` content to a scratch file, starting with the line `Unit <ID> · <title>`.
   - *Done when* the file exists outside the repo.
4. **Update `ROADMAP.md`.**
   - Set the unit's status to `done (<YYYY-MM-DD>)` and add 1–3 lines of notes: what changed and any deviations.
   - Add every follow-up from `PLAN.md`, either as a new `todo` unit in the right phase (full unit format) or as a note on an existing unit.
   - *Done when* no follow-up is left unrecorded.
5. **Program closeout** (only when this unit is the last unit of its phase).
   - Replace the phase's detailed unit entries with a short **program summary**: the capability delivered, success criteria with measured results, key decisions with ADR links, deviations from the phase block, and the list of PRs.
   - Make sure the next phase's planning unit (`Ux-00`) is `todo`.
   - Update `docs/REFACTOR_PLAN.md` if the program changed the design.
   - *Done when* the phase reads as a summary, and the next phase is ready to plan.
6. **Prune `CLAUDE.md`.**
   - Delete every Current gotchas line that names this unit, and any other line the unit made stale.
   - *Done when* no line in `CLAUDE.md` describes behaviour that no longer exists.
7. **Reset `PLAN.md`** to its stub:

   ```markdown
   # PLAN.md

   No active unit. Run `/start-unit` to start the next one (see [`CLAUDE.md`](CLAUDE.md)).
   ```

8. **Commit.**
   - Stage everything, including `PLAN.md` and `ROADMAP.md`. If the branch has earlier work-in-progress commits, squash them, so the branch holds exactly **one** commit on top of `main`.
   - Subject: the unit's Conventional Commit message. It's descriptive, with no unit or finding IDs.
   - Body: bullets grouped under `Implementation:`, `Tests:` and `Documentation:`.
   - *Done when* `git log main..HEAD` shows one commit and `git status` is clean.
9. **Push and open the PR.**
   - `git push -u origin HEAD`
   - `gh pr create --title "<commit subject>" --body-file <saved PLAN.md content>`
   - Watch CI with `gh pr checks --watch`. If it fails, fix the problem, amend the single commit, and push with `--force-with-lease`.
   - *Done when* the PR is open and its CI is green. The user reviews and merges. The next unit records the PR number and merge SHA when it starts.
