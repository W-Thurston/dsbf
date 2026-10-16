# DSBF Roadmap

This is the backlog of upcoming work. Each **phase** is a program: its capability, motivation,
boundaries, sequence and success criteria are locked at the top of the phase. Work inside a phase
is broken into **units**. A unit is one branch, one PR, one commit and one session. The workflow for running a unit is in [`CLAUDE.md`](CLAUDE.md), and the active unit (at
most one) lives in [`PLAN.md`](PLAN.md).

- **Design:** [`docs/REFACTOR_PLAN.md`](docs/REFACTOR_PLAN.md)
- **Findings cited as F-xx / N-xx:** [`docs/REVIEW_FINDINGS.md`](docs/REVIEW_FINDINGS.md)
- **How a run works today, with measurements:** [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md)

**Status values:**
- `todo`: ready once its dependencies are `done`
- `active`: currently in `PLAN.md`
- `done`: merged; the entry carries notes and the PR number. When a phase's last unit closes,
  its unit entries are condensed into a short program summary (see `.claude/commands/closeout.md`).
- `placeholder`: a later phase, detailed by that phase's planning unit

**Next unit:** the first `todo` unit, in order, whose dependencies are all `done`.

Phases 0–3 are broken down in full. Phases 4–9 are placeholders. Each starts with a `Ux-00`
planning unit that turns the placeholders into full units, using what earlier phases taught.

---

## Phase 0: Stabilise and baseline

- **Capability:** A trustworthy, measured baseline on the target toolchain. That means green CI (uv, pytest markers, pyrefly), hermetic tests, no known task crashes, committed golden and performance baselines, and accepted architecture decisions.
- **Motivation:** Every later phase is judged by diffing against these baselines. They only mean something if the current code is stable and measured first.
- **Boundaries:** Work on the **current** code is limited to blockers and cheap safety fixes (REFACTOR_PLAN §12, Phase 0). Other findings wait for the code that replaces them. No new features.
- **Sequence and dependencies:** Toolchain first (CI, tests, dependencies, uv, markers), then isolation and crash fixes, then hygiene and type checking, then baselines and decisions.
- **Success criteria:** Every Phase 0 unit is `done`. CI runs the fast, full, type-check, golden and perf jobs green. ADRs D1–D11 are accepted.

### U0-01 · Park in-flight chart work and adopt the unit workflow
- **Status:** done (2026-10-15) · **Notes:** ECharts work parked as `d4d949f` on `origin/wip/echarts-charts` (histogram-bin warning in the commit body). The NFL config edits, metadata churn, `poetry.lock` and `.vscode` edits were discarded, and local `feature/viz_refactor` was deleted (never on origin; tree identical to `main`). **Deviation:** the new CI lint (unpinned ruff 0.17) also formats Python code blocks in Markdown, and `main` was already red on it, so the code blocks in `CONTRIBUTING.md` and `docs/REFACTOR_PLAN.md` were reformatted (whitespace only) to get this PR green; ruff 0.11 is unaffected. U0-02 still owns pinning ruff. PR #29, merged as `85f8697`. · **Size:** S · **Depends on:** none · **Resolves:** F-05
- **Goal:** Move the uncommitted ECharts work to its own branch, sync `main`, and commit the planning documents so the unit workflow exists on `main`.
- **Context:** `git status` on `feature/viz_refactor`. That branch's *committed* work is already on `main` (tree-identical to local `main`; `origin/main` adds 4 Dependabot merges). F-12: the parked histogram uses percentile bins and must not ship as-is.
- **Assumptions to confirm:**
  - The uncommitted chart files (`DsbfChart.vue`, `chartTheme.js`, `BoxPlotChart.vue`, `DistributionHistogramChart.vue`, `ValueCountsChart.vue`, plus edits to `DistributionsTab.vue`, `ValueCountsTable.vue`, `main.js`, `package.json`, `package-lock.json`) should be kept on a branch, not discarded. **Ask the user.**
  - These working-tree changes are discarded: `dsbf/config/default_config.yaml` (NFL time-series settings, F-05), `dsbf/static_metadata/task_metadata.json` (run churn), `poetry.lock` (superseded by `origin/main`). **Ask about `.vscode/settings.json`.**
  - `feature/viz_refactor` has nothing left that isn't on `main`. **Ask before deleting it locally and on origin.**
  - The untracked root files (`clean_dsbf_db.txt`, `dsbf_*.md/.csv/.json`, `package-lock.json`, `tests/generators/dsbf_test_generator_backup.py`) stay untracked until U0-11.
- **Scope:** In: branch hygiene and the planning docs. Out: any code change.
- **Steps:**
  1. `git switch -c wip/echarts-charts`, then stage only the chart files listed above and commit `wip(dashboard): park ECharts chart components`. Put "the histogram bins on percentiles and misrepresents distributions; fix before merging" in the commit body. Push.
  2. `git restore` the discarded files. `git switch main && git pull --ff-only`.
  3. `git switch -c unit/U0-01-adopt-unit-workflow`. Add `ROADMAP.md`, `PLAN.md`, `CLAUDE.md`, `.claude/commands/closeout.md`, `.claude/commands/start-unit.md`, `docs/REFACTOR_PLAN.md`, `docs/REVIEW_FINDINGS.md`, `docs/WALKTHROUGH.md` and `docs/practice_run/`.
  4. Confirm `docs/practice_run/*.py` still pass ruff (they did when written).
- **Acceptance:** `wip/echarts-charts` exists on origin with the chart work. On the unit branch, `git status` is clean apart from intended files. `ruff check .` and `ruff format --check .` pass locally. `PLAN.md` is the stub after close-out.
- **Commit:** `docs: adopt unit workflow and add refactor plan, review findings and walkthrough`

### U0-02 · Make CI lint match local tooling
- **Status:** done (2026-10-15) · **Notes:** The lint job installs the ruff version read from `poetry.lock`; the pre-commit rev is aligned to `v0.11.13`. **Deviation:** all four actions moved to Node 24 majors (checkout v7, setup-python v7, cache v6, upload-artifact v7), not just checkout and setup-python. PR #31, merged as `4b80da8`. · **Size:** S · **Depends on:** U0-01 · **Resolves:** F-01 (lint part)
- **Goal:** The CI lint job uses the ruff version pinned in `poetry.lock`, and actions run on Node 24.
- **Context:** F-01. `.github/workflows/ci.yml:22` runs an unpinned `pip install ruff`. `.pre-commit-config.yaml` pins `v0.11.10`.
- **Assumptions to confirm:**
  - The lint failure is only formatter drift between ruff versions. Confirm with `uvx ruff@latest format --check .`, or the latest failed run's log.
  - Current majors of `actions/checkout` and `actions/setup-python` target Node 24 (check their release notes).
- **Scope:** In: `ci.yml` and pre-commit rev alignment. Out: the failing tests (U0-03).
- **Steps:**
  1. Install ruff from the lock in the lint job (`poetry install --only dev`, or `pip install "ruff==<locked>"`).
  2. Bump the action majors.
  3. Align the pre-commit `rev` with the lock.
  4. Add a comment explaining the pin.
- **Acceptance:** On the PR, the lint job is green and the test job runs. The test job may fail with only the 5 known F-03 failures; record them.
- **Commit:** `ci: pin ruff to the locked version and move actions to Node 24`

### U0-03 · Fix the five failing tests
- **Status:** done (2026-10-15) · **Notes:** `make_empty_result` takes a `data` argument, and the 18 fixed-key tasks pass their real empty shape (30 per-column tasks keep `{}`). **Deviation:** the five failures had four causes, not one: only the two bimodal tests were the empty-shape bug. The normality and encoding failures were test setup (no `infer_types`; a task built without its registry name), and the memory test needed the helper to accept a duck-typed stand-in. New `test_empty_results.py` pins the 18 shapes. PR #32, merged as `fb6d825`. · **Size:** M · **Depends on:** U0-02 · **Resolves:** F-03
- **Goal:** `pytest` passes locally and in CI on 3.12 and 3.13.
- **Context:** F-03. `BaseTask.make_empty_result()` (`dsbf/core/base_task.py:392`) returns `data={}`, which drops task-specific keys. `tests/helpers/context_utils.py:51` requires registered task classes.
- **Assumptions to confirm:**
  - CI test failures on `main` are exactly these 5 tests, with no network or Python 3.14 issues. Check with `gh run view --log-failed` on the latest `main` run.
    - Seen on the U0-01 PR run (2026-10-15): 3.12 and 3.13 fail exactly these 5 tests. The 3.14 job (`experimental: true`, non-blocking) fails earlier, at install, because scipy 1.16.0 has no 3.14 build (PEP 517 sdist build error). Decide whether to bump scipy (U0-04 covers dependencies) or keep 3.14 experimental.
  - The `normality_tests` and `suggest_categorical_encoding` failures share the empty-result root cause. Not yet proven.
- **Scope:** In: `make_empty_result` contract, the affected tasks, test helper registration. Out: other behaviour changes.
- **Steps:**
  1. Add a `data=` parameter to `make_empty_result`, and pass each migrated task's empty shape (for example `{"bimodal_flags": {}}`). Audit every caller.
  2. Root-cause and fix the normality and encoding failures.
  3. Let the memory test register its task through a fixture (or let the helper accept unregistered classes).
- **Acceptance:** `poetry run pytest` passes with 0 failures. CI test jobs are green for 3.12 and 3.13.
- **Commit:** `fix(core): keep task-specific empty result shapes and repair failing tests`

### U0-04 · Clear the Dependabot backlog
- **Status:** done (2026-10-15) · **Notes:** Seven of the eight listed PRs merged (#23, #26, #24, #28, #27, #20, #22), plus #36 (axios 1.20.0) in place of #25, which Dependabot closed as superseded; the user chose to merge #36. Smoke checks all passed: pillow 12.3.0 (47 PNGs decode), pyarrow 23.0.1 (parquet round trip and `dsbf profile` on a parquet file; constraint now `>=20,<24`), axios 1.20.0 (real `api.js` against the API), starlette 1.3.1 (every endpoint, 404s, figure file). **Deviations:** `@dependabot rebase` failed once, so `update-branch` through the REST API was used for #24, #28 and #36. Dependabot opened #34, #37, #38 and #39 during the unit, and these are left open and recorded under U0-05 and U1-03. Python 3.14 stays experimental (scipy 1.16.0 has no 3.14 wheel; the bump is recorded under U0-05). PR number and merge SHA: filled in by the next `/start-unit`. · **Size:** M · **Depends on:** U0-03 · **Resolves:** F-01 (backlog part)
- **Goal:** No stale Dependabot PRs; dependency majors are verified or declined with reasons.
- **Context:** Open as of 2026-10-15:
  - #20 pyarrow 20 → 23 (major; `pyproject.toml` pins `^20`)
  - #22 starlette 0.52 → 1.3 (major; must stay compatible with `fastapi ^0.135`)
  - #23 tornado
  - #24 form-data, #25 axios, #27 postcss, #28 nanoid (all in the frontend)
  - #26 pillow
- **How the packages are reached (checked in the lock on 2026-10-15):**
  - `tornado` is dev-only (through `ipykernel` and `jupyter-client`). `pillow` comes in through `matplotlib` only.
  - `fastapi` 0.135.1 declares `starlette >=0.46.0` with no upper bound, so FastAPI doesn't block starlette 1.x on paper. A smoke test still has to confirm it.
  - `pyarrow` is a direct dependency (`^20.0.0`) that no package in the lock requires and no code imports. Check whether the polars and pandas conversions need it before bumping the major.
  - In the frontend, `axios` is the only direct dependency. `form-data`, `nanoid` and `postcss` are transitive, so those three PRs change the lockfile only.
- **Assumptions to confirm:**
  - The list is still current (`gh pr list`).
  - **Decided by the user on 2026-10-15:** this unit may merge the Dependabot PRs itself (squash merge, as the history does). Limits: only the open Dependabot PRs listed here; never merge on a red lint job or a red 3.12 or 3.13 test job (3.14 is non-blocking); a major goes in only after its smoke checks pass.
  - The FastAPI version in the lock supports starlette 1.x. If not, close #22 with a note.
  - The Python 3.14 test job fails at install because scipy 1.16.0 has no 3.14 build (seen in U0-03). Decide whether a scipy bump fixes it or 3.14 stays experimental, and record the answer.
- **Scope:** In: rebasing (`@dependabot rebase`), verification, merging or closing, and any `pyproject.toml` constraint change needed for pyarrow. Out: unrelated upgrades.
- **Steps:**
  1. Merge in this order, one at a time. The PRs share `poetry.lock` or `package-lock.json`, so after each merge comment `@dependabot rebase` on the next PR and wait for green CI before merging it. The order goes from lowest to highest risk, with the PRs that are likeliest to be declined last:
     1. #23 tornado 6.5.5 → 6.5.7: patch, dev-only. CI is enough.
     2. #26 pillow 12.2 → 12.3: minor, reached only through matplotlib. Run `dsbf quickstart titanic` and check that figures are written.
     3. #24 form-data, #28 nanoid, #27 postcss: patch bumps of transitive frontend packages (lockfile only). Run `npm ci && npm run build`.
     4. #25 axios 1.15 → 1.18: minor, the one direct frontend dependency (its `package.json` range changes). Run `npm ci && npm run build`, then load the dashboard against a running API and open a dataset page.
     5. #20 pyarrow 20 → 23: major. Raise the `pyproject.toml` constraint if the PR needs it. Run `pytest` and `dsbf quickstart titanic`.
     6. #22 starlette 0.52 → 1.3: major, last because it is the likeliest to be declined. Run `pytest` and the API smoke test (`uvicorn`, then `curl /health` and `/api/datasets`). Close with a note if it is incompatible.
  2. For each major (5 and 6), check out the PR branch and run the checks listed there before merging. Merge, or close with the reason.
  3. Record outcomes in the close-out notes.
- **Acceptance:** `gh pr list --author app/dependabot` shows no PR older than this unit. `main` CI is green after the merges.
- **Commit:** `chore(deps): record dependency backlog outcomes` (the close-out commit; the dependency changes come from the merged PRs)

### U0-05 · Migrate dependency management from Poetry to uv
- **Status:** todo · **Size:** M · **Depends on:** U0-04 · **Resolves:** toolchain alignment (ADR D11)
- **Goal:** `pyproject.toml` uses standard `[project]` metadata, `uv.lock` replaces `poetry.lock`, and CI, docs and contributor instructions use uv.
- **Context:** `pyproject.toml` (`[tool.poetry]`, `[tool.poetry.scripts]`, dev group), `.github/workflows/ci.yml` (Poetry install and cache), README and CONTRIBUTING install commands.
- **Assumptions to confirm:**
  - The build backend: `hatchling` or `uv_build`. **Ask the user.**
  - `uv lock` resolves the same versions as `poetry.lock` for direct dependencies. Diff `uv pip freeze` against `poetry export` and explain each difference.
  - The dev group maps to `[dependency-groups] dev`, and Dependabot supports the `uv` ecosystem for this repo.
- **Note from U0-04:** `scipy` 1.16.0 (transitive, locked) has no Python 3.14 wheel, which is why the 3.14 CI job fails at install; 1.16.1 and later do. Let `uv lock` pick a release with cp314 wheels, then decide whether 3.14 stays `experimental`. Also, Dependabot PRs #37 (virtualenv 20.36.1 → 21.7.13) and #39 (anyio 4.9.0 → 4.14.2) were opened against `poetry.lock` during U0-04 and left open. They are security updates (alerts 96 critical and 95 for anyio, 104, 105, 109 and 110 for virtualenv), so after the migration confirm that `uv.lock` resolves anyio ≥ 4.14.2 and virtualenv ≥ 21.7.13, and close the two PRs with a pointer here. Dependabot also regenerates `poetry.lock` with Poetry 2.5.1 (the lock header says 2.2.1), so lock diffs carry format churn until `poetry.lock` is gone.
- **Note from U0-02:** the repo has no `.github/dependabot.yml`. Add one with the `uv` and `github-actions` ecosystems when the Dependabot ecosystem moves.
- **Scope:** In: the pyproject conversion, `uv.lock`, deleting `poetry.lock`, CI on `astral-sh/setup-uv` with `uv sync --frozen`, the lint job pinned through the lock, doc and CLAUDE.md command updates. Out: version upgrades beyond what resolution forces.
- **Steps:**
  1. Convert the metadata, scripts and groups.
  2. `uv lock` and compare resolutions.
  3. Update CI and docs.
  4. Delete `poetry.lock` and `[tool.poetry]`.
- **Acceptance:** From a fresh clone, `uv sync && uv run pytest` passes and `uv run dsbf version` works. CI is green. No Poetry references remain in code, CI, README or CONTRIBUTING (CHANGELOG and planning docs keep their history).
- **Commit:** `build: migrate dependency management from Poetry to uv`

### U0-06 · Test tier markers and a fast gate
- **Status:** todo · **Size:** S · **Depends on:** U0-05
- **Goal:** Every test carries a tier marker (`unit`, `integration`, `slow`, `network`; `golden` and `perf` are reserved), unmarked tests fail collection, and the fast gate `uv run pytest -m "unit and not slow"` runs on every push.
- **Context:** `[tool.pytest.ini_options]` in `pyproject.toml`. Tests that run the whole engine (`clean_engine_run`, the validation suite) or download built-in datasets.
- **Assumptions to confirm:**
  - Which tests touch the network (seaborn, sklearn and OpenML loaders).
  - Per-file runtimes (`--durations=0`), to classify `slow`.
  - A fast-gate time target. Propose one from the measurements.
- **Note from U0-02:** the test job's `upload-artifact` step uploads the hidden `.pytest_cache/`, which the action skips by default, so no artifact is stored. Fix it in the CI changes here (`include-hidden-files: true` or another path).
- **Note from U0-04:** there are no API tests (nothing in `tests/` uses `TestClient`), so the starlette 1.x bump was covered only by a manual smoke run (every `/api` endpoint, 404s and the figure file). Add an `integration` test that starts the FastAPI app against a small run in a temporary database and checks the same endpoints.
- **Scope:** In: marker registration with `--strict-markers`, a collection hook that fails unmarked tests, marking via `pytestmark` per module, and CI jobs for the fast gate and the full suite. Out: speeding up slow tests.
- **Acceptance:** Collection fails for an unmarked test (proved with a temporary test). The fast-gate time is recorded. Both CI jobs are green.
- **Commit:** `test: classify tests by tier and add a fast gate`

### U0-07 · Run tests hermetically from a temporary directory
- **Status:** todo · **Size:** M · **Depends on:** U0-06 · **Resolves:** F-02, F-31
- **Goal:** The test suite and DSBF itself run from any directory without touching the developer's DB, outputs or tracked files.
- **Context:**
  - F-02: `tests/conftest.py:27` sets an ignored `engine.output_path`, and no `DSBF_DB_PATH` is set.
  - F-31: `dsbf/utils/reco_engine.py:7` loads package data from a CWD-relative path. 12 tasks fail outside the repo root.
- **Assumptions to confirm:**
  - The full list of CWD-relative package-data reads. Grep `Path("dsbf/`, `open("dsbf/`, and `"dsbf/` string literals.
  - Which tests depend on the CWD being the repo root.
- **Note from U0-03:** a running pytest deletes run directories under the shared `dsbf/outputs/` (a CLI run overlapped a test run and lost its output). Acceptance should include that a test run leaves other runs' output directories alone.
- **Scope:** In: an autouse session fixture (tmp `DSBF_DB_PATH`, `DSBF_AUTO_EXPORT_METADATA=0`, `chdir` to a tmp dir); fixing `clean_engine_run` to pass `output_dir`; `importlib.resources` for every package-data **read**; making a missing tip non-fatal in `get_recommendation_tip`. Out: moving output, DB or metadata **write** locations (U1-04).
- **Steps:**
  1. Add the fixture.
  2. Fix the fixtures and tests that break under it.
  3. Switch package-data reads to `importlib.resources`.
  4. Wrap the tip lookup so a failure logs and returns `None`.
- **Acceptance:**
  - After `uv run pytest`, `git status` is unchanged and `dsbf/dsbf.db` keeps its mtime.
  - `cd "$(mktemp -d)" && uv run --project <repo> dsbf quickstart titanic` finishes with **0** `FileNotFoundError` task failures.
- **Commit:** `test: isolate test runs and load package data via importlib.resources`

### U0-08 · Make the default config pass its own validation
- **Status:** todo · **Size:** S · **Depends on:** U0-07 · **Resolves:** F-04
- **Goal:** `validate_config_and_graph(load_default_config())` returns no errors, and strict mode works with the defaults.
- **Context:** F-04. `tasks.time_series` (`default_config.yaml:285`) is a shared block that `config_validation.py:27` flags as an unknown task. `BaseTask.get_shared_param()` reads shared blocks from under `tasks`.
- **Assumptions to confirm:**
  - `time_series` is the only shared block. Grep every `get_shared_param` caller.
  - Moving it to a top-level `time_series:` section is acceptable (Phase 3 replaces config handling anyway).
- **Scope:** In: config layout, `get_shared_param`, the validator, a regression test. Out: other config redesign.
- **Steps:**
  1. Move the block and update `get_shared_param`. Read only the new location; DSBF has no external configs to stay compatible with.
  2. Add a test asserting the default config validates cleanly.
  3. Add a strict-mode titanic run test.
- **Acceptance:** The new tests pass. `dsbf run -c <default config with safety.strict_mode: true>` completes.
- **Commit:** `fix(config): move shared time_series settings out of the tasks block`

### U0-09 · Stop task crashes on adversarial data
- **Status:** todo · **Size:** M · **Depends on:** U0-07 · **Resolves:** F-33, F-32 (crash guard only), F-10 (leakage crash only)
- **Goal:** Every registered task completes, without an exception, on a frame of hostile but legal data.
- **Context:**
  - F-33: `summarize_modes.py:59` crashes on an all-null column; a log f-string at `normality_tests.py:223` formats `None`.
  - F-32: `detect_collinear_features.py:85` ends up with zero rows after `dropna()`.
  - F-10: `detect_data_leakage.py:110` splits pair keys on `|`.
  - The adversarial frame spec is in `docs/practice_run/make_data.py`.
- **Assumptions to confirm:**
  - Running all tasks on a 200-row frame stays under about 60 s in the test suite. If not, mark the test `slow`.
  - No other task crashes on the frame. Run it first and list the crashes.
- **Scope:** In: a `tests/fixtures/adversarial.py` frame builder; a parametrised test running every registered task with dependencies; **minimal** guards for each crash (empty modes produce `None` plus a note; safe log formatting; a VIF row-count guard returning an empty result with a message; leakage keeping `(col1, col2)` tuples internally). Out: statistical redesign (Phase 4) and changing output key formats (Phase 1).
- **Steps:**
  1. Add the fixture and parametrised test, and confirm they fail.
  2. Fix each crash minimally.
  3. Anything larger than about 30 lines becomes an `xfail` with a descriptive reason, plus a follow-up in `ROADMAP.md`.
- **Acceptance:** The adversarial test passes. Rerunning `docs/practice_run/run_stages.py kitchen5k …` from a temp dir shows **0** task errors.
- **Commit:** `fix(tasks): guard against all-null, non-finite and delimiter-bearing columns`

### U0-10 · Escape user text in dashboard search highlighting
- **Status:** todo · **Size:** S · **Depends on:** U0-01 · **Resolves:** F-14
- **Goal:** Dataset names and run keys can't inject HTML.
- **Context:** F-14. `highlight()` is duplicated in `DatasetListView.vue:86` and `RunListView.vue:105` and rendered with `v-html`.
- **Assumptions to confirm:** No other `v-html` usages render data (grep `v-html` in `src/`).
- **Scope:** In: one shared `highlightHtml(text, query)` in `utils.js` that escapes first, used by both views. Out: frontend test infrastructure (U1-03).
- **Steps:** Implement the helper, replace both call sites, verify manually.
- **Acceptance:**
  - A dataset named `<img src=x onerror=alert(1)>` (created via `dsbf profile f.csv --name '<img src=x onerror=alert(1)>'` with an isolated `DSBF_DB_PATH`) renders as literal text, with highlighting still working.
  - `npm run build` passes.
- **Commit:** `fix(dashboard): escape names before highlighting search matches`

### U0-11 · Repository hygiene and README accuracy
- **Status:** todo · **Size:** M · **Depends on:** U0-07 · **Resolves:** F-27, F-28
- **Goal:** The root holds only intentional files, dead code is removed, and the README matches reality.
- **Context:** F-27 and F-28. The untracked root files are referenced only by `dsbf_directory_structure.md` and the review docs.
- **Assumptions to confirm (ask the user per file):**
  - Delete: `clean_dsbf_db.txt`, root `package-lock.json`, `tests/generators/dsbf_test_generator_backup.py`, `dsbf_directory_structure.md` (stale), `.flake8`.
  - Delete `dsbf_reference_dataset.csv` (5.3 MB, regenerable by the generator) and `dsbf_test_schema.json`, or move them to `tests/fixtures/`.
  - Move `dsbf_handoff_doc.md` to `docs/architecture.md`, trimmed to what's still true.
  - Remove `dsbf/utils/check_recommendation_contexts.py`, `dsbf/utils/display_registered_tasks.py`, `dsbf/layouts/default.json` and the commented `render_dashboard` in `cli.py`. Confirm they have no importers.
  - `run_profile.py` and `migrate_runs.py` stay until Phase 5 and 6 replace them.
  - The local branch `feature/time-series-tab` (`170a5e8`, never pushed; its tree differs from `main` in 3 files, including `poetry.lock`) is stale or still wanted. **Ask the user** whether to delete it.
- **Scope:** In: the files above; README fixes from F-27 (uv-based install, Python 3.12+, working API examples, real defaults, ECharts not D3, no type-checker badge until one runs, the README Roadmap section replaced by a link to `ROADMAP.md`). Out: new features.
- **Steps:**
  1. Confirm each file with the user and apply.
  2. Fix the README, executing every Python and CLI example in an isolated temp dir.
- **Acceptance:** Every README example runs as written. Tests and ruff pass. `git ls-files` at the root lists only intentional files.
- **Commit:** `chore: remove stray files and dead modules, correct the README`

### U0-12 · Adopt pyrefly type checking
- **Status:** todo · **Size:** M · **Depends on:** U0-11
- **Goal:** `pyrefly check` runs in CI and passes, with a committed baseline that suppresses existing errors in legacy modules, so new code is checked strictly.
- **Context:** No type checker is configured today. The legacy code has about 328 `Any` annotations.
- **Assumptions to confirm:**
  - The current pyrefly config format and suppression or baseline mechanism (read its docs for the installed version).
  - The error count on today's code.
  - **Ask the user:** run it unpinned through `uvx pyrefly check`, or pin it as a dev dependency (`uv run pyrefly check`)? An unpinned linter is what broke CI before (F-01).
- **Scope:** In: pyrefly configuration, a baseline of existing errors (no code fixes beyond trivial ones), a CI job, and the CLAUDE.md gate. Out: fixing legacy type errors at scale.
- **Acceptance:** The check exits 0 locally and in CI. A deliberate type error in a non-baselined module fails CI (proved on a throwaway commit).
- **Commit:** `build: add pyrefly type checking with a baseline for existing code`

### U0-13 · Golden-run baselines
- **Status:** todo · **Size:** M · **Depends on:** U0-08, U0-09 · **Resolves:** baseline for Phases 2–4
- **Goal:** A committed, deterministic snapshot of what today's code reports, used to review every later behaviour change.
- **Context:** REFACTOR_PLAN §11 and §14. Datasets: the 8 validation archetypes (`tests/generators`), titanic, and kitchen5k (generator seed 42).
- **Assumptions to confirm:**
  - Titanic can be vendored as a CSV fixture (check the seaborn-data license), so baselines don't need network access.
  - Runs are deterministic: run twice and diff. Known risks are isolation forest, sampling and set ordering in fuzzy candidate pairs.
  - **Runtime:** kitchen5k takes about 5 min, so the golden suite gets its own marker and CI job (nightly plus a PR label), not the default `pytest` run.
- **Scope:** In:
  - `tests/golden/` with a normaliser (drop timestamps, durations, memory and paths; round floats to 6 significant figures);
  - a snapshot per dataset (status per task, guidance titles and levels, key `data` summaries);
  - a `--update-golden` option;
  - a `golden` CI workflow.

  Out: fixing anything the baselines reveal (record as follow-ups).
- **Steps:**
  1. Write the normaliser and runner.
  2. Generate the snapshots and check determinism.
  3. Add the workflow.
  4. Document how to review and update the snapshots.
- **Acceptance:** Two consecutive runs produce byte-identical snapshots. The golden CI job is green. The snapshots are committed.
- **Commit:** `test: add golden-run baselines for archetypes, titanic and kitchen5k`

### U0-14 · Performance benchmark harness and baseline
- **Status:** todo · **Size:** M · **Depends on:** U0-07 · **Resolves:** baseline for the REFACTOR_PLAN §11 budgets
- **Goal:** A repeatable benchmark that records per-stage and per-task timings and memory, with today's numbers committed.
- **Context:** `docs/practice_run/` scripts and `docs/WALKTHROUGH.md` (measured on 2026-10-15: titanic 80 s, kitchen5k 298 s, kitchen50k 707 s and 2.1 GB).
- **Assumptions to confirm:**
  - Benchmarks run outside the default test run (`-m perf`).
  - A nightly CI runner is acceptable, given GitHub runners' timing noise. Compare with tolerance, and use relative regressions only.
- **Scope:** In: `tests/perf/` (harness promoted from `run_stages.py`, dataset generation, JSON results), a baseline file, and a nightly workflow that fails on a >25% per-stage regression. Out: performance fixes.
- **Steps:**
  1. Promote the scripts.
  2. Add the marker and workflow.
  3. Record the baseline on CI hardware.
  4. Delete `docs/practice_run/` (point the walkthrough at `tests/perf/`).
- **Acceptance:** `pytest -m perf` produces a results JSON. The nightly workflow runs green against the committed baseline.
- **Commit:** `test: add performance benchmark harness and baseline`

### U0-15 · Architecture decision records D1–D11
- **Status:** todo · **Size:** M · **Depends on:** U0-01 · **Resolves:** REFACTOR_PLAN §13
- **Goal:** Each of the ten design decisions is accepted (or amended) and recorded before Phase 1 code is written.
- **Context:** REFACTOR_PLAN §13 (D1 layering, D2 findings, D3 Polars/NumPy, D4 Ibis, D5 pydantic contracts, D6 report-only failure semantics, D7 entry-point plugins, D8 SQLAlchemy/Alembic/fsspec storage, D9 single distribution with extras, D10 "check" naming, D11 toolchain: uv, pyrefly, pnpm, decided 2026-10-15).
- **Assumptions to confirm:** **Each decision is walked through with the user.** Their answer is the confirmation. D11 is already decided; record it.
- **Scope:** In: `docs/adr/0001-…md` through `0011-…md` (context, decision, alternatives, consequences, status), plus edits to REFACTOR_PLAN if any decision changes. Out: code.
- **Acceptance:** 11 ADRs with status `Accepted` (or `Superseded` with a pointer). REFACTOR_PLAN is consistent with them.
- **Commit:** `docs(adr): record architecture decisions for the library-first redesign`

---

## Phase 1: Contracts

- **Capability:** The `src/` layout, a top-level frontend on pnpm, runtime paths under `DSBF_HOME`, optional extras, and the versioned result contracts (`dsbf.model`) with JSON Schema and TypeScript export.
- **Motivation:** All three usage modes depend on stable, published contracts (REFACTOR_PLAN §4). Embedded users consume the schema without the rest of DSBF.
- **Boundaries:** Packaging, paths and models only. No profiling or engine code. The legacy engine keeps working throughout.
- **Sequence and dependencies:** Depends on Phase 0. Layout and toolchain first, then contracts from the leaves (evidence) to the root (`Run`), then export.
- **Success criteria:** A hand-built `Run` validates, round-trips, and exports as JSON Schema and TypeScript, and the breaking-change gate is active in CI.

### U1-01 · Move the Python package to a `src/` layout
- **Status:** todo · **Size:** M · **Depends on:** U0-15 · **Resolves:** groundwork (REFACTOR_PLAN §3.1)
- **Goal:** `dsbf/` lives at `src/dsbf/` with no behaviour change.
- **Assumptions to confirm:**
  - Code that relies on the repo layout: `Path(__file__).parents[2]` in `storage/schema.py`, `api/main.py` and `api/db.py`, plus test paths.
  - The build backend's package discovery config.
  - CI caching keys.
- **Scope:** In: the move, imports, packaging config, path fixes, tests. Out: the frontend (U1-02) and runtime paths (U1-04).
- **Acceptance:** All tests and golden runs pass unchanged. `uv build` produces a wheel that installs in a clean venv, and `dsbf version` works from a temp dir.
- **Commit:** `refactor: move package to src layout`

### U1-02 · Move the frontend to a top-level `frontend/`
- **Status:** todo · **Size:** S · **Depends on:** U1-01
- **Goal:** The dashboard source no longer sits inside the Python package.
- **Assumptions to confirm:** The paths that reference `dsbf/dashboard/frontend`: Dependabot directories, `.nvmrc`, docs, and `.gitignore`.
- **Scope:** In: the move and path updates; `dsbf/dashboard/__init__.py` cleanup. Out: bundling the built frontend into the wheel (Phase 6).
- **Acceptance:** `npm ci && npm run build` works in `frontend/`. The dev proxy still reaches the API. Dependabot config points at the new path.
- **Commit:** `refactor(dashboard): move frontend to top-level directory`

### U1-03 · Frontend toolchain: pnpm, lint and unit tests
- **Status:** todo · **Size:** M · **Depends on:** U1-02 · **Resolves:** F-29 (tooling part)
- **Goal:** `frontend/` uses pnpm with `lint`, `build` and `test:run` scripts. ESLint (with `eslint-plugin-vue`) and Vitest are configured, and CI runs all three.
- **Context:** The frontend has only `dev`, `build` and `preview` scripts, plus a Vite 8 beta pin with `overrides` (F-29). The escaped search highlighter is a natural first unit test.
- **Assumptions to confirm:**
  - The pnpm version through the `packageManager` field and corepack.
  - A stable Vite release that can replace the beta pin.
  - **Ask the user:** include Prettier, or rely on ESLint stylistic rules?
- **Note from U0-04:** Dependabot PRs #34 (echarts 5 → 6.1.0, vue-echarts) and #38 (vue, @vue/server-renderer 3.5.42) were opened against `package-lock.json` during U0-04 and left open. They are security updates (alerts 112 medium for echarts, 115 high for @vue/server-renderer). Take both in this unit, checking the echarts 6 major against the dashboard charts, then close the PRs with a pointer here. `axios` is now `^1.20.0` and `postcss` and `nanoid` are at current patch releases.
- **Scope:** In: `pnpm-lock.yaml` (delete `package-lock.json`), scripts, ESLint and Vitest config with a few utility tests, a CI job, and the Dependabot ecosystem update. Out: refactoring components beyond lint autofixes. Remaining violations are suppressed per rule and recorded as follow-ups.
- **Acceptance:** `pnpm lint && pnpm build && pnpm test:run` pass locally and in CI.
- **Commit:** `build(dashboard): move frontend to pnpm with lint and unit tests`

### U1-04 · Runtime paths under `DSBF_HOME`
- **Status:** todo · **Size:** M · **Depends on:** U1-01 · **Resolves:** F-16 and F-17 for the legacy engine
- **Goal:** Outputs, the database, and run records go to one configurable home, so nothing depends on the CWD or writes into the package.
- **Context:**
  - `base_engine.py:93` and `:99` (outputs, `dsbf_run.json`);
  - `storage/schema.py:26` (DB inside the package);
  - `task_registry.py:276` (metadata export on every run);
  - API path resolution relative to the repo root.
- **Assumptions to confirm:**
  - The default home: `platformdirs.user_data_dir("dsbf")`, or `./.dsbf` per project. **Ask the user.**
  - Absolute paths are stored in the DB from now on. Old rows stay as they are.
- **Scope:** In: a `dsbf.paths` module; outputs, DB and record paths; metadata export becomes opt-in (`dsbf tasks export`); `record_run` removed or capped; API path resolution. Out: a new storage design (Phase 6).
- **Acceptance:** Running from any directory writes only under the home. The dashboard shows figures for a run started outside the repo. Tests pass.
- **Commit:** `feat: resolve runtime paths from DSBF_HOME`

### U1-05 · Declare optional-dependency extras
- **Status:** todo · **Size:** S · **Depends on:** U1-01
- **Goal:** `pyproject.toml` defines `ml`, `timeseries`, `text`, `pandas`, `sql`, `server` and `all` as in REFACTOR_PLAN §3.1. The base install doesn't change yet; legacy tasks still need everything.
- **Note from U0-04:** no DSBF code imports `pyarrow`; pandas uses it as its parquet engine (`pd.read_parquet` in `dsbf/api/db.py` and `dsbf/eda/profile_engine.py`) and polars doesn't need it. Put it in the extra that covers parquet input rather than the base install. The constraint is currently `>=20,<24` after the 23.0.1 bump.
- **Assumptions to confirm:** `[project.optional-dependencies]` resolves with `uv`, and `pip install dsbf[all]` from the built wheel resolves.
- **Acceptance:** `pip install "<wheel>[all]"` in a clean venv passes the test suite.
- **Commit:** `build: declare optional dependency extras`

### U1-06 · Contracts I: subjects, severities, dimensions and evidence
- **Status:** todo · **Size:** M · **Depends on:** U1-01 · **Resolves:** F-10, F-36 (schema side)
- **Goal:** `dsbf.model` provides `Subject`, `Severity`, `Confidence`, `Dimension`, `Evidence`, `Interval`, `AssumptionCheck`, `SamplingInfo` and `Timing`, as in REFACTOR_PLAN §4.2.
- **Assumptions to confirm:**
  - pydantic v2 is already a transitive dependency; make it direct.
  - The `Dimension` enum values (the five quality dimensions plus distribution, relationship and temporal).
- **Scope:** In: models, validators (finite floats only; NaN/inf → `None` plus a flag), unit tests including column names with `|`, brackets, unicode and empty strings. Out: Finding and Run (U1-07, U1-08).
- **Acceptance:** Round-trip tests pass. A non-finite float can't produce invalid JSON.
- **Commit:** `feat(model): add subject, evidence and severity contracts`

### U1-07 · Contracts II: findings, notes, check results and run health
- **Status:** todo · **Size:** M · **Depends on:** U1-06 · **Resolves:** F-20 (status model)
- **Goal:** `Finding` (with a stable id), `Guidance`, `Action`, `Note`, `CheckResult` (statuses `ok | not_applicable | skipped | unavailable | error | timeout`) and `RunHealth`.
- **Assumptions to confirm:**
  - The `Finding.id` hash inputs: check id, subject, code.
  - Mapping today's guidance blurbs onto `Guidance`/`Action` loses nothing. Compare with `BaseTask.add_guidance`.
- **Acceptance:**
  - Unit tests pass.
  - A test builds findings from a real legacy `TaskResult.guidance` sample without data loss.
  - `RunHealth` renders the plain-text summary shown in REFACTOR_PLAN §7.1.
- **Commit:** `feat(model): add finding, note, check result and run health contracts`

### U1-08 · Contracts III: profiles and the run manifest
- **Status:** todo · **Size:** M · **Depends on:** U1-07
- **Goal:** `ColumnProfile` (variants by semantic type), `DatasetProfile`, `DatasetManifest`, `PlanManifest` and `Run`, with `schema_version`.
- **Assumptions to confirm:** The profile fields cover everything listed in REFACTOR_PLAN §5.3 that the dashboard needs today. Check against current tab usage.
- **Acceptance:** A hand-built `Run` validates and round-trips through JSON. Its size for titanic-like content is under 200 KB without evidence artifacts.
- **Commit:** `feat(model): add profile and run manifest contracts`

### U1-09 · Configuration and parameter base models
- **Status:** todo · **Size:** M · **Depends on:** U1-06 · **Resolves:** F-22 (loader part)
- **Goal:** A `CheckParams` base class, and a layered config loader (defaults < project `dsbf.yaml` < env vars < CLI < keyword arguments) with deep merge and precise errors.
- **Assumptions to confirm:** The env var naming (`DSBF__SECTION__KEY`), and whether `pydantic-settings` is acceptable.
- **Acceptance:** Tests cover merge order, unknown keys (error naming the key and allowed keys), and type errors.
- **Commit:** `feat(config): add layered configuration and parameter base models`

### U1-10 · Policy rule schema
- **Status:** todo · **Size:** S · **Depends on:** U1-07
- **Goal:** Typed models for `policy.yaml` (`when` matchers, `then` actions `annotate | warn | fail | notify`) and `PolicyOutcome`. There's no evaluator yet (Phase 5).
- **Assumptions to confirm:** The REFACTOR_PLAN §7.2 example covers the matcher operators users need.
- **Acceptance:** The §7.2 example parses. Invalid rules produce precise errors.
- **Commit:** `feat(model): add policy rule schema`

### U1-11 · Schema export, breaking-change gate and TypeScript types
- **Status:** todo · **Size:** M · **Depends on:** U1-08, U1-09, U1-10, U1-03
- **Goal:** `dsbf schema export`. The JSON Schema is committed under `schema/`. CI fails on an unannounced breaking change. TS types are generated into `frontend/src/generated/`.
- **Assumptions to confirm:**
  - The tool choice for TS generation (for example `json-schema-to-typescript`).
  - How a "breaking change" is detected: removed or renamed fields and narrowed types.
- **Acceptance:**
  - Deleting a field in a test branch fails CI.
  - Regenerating the types is a no-op on a clean tree.
  - The Phase 1 exit check (a hand-built `Run` validates, round-trips and exports) passes in a test.
- **Commit:** `feat(model): export JSON Schema and TypeScript types with a breaking-change gate`

---

## Phase 2: Data and profile

- **Capability:** Tier-1 sources (in-memory data and files), a `Dataset` with fingerprint and sampling provenance, and a single profiling pass that replaces per-task rescans.
- **Motivation:** "Measure once, decide many times" (REFACTOR_PLAN §2). It removes the repeated per-column computation and the per-task pandas copies, and fixes type inference cost, listwise deletion and histogram honesty at the source.
- **Boundaries:** No check API and no changes to legacy tasks. The profile is new code, compared against legacy outputs.
- **Sequence and dependencies:** Depends on Phase 1 contracts. Sources, then dataset, then type inference, then column statistics, then dataset-level profile, then parity, then the performance budget.
- **Success criteria:** Profile parity with legacy outputs is documented, and the profile pass runs in under 2 s at 52k × 60 on CI hardware.

### U2-01 · Source adapters for in-memory data and files (tier 1)
- **Status:** todo · **Size:** M · **Depends on:** U1-08
- **Goal:** `dsbf.data.load(source)` accepts pandas, Polars or Arrow objects, or CSV/Parquet/Excel/JSON/NDJSON paths, and returns a Polars-backed `Dataset`. Unreadable input raises a typed error with a hint.
- **Assumptions to confirm:** The Excel reader dependency (`fastexcel` vs `openpyxl`); whether pandas input converts zero-copy for common dtypes.
- **Acceptance:** A table-driven test covers each source type, including the adversarial frame and tz-aware datetimes.
- **Commit:** `feat(data): add tier-1 source adapters`

### U2-02 · Dataset fingerprint, sampling provenance and value cache
- **Status:** todo · **Size:** M · **Depends on:** U2-01 · **Resolves:** F-09
- **Goal:**
  - `Dataset.fingerprint`;
  - `Dataset.sample(...)` with a recorded `SamplingInfo` (head, seeded random, stratified-or-error);
  - `Dataset.values(col, dropna=True)`, which returns cached NumPy arrays within a memory bound.
- **Assumptions to confirm:** The hashing approach and its cost at 50k and 1M rows; the cache memory bound default.
- **Acceptance:** The fingerprint is stable across loads and changes when one cell changes. Sampling is reproducible by seed. Cache eviction is tested.
- **Commit:** `feat(data): add dataset fingerprint, sampling provenance and value cache`

### U2-03 · Sample-based semantic type inference
- **Status:** todo · **Size:** M · **Depends on:** U2-02 · **Resolves:** F-37
- **Goal:** Semantic types are inferred from a sample, with confirmation on the full column only where needed. The results agree with `infer_types` on the golden datasets, or the differences are documented.
- **Context:** F-37. `_is_datetime_like` took 8.6 s of `infer_types`' 9.0 s on kitchen5k and grew 10× at 50k rows.
- **Acceptance:** Parity is checked against U0-13 baselines, and the differences are listed in the PR. Inference time is under 0.5 s on kitchen50k.
- **Commit:** `feat(profile): add sample-based semantic type inference`

### U2-04 · Column profile I: counts and numeric statistics
- **Status:** todo · **Size:** M · **Depends on:** U2-03 · **Resolves:** F-32 (per-column statistics)
- **Goal:** One pass computes counts, nulls, distinct (exact or sketched), zeros, empty strings, non-finite values, and numeric min, max, mean, std, skew, kurtosis, MAD and quantiles, per column on that column's non-null values.
- **Acceptance:** Values match `summarize_numeric`, `summarize_nulls` and `summarize_unique` on the baselines within tolerance.
- **Commit:** `feat(profile): add count and numeric column statistics`

### U2-05 · Column profile II: histograms, top-k, text and datetime
- **Status:** todo · **Size:** M · **Depends on:** U2-04 · **Resolves:** F-12
- **Goal:**
  - equal-width histogram bins, with the rule recorded;
  - top-k values with counts;
  - string length and pattern statistics;
  - datetime range, inferred frequency, gaps and timezone.
- **Assumptions to confirm:** The default binning rule (Freedman–Diaconis with a Sturges fallback) and the top-k default.
- **Acceptance:** Histogram counts sum to `n_non_null`. The adversarial frame profiles without error. Values match `summarize_value_counts` and `categorical_length_stats`.
- **Commit:** `feat(profile): add histograms, top-k, text and datetime profiles`

### U2-06 · Dataset profile and stage inference
- **Status:** todo · **Size:** S · **Depends on:** U2-05 · **Resolves:** F-08
- **Goal:** Shape, a hash-based duplicate-row count, a missingness co-occurrence summary, and stage inference from the profile with one vocabulary shared with check metadata.
- **Acceptance:** Stage inference gives the same answer for pandas and Polars input of the same data (the F-08 regression test).
- **Commit:** `feat(profile): add dataset-level profile and stage inference`

### U2-07 · Profile parity report against legacy outputs
- **Status:** todo · **Size:** M · **Depends on:** U2-06
- **Goal:** An automated comparison between the new profile and the legacy `summarize_*` and `infer_types` outputs on every golden dataset, plus a committed `docs/parity/profile.md` listing accepted differences with reasons.
- **Acceptance:** The parity test passes with only the listed differences.
- **Commit:** `test(profile): add legacy parity comparison and accepted differences`

### U2-08 · Profile performance budget
- **Status:** todo · **Size:** S · **Depends on:** U2-07, U0-14 · **Resolves:** F-30 (profiling part)
- **Goal:** Add the profile pass to the perf harness, with the budget enforced.
- **Acceptance:** Profile pass under 2 s at kitchen50k (52.5k × 60) on CI hardware, and under 2.5× peak memory over the in-memory data size. Or record the actual numbers and amend the budget with reasons.
- **Commit:** `test(perf): enforce profiling pass budget`

---

## Phase 3: Engine and check API

- **Capability:** The check API, registry and packs, the planner, an isolated, parallel and cached executor, library-friendly logging, a legacy adapter running every existing task, and the public `dsbf.profile()` entry point.
- **Motivation:** "Choose your own analysis" and report-only failure handling (REFACTOR_PLAN §1, R1 and R2) live in the engine. The adapter lets checks migrate one at a time against a running system.
- **Boundaries:** No check migrations (Phase 4), and no storage, API or dashboard changes (Phase 6).
- **Sequence and dependencies:** Depends on Phase 2. Interface, registry, planner, executor (isolation, then timeouts and events, then parallelism), cache, logging, adapter, entry point.
- **Success criteria:** Golden runs reproduce through the adapter. A forced exception in any check still yields a complete `Run` with a note. Changing one parameter re-executes only the affected checks.

### U3-01 · Check interface and check context
- **Status:** todo · **Size:** M · **Depends on:** U2-08, U1-09
- **Goal:** The `@check` decorator and metadata (id, version, kind, params, `applies_to`, `requires`, `after`, cost, extras, packs, dimensions). `CheckContext` provides filtered `columns`, `values()`, `profile`, `finding()`, `note()` and `output()`.
- **Assumptions to confirm:** The decorator vs. class-based API (REFACTOR_PLAN §6.2 shows the decorator); naming per ADR D10.
- **Acceptance:** Three toy checks (rule, analysis, synthesis) run in a test harness and produce valid `CheckResult`s. A check never sees columns outside `applies_to`.
- **Commit:** `feat(engine): add check interface and check context`

### U3-02 · Registry, packs and plugin discovery
- **Status:** todo · **Size:** M · **Depends on:** U3-01 · **Resolves:** F-23
- **Goal:** An explicit `Registry` object (no import side effects), the `dsbf.checks` and `dsbf.packs` entry-point groups, local plugin directories imported under unique module names, and reload safety.
- **Acceptance:**
  - A test plugin package installed in a venv is discovered.
  - Two plugin dirs that each contain a `utils.py` coexist.
  - Loading twice in one process is a no-op.
- **Commit:** `feat(engine): add registry, packs and entry-point plugin discovery`

### U3-03 · Planner I: selection and dependency resolution
- **Status:** todo · **Size:** M · **Depends on:** U3-02 · **Resolves:** F-21
- **Goal:** Select checks by packs, include/exclude lists, depth presets, cost budget, dimension and tag. Resolve hard requirements (auto-included and marked) and soft `after` ordering. Report cycles as plan errors.
- **Acceptance:** Table-driven tests cover selection. A soft-after dependency that errors doesn't skip its dependent (the F-21 scenario).
- **Commit:** `feat(engine): add plan selection and dependency resolution`

### U3-04 · Planner II: validation, expected statuses and `dsbf plan`
- **Status:** todo · **Size:** M · **Depends on:** U3-03 · **Resolves:** F-07, F-22
- **Goal:**
  - Validate parameters against each check's model, with precise errors.
  - Annotate the plan with `will_run`, `not_applicable` or `unavailable` (missing extra), with reasons.
  - Print a readable plan via a new `dsbf plan` CLI command.
- **Acceptance:**
  - A misspelled parameter is an error naming the check, key and allowed keys (the F-07 class).
  - A missing extra shows as `unavailable` with an install hint.
  - `dsbf plan data.csv` output has a snapshot test.
- **Commit:** `feat(engine): validate plans and add dsbf plan`

### U3-05 · Executor I: isolation, statuses and notes
- **Status:** todo · **Size:** M · **Depends on:** U3-04 · **Resolves:** F-06, F-20 (engine side)
- **Goal:** Sequential execution where every exception becomes `error` with a note (type, message, hint, traceback artifact), hard dependents become `skipped` with the blocker named, and the run always completes with `RunHealth`.
- **Acceptance:** A forced exception in each toy check yields a complete `Run`. The health summary lists the reason. A check returning a failed result is never counted as ok (the F-06 regression test).
- **Commit:** `feat(engine): execute checks in isolation with explicit statuses`

### U3-06 · Executor II: timeouts, resource accounting and events
- **Status:** todo · **Size:** M · **Depends on:** U3-05 · **Resolves:** F-24
- **Goal:** Per-check timeouts (→ `timeout` plus a note); an optional run budget; wall, CPU and sampled peak memory per check, labelled approximate; and an event hook API (`check_started`, `check_finished`, `note`).
- **Assumptions to confirm:** How timeouts work for thread-based checks (cooperative cancellation vs. process isolation for heavy checks).
- **Acceptance:** A sleeping toy check times out without stalling the run. Events arrive in order. Memory figures are within reason of `/usr/bin/time` on a perf dataset.
- **Commit:** `feat(engine): add timeouts, resource accounting and execution events`

### U3-07 · Executor III: parallel execution
- **Status:** todo · **Size:** M · **Depends on:** U3-06
- **Goal:** Independent checks at each DAG level run concurrently (threads for compiled-code checks, a process pool for pure-Python ones). Results are deterministic regardless of worker count.
- **Acceptance:** The same `Run` (excluding timing) with 1 worker and N workers. A speed-up is measured on a perf dataset.
- **Commit:** `feat(engine): run independent checks in parallel`

### U3-08 · Result cache
- **Status:** todo · **Size:** M · **Depends on:** U3-07
- **Goal:** Results cached by `(fingerprint, check id, check version, params hash, sampling)` on local disk under `DSBF_HOME`, with invalidation of dependents.
- **Acceptance:** Changing one parameter re-executes only that check and its dependents (a test asserts the execution counts). Bumping a check version invalidates its entries.
- **Commit:** `feat(engine): cache check results by data fingerprint and parameters`

### U3-09 · Library-friendly logging
- **Status:** todo · **Size:** S · **Depends on:** U3-05 · **Resolves:** F-19
- **Goal:**
  - a `dsbf` logger hierarchy with no global logging changes;
  - user data never interpreted as markup;
  - an `error` level;
  - the CLI attaches handlers;
  - per-run file logs, with handlers closed afterwards.
- **Acceptance:** Importing `dsbf` leaves `logging.getLoggerClass()` unchanged. Logging a column named `[/x]` produces no logging error. Task-level messages appear in the run log file.
- **Commit:** `refactor(logging): make logging library-friendly and markup-safe`

### U3-10 · Legacy task adapter
- **Status:** todo · **Size:** M · **Depends on:** U3-08, U3-09
- **Goal:** `LegacyTaskCheck` wraps any `BaseTask`. It provides a legacy `AnalysisContext` shim (data, config, `semantic_types` from the new profile) and maps `TaskResult` statuses onto the new statuses.
- **Assumptions to confirm:** Which `context.metadata` keys legacy tasks read from each other (grep `get_metadata`, `context.results`).
- **Acceptance:** Five representative legacy tasks (a rule-like task, a pandas-heavy one, a plotting one, a synthesis scorer, and a time-series one) run through the new engine with their outputs unchanged.
- **Commit:** `feat(engine): add legacy task adapter`

### U3-11 · All legacy tasks through the new engine with findings
- **Status:** todo · **Size:** M · **Depends on:** U3-10
- **Goal:** Every registered legacy task runs on the new engine. Guidance blurbs convert into `Finding`s with evidence where it can be derived.
- **Acceptance:** Golden runs (U0-13) reproduce through the adapter, with differences listed in the PR. The adversarial frame produces a complete `Run` with no unhandled errors.
- **Commit:** `feat(engine): run all legacy tasks through the new engine`

### U3-12 · `dsbf.profile()` entry point and run health output
- **Status:** todo · **Size:** M · **Depends on:** U3-11 · **Resolves:** F-16, F-17, F-18, F-34
- **Goal:** The public `dsbf.profile(source, packs=…, checks=…, target=…, time_index=…, sample=…)` returns an in-memory `Run` with no filesystem writes. Version comes from `importlib.metadata`. `run.health.summary()` matches REFACTOR_PLAN §7.1.
- **Acceptance:**
  - A test asserts no files are created during `dsbf.profile()`.
  - The README library example works.
  - Phase 3 exit checks pass: golden via the adapter, forced exception gives a complete `Run`, a parameter change re-executes only the affected checks.
- **Commit:** `feat: add dsbf.profile() library entry point`

---

## Phase 4: Migrate checks (placeholder)

- **U4-00 · Plan Phase 4** (`todo`, depends on U3-12): break down the batches below into full units using Phase 3 learnings and golden-run evidence. Update REFACTOR_PLAN if needed.
- U4-A · Rule checks onto the profile, about 20 tasks (`placeholder`). From U0-03: the migrated guards only check that some column matches; tasks still loop over every numeric or object dtype column, not `matched_cols`, so they also analyse columns whose semantic type was excluded. And a migrated task run without `infer_types` now skips instead of analysing all numeric columns. Decide the intended behaviour per check
- U4-B · Analysis checks with statistical fixes: F-11, F-13, confidence intervals, bias-corrected Cramér's V (`placeholder`)
- U4-C · Heavy checks redesigned with budgets: fuzzy duplicates (F-35), time series (F-39), drift against reference profiles (`placeholder`)
- U4-D · Synthesis: quality dimensions, ML readiness, target-based leakage (F-26) (`placeholder`)
- U4-E · Remove the legacy adapter, backend plotting and plotting dependencies (N-A2, F-38) (`placeholder`)

## Phase 5: Policy, sinks, CLI v2. First release for embedded use (placeholder)

- **U5-00 · Plan Phase 5** (`todo`, depends on the Phase 4 units)
- Placeholders: policy evaluator (report-only default) · JSON/NDJSON sinks · Parquet sinks with a consumer schema guide · static HTML report · CLI v2 (`profile`, `check`, `plan`, `schema export`, run health summary) · pipeline and consumer docs · install smoke test from a temp dir · release `1.0.0a1`

## Phase 6: Store, service and dashboard v2. Standalone local (placeholder)

- **U6-00 · Plan Phase 6** (`todo`, depends on the Phase 5 units)
- Placeholders: run repository on SQLite with Alembic · artifact store and per-run data snapshot · `dsbf runs …` · API v2 · local job runner · generated TS client · dashboard shell and run health · findings explorer · column view (reuse `wip/echarts-charts` components with real histogram bins) · relationships · quality · ML readiness · time series · run compare and history · settings → rerun · frontend component and end-to-end tests (F-29) · bundle the dashboard into the wheel with `dsbf dashboard` · remove API v1 and the old store (F-15, F-25) · release `1.0.0`

## Phase 7: Scale and sources (placeholder)

- **U7-00 · Plan Phase 7**
- Placeholders: larger-than-memory files and object storage (T2) · Ibis sources with DuckDB/Postgres, then Snowflake/BigQuery (T3) · recurring datasets, history and profile-based drift (T4) · scheduling recipes

## Phase 8: Team and cloud (placeholder)

- **U8-00 · Plan Phase 8**
- Placeholders: Postgres repository · object-storage artifacts · auth (token, OIDC) · job queue · Docker image and Helm chart · GitHub Action · Airflow, Prefect and Dagster adapters · dbt recipe · notifiers and OpenLineage · MCP server

## Phase 9: Modalities and workflow phases (placeholder)

- **U9-00 · Plan Phase 9**
- Placeholders: text-corpus modality · domain packs · modelling-prep exports (ML guidance actions as sklearn pipelines) · modelling diagnostics
