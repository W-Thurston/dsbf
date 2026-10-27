# Contributing to DSBF

Thanks for your interest in contributing. This document covers everything you need
to know to work effectively in this codebase — from environment setup through writing
a new check, adding a validation dataset, and submitting a pull request.

---

## Table of contents

1. [Development setup](#development-setup)
2. [Project structure](#project-structure)
3. [Code style](#code-style)
4. [Writing a new check](#writing-a-new-check)
5. [Adding a validation dataset](#adding-a-validation-dataset)
6. [Commit messages](#commit-messages)
7. [Pull request process](#pull-request-process)
8. [Known gotchas](#known-gotchas)

---

## Development setup

```bash
git clone https://github.com/W-Thurston/dsbf.git
cd dsbf
uv sync
uv run pre-commit install
```

Run the fast gate to confirm everything is working. It takes about 30 seconds:

```bash
uv run pytest -m "unit and not slow"
```

`uv sync` installs the dev tools and the `ml`, `timeseries`, `text`, `excel` and `pandas`
extras, since the tests exercise every check. `uv sync --no-dev` is the base install, which is
what CI's `extras` job installs, with each extra added alone, to test that the core runs without
the rest.

The full suite adds the integration tests, which run `dsbf.profile()` end to end. It takes
a few minutes:

```bash
uv run pytest
```

### Test tiers

Every test carries exactly one tier marker, and collection fails for a test without
one. Set it for a whole module with `pytestmark = pytest.mark.unit`.

| Marker | Meaning |
|---|---|
| `unit` | Fast and isolated: no network and no full engine run. |
| `integration` | Exercises several components together, such as `dsbf.profile()` on a frame or a file. |
| `golden` | Profiles a dataset end to end and compares it with a committed snapshot. Deselected by default; see [Golden runs](#golden-runs). |
| `perf` | Times full runs and compares them with a committed baseline. Deselected by default; see [Performance runs](#performance-runs). |
| `calibration` | Measures each check's detection and false-alarm rates on planted-truth frames against a committed baseline. Deselected by default; see [Calibration runs](#calibration-runs). |

Two modifiers can be added to a test: `slow` (more than 5 s on a developer machine) and
`network` (needs network access). The fast gate is `unit and not slow`. Select any
tier with `-m`, for example `uv run pytest -m "integration and not network"`.

### Golden runs

`tests/golden/` holds snapshots of what DSBF reports on ten datasets (the eight validation
archetypes, titanic and a 5,000-row kitchen-sink frame). They take about ten minutes, so
`pytest` skips them. Run them with `uv run pytest -m golden`, and add the `golden` label to
a pull request to run them in CI. If a change to a check or the engine moves a snapshot,
review the diff and accept it with `uv run pytest -m golden --update-golden`. The review
steps are in [`tests/golden/README.md`](tests/golden/README.md).

The `dsbf plan` output snapshot in `tests/test_interfaces/snapshots/` is a unit test, but
its update option is the same: `uv run pytest tests/test_interfaces/test_cli_plan.py --update-golden`.

### Performance runs

`tests/perf/` times the profile pass and `dsbf.profile()` on titanic and on the kitchen-sink
frame at 5,000 and 50,000 rows, and fails when a stage, a check or the peak memory grows by
more than 25% against `tests/perf/baseline.json`, or when the standard plan takes over 5 s on
titanic or over 60 s and 1 GB on kitchen50k. The suite
takes about fifteen minutes, so `pytest` skips it. Run it with `uv run pytest -m perf`, and add
the `perf` label to a pull request to run it in CI, where the baseline was recorded. Local
numbers from another machine are only roughly comparable. How to read the results and update
the baseline is in [`tests/perf/README.md`](tests/perf/README.md).

### Calibration runs

`tests/calibration/` generates frames with a known truth (a planted leak, outliers,
dependent nulls, a typo'd duplicate, nothing at all) from a seed, profiles each over many
seeds, and records how often every check finds what was planted and how often it raises
an alarm nobody planted. A check whose rate got worse than `tests/calibration/baseline.json`
fails its scenario. The suite takes several minutes, so `pytest` skips it. Run it with
`uv run pytest -m calibration`, and add the `calibration` label to a pull request that
changes a threshold, a method or a check's version; CI runs it nightly and uploads the
rates. A change that moves a rate on purpose is accepted with
`uv run pytest -m calibration --update-calibration-baseline`, and the new check version
goes in the same commit. How to add a scenario and read the results is in
[`tests/calibration/README.md`](tests/calibration/README.md).

### Test isolation

The suite never touches your own runs. `tests/conftest.py` points `DSBF_HOME` at a
temporary root and runs the whole session from a temporary working directory. Outputs
land under that root and are deleted afterwards, and a stray write relative to the
working directory lands in the temporary directory. The variable is forced, so a value
you export is ignored while tests run.
Write tests that don't assume the working directory is the repository root: build paths
from `Path(dsbf.__file__)` or `tmp_path`.

---

## Project structure

```
src/dsbf/
├── checks/             # The core pack: one module per check, and what they share
├── config/             # ProfileConfig and the settings loader
├── data/               # Loading, sampling and fingerprinting a Dataset
├── engine/             # Check contract, registry, planner, executor, cache, tips
├── interfaces/         # The dsbf command and the schema export
├── model/              # The Run and every model inside it (published as JSON Schema)
├── profiling/          # The dataset profile that checks read
├── static_metadata/    # The recommendation tip library
├── utils/              # Logging and the version
├── library.py          # dsbf.profile()
└── paths.py            # The DSBF home and the output directory

frontend/               # Vue 3 + Vite dashboard source (not wired to an API; see ROADMAP.md)

tests/
├── calibration/        # Planted-truth scenarios, rates per check and their baseline
├── generators/         # Dataset generators for the validation suite
├── golden/             # Core snapshots on ten datasets and the code that compares them
├── perf/               # Performance benchmark and its committed baseline
├── test_checks/        # One module per check or family of checks
├── test_engine/        # The engine, registry, planner, executor and cache
├── validation/         # The eight archetype datasets against what each should show
└── ...                 # data, model, profile, config, library and interface tests
```

---

## Code style

**Python:** Ruff for linting, Pyrefly for type checking. Max line length is **88 characters**.

```bash
uv run ruff check .
uv run ruff format .
uv run pyrefly check --baseline .pyrefly-baseline.json --error-stale-baseline
```

Pyrefly is pinned in the dev group and configured under `[tool.pyrefly]` in
`pyproject.toml`. Errors that predate the checker are listed in
`.pyrefly-baseline.json`, which is generated: don't edit it by hand. A new error fails
the check, and so does a baseline entry that no longer matches, so the file only shrinks.
After you fix baselined errors, drop their entries:

```bash
uv run pyrefly check --baseline .pyrefly-baseline.json --prune-baseline
```

`--update-baseline` rewrites the file with every current error, so use it only when a
dependency upgrade changes legacy errors that you are not fixing, and say so in the PR.
The baseline matches on file, column, error kind and message, not line number, so an
identical error added to a file that already has it isn't reported.

All Python files must have:
- Module-level docstring explaining the file's purpose
- Docstrings on every public class and method
- Type annotations on all function signatures

**Vue / JavaScript:** ESLint (with `eslint-plugin-vue`) lints and Prettier formats. Prettier
owns layout (no semicolons, single quotes, 100 columns), so don't hand-align code. Follow the
patterns established in the existing components, and put custom CSS in scoped `<style>`
blocks. The frontend uses pnpm, pinned through the `packageManager` field in
`frontend/package.json`; `corepack enable` once and the right version is fetched.

```bash
cd frontend
pnpm install
pnpm lint        # ESLint, then a Prettier check
pnpm format      # rewrite files with Prettier
pnpm build
pnpm test:run    # Vitest; `pnpm test` watches
```

CI runs the same three commands in the Frontend job. A few existing ESLint findings
(unused variables, a computed property with a side effect, props without defaults) are
warnings rather than errors until they are fixed.


### Result schema

The result models in `dsbf.model` are published as `schema/run.schema.json`, with
TypeScript types in `frontend/src/generated/run.d.ts`. Both are generated, so change the
models and regenerate; never edit them by hand:

```bash
uv run dsbf schema export --types frontend/src/generated/run.d.ts
uv run dsbf schema check --base origin/main   # is the change announced by its version?
```

The Contracts job in CI fails if either file is out of date (`schema export ... --check`)
or if the schema changed against the pull request's base branch without raising
`SCHEMA_VERSION` in `src/dsbf/model/run.py`. Anything that can make a stored run invalid,
or give a reader a value it wasn't told about (a removed field, a changed type, a new enum
member, a new field without a default), is breaking and needs a higher minor version while
the schema is 0.x and a higher major version from 1.0. A new field with a default needs a
higher minor version.

---

## Writing a new check

Every check is a function in its own module under `src/dsbf/checks/`, declared with
`@check` at the top level so a worker process can import it by name. Add it to
`core_pack()` in `src/dsbf/checks/__init__.py`. A minimal check:

```python
# src/dsbf/checks/negative_values.py
"""Flag numeric columns that hold a negative value."""

from dsbf.engine import CheckContext, CheckOutput, Requires, SemanticTypes, check
from dsbf.model import (
    CheckParams,
    Confidence,
    Cost,
    Dimension,
    Evidence,
    NumericProfile,
    Severity,
    Subject,
)


@check(
    id="negative_values",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes("numeric"),
    dimensions=[Dimension.VALIDITY],
    requires=Requires(profile=["counts"]),
    cost=Cost.CHEAP,
    packs=["core"],
)
def negative_values(ctx: CheckContext, params: CheckParams) -> CheckOutput | None:
    """Flag numeric columns whose smallest value is below zero."""
    for column in ctx.profile.columns:
        if not isinstance(column, NumericProfile) or column.stats is None:
            continue
        lowest = column.stats.min
        if lowest is not None and lowest < 0:
            ctx.finding(
                code="negative_value",
                subject=Subject.column(column.name),
                severity=Severity.INFO,
                confidence=Confidence.HIGH,
                title=f"{column.name!r} has negative values",
                message=f"The smallest value of {column.name!r} is {lowest}.",
                evidence=Evidence(
                    metrics={"min": lowest},
                    n_used=column.counts.non_null,
                    n_total=ctx.profile.n_rows,
                    method="minimum",
                ),
            )
```

The module docstring of `src/dsbf/checks/__init__.py` lists the conventions every core
check follows. The ones that matter most:

**Let `applies_to` choose the columns** — read them through `ctx.columns`, the profile
and `ctx.values`; never pick columns by dtype. A boolean stored as `int64` then doesn't
enter outlier or VIF computation.

**Declare what the check needs** — `Requires` names the profile parts and the checks it
reads. A check that reads another's result without declaring it runs before that result
exists.

**Handle no eligible columns** — the planner marks a check `not applicable` when no
column matches, so the check body can assume one. The all-categorical and single-column
validation datasets test that this holds.

**State `n_used` and the method on every finding**, bound every parameter, and put
advice in `src/dsbf/static_metadata/recommendation_library.yaml` as a tip rather than in
the check.

**Import an extra's package inside the function that uses it.** The core pack loads on the
base install, which holds only NumPy, SciPy, Polars, PyArrow and the tools the engine needs.
A check that needs scikit-learn, statsmodels or rapidfuzz names the extra in `@check(extras=[...])`,
imports it at the top of the function that calls it (`from sklearn.mixture import
GaussianMixture  # the `ml` extra`), and the engine reports it `unavailable` with an install
hint when the package is missing. Add the module to `PRELOAD` in `src/dsbf/engine/extras.py`:
the executor imports those modules before the checks run side by side, because a first import
on a worker thread can hang. `tests/test_packaging.py` fails when an import is at module level
without a base dependency, when a package is declared nowhere, or when `PRELOAD` and the
imports disagree. A new dependency goes in the base only if `dsbf` imports it when it loads.

**Bump `version`** when a change alters what the check reports; the cache and the
calibration baseline key on it.

**Write tests** — add `tests/test_checks/test_negative_values.py` with at least the
happy path, the no-flag case and a numerical edge case. `tests/test_checks/support.py`
has `run_core_check(frame, "negative_values")` for a single check. A check with a
detection or false-alarm rate to protect also gets a scenario in `tests/calibration`.

A check that is not part of DSBF lives in a directory of your own and is loaded with
`plugins=[...]` or `dsbf profile --plugin`; see the README.

---

## Adding a validation dataset

The validation suite lives in `tests/validation/`. It profiles each archetype in memory
with `dsbf.profile()` and states what the core checks must say about it. Each dataset has
two parts:

**1. A generator function** in `tests/generators/dsbf_test_dataset_generator.py`:

```python
def generate_my_dataset(n_rows: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Docstring explaining: what the dataset tests, column inventory,
    expected findings, and what is deliberately absent.
    """
    rng = np.random.default_rng(seed)
    # ...
    return pd.DataFrame({...})
```

Add it to the `GENERATORS` dict at the bottom of the file.

**2. A test** in `tests/validation/test_archetypes.py` that takes the `archetype` fixture
(a `Run`, profiled once per session) and reads it with the helpers in
`tests/validation/support.py`:

```python
def test_my_dataset(archetype: Archetype) -> None:
    run = archetype("my_dataset")
    assert flagged(run, "skewness", "skewed", Severity.WARN) == {"purchase_amount"}
    assert quality(run, "completeness")["level"] == "amber"
```

Add the name to `ARCHETYPES` so that "no check fails" covers it, and to
`tests/validation/DASHBOARD_CHECKLIST.md` with what to look at once the dashboard is rebuilt.

**Assertion writing principles:**
- Check both directions where relevant (skewed columns flagged AND clean columns not flagged)
- Use ranges, not exact values, for seeded floats (±5pp is typical)
- Read the `Run` of the real dataset before writing the assertion: a check's output is
  not always what its name suggests
- A rate across many seeds belongs in `tests/calibration`, not here

---

## Commit messages

DSBF uses [Conventional Commits](https://www.conventionalcommits.org/). The format is:

```
<type>(<scope>): <short description>

<optional body>
```

**Types:**

| Type | When to use |
|---|---|
| `feat` | New check, new command, new setting |
| `fix` | Bug fix — include what was wrong and what changed |
| `refactor` | Code restructuring with no behaviour change |
| `test` | New or updated tests / validation datasets |
| `docs` | README, docstrings, guides |
| `chore` | Dependency updates, config changes, CI |
| `style` | Formatting, linting — no logic change |

**Scope** (optional but helpful): `checks`, `engine`, `cli`, `model`, `config`, `test`

**Examples:**

```
feat(checks): add a bimodality check with Ashman's D

fix(checks): collinearity — centre the columns before the VIF

Without centring, features with nonzero means produce spuriously high VIF
(10–35) even when pairwise correlations are near zero.

test(validation): add wide dataset (100 columns) — passes first try

chore: bump ruff to 0.4.0, update pre-commit hooks
```

Keep the subject line under 72 characters. Use the body to explain *why*, not *what* —
the diff shows what changed.

---

## Pull request process

1. **Branch from `main`** — use a descriptive branch name:
   `feat/bimodality`, `fix/vif-centring`, `test/wide-dataset`

2. **Keep PRs focused** — one logical change per PR. A PR that adds a new check,
   refactors the engine, and changes the CLI is three PRs.

3. **Before opening the PR:**
   ```bash
   uv run ruff check .
   uv run pytest -m "unit and not slow"
   uv run pytest
   ```

4. **PR description** — explain what changed, why, and how to verify it. Link to
   any relevant issues.

5. **Checklist before merge:**
   - [ ] Tests pass (`pytest -m "unit and not slow"` and `pytest`), and new tests carry a tier marker
   - [ ] Ruff clean
   - [ ] Docstrings on new public classes and methods
   - [ ] The check is in `core_pack()` and its version bumped (if adding or changing one)
   - [ ] CHANGELOG entry added (user-visible changes only, under the unreleased version)

---

## Known gotchas

**F541 — bare f-strings:** Continuation strings in multi-line expressions where
the placeholder `{}` is on an adjacent line get flagged by Ruff if any segment
lacks a placeholder. Extract the problematic segment to a plain string literal
(drop the `f` prefix) or use a named variable.

**E201/E202 — whitespace inside braces:** The pattern `f"...{ {k: v for ...} }..."`
(a dict comprehension inside an f-string) triggers E201/E202. Always extract dict
comprehensions to a named variable before embedding in an f-string:
```python
result_str = {k: round(v, 1) for k, v in scores.items()}
message = f"scores: {result_str}"
```

**Report the evidence a finding rests on:** `Evidence` needs `n_used`, `n_total` and the
`method`. A finding without them fails when it is created, and the check ends in
`error`; read the note on the result.

**Do not catch warnings inside a check:** the engine records a warning a check raises as
a `python_warning` note with a count. `warnings.catch_warnings()` would also hide the
warnings of the checks running beside it.

**Fit scikit-learn models on float64:** `GaussianMixture` mis-scores integer input, and
differently in 1.7 and 1.8.

**A VIF needs centred columns:** `collinearity` takes the diagonal of the inverse
correlation matrix, so it needs no intercept. A VIF computed from uncentred columns is
spuriously high wherever the means are far from zero.
