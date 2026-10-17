# Contributing to DSBF

Thanks for your interest in contributing. This document covers everything you need
to know to work effectively in this codebase — from environment setup through writing
a new task, adding a validation dataset, and submitting a pull request.

---

## Table of contents

1. [Development setup](#development-setup)
2. [Project structure](#project-structure)
3. [Code style](#code-style)
4. [Writing a new task](#writing-a-new-task)
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

The full suite adds the engine and API tests, and needs network access (the engine
tests load seaborn's titanic dataset). It takes about 5 minutes:

```bash
uv run pytest
uv run python tests/validation/run_validation.py
```

### Test tiers

Every test carries exactly one tier marker, and collection fails for a test without
one. Set it for a whole module with `pytestmark = pytest.mark.unit`.

| Marker | Meaning |
|---|---|
| `unit` | Fast and isolated: no network and no full engine run. |
| `integration` | Exercises several components together, such as the engine or the API. |
| `golden` | Profiles a dataset end to end and compares it with a committed snapshot. Deselected by default; see [Golden runs](#golden-runs). |
| `perf` | Times full runs and compares them with a committed baseline. Deselected by default; see [Performance runs](#performance-runs). |

Two modifiers can be added to a test: `slow` (more than 5 s on a developer machine) and
`network` (needs network access). The fast gate is `unit and not slow`. Select any
tier with `-m`, for example `uv run pytest -m "integration and not network"`.

### Golden runs

`tests/golden/` holds snapshots of what DSBF reports on ten datasets (the eight validation
archetypes, titanic and a 5,000-row kitchen-sink frame). They take about ten minutes, so
`pytest` skips them. Run them with `uv run pytest -m golden`, and add the `golden` label to
a pull request to run them in CI. If a change to a task or the engine moves a snapshot,
review the diff and accept it with `uv run pytest -m golden --update-golden`. The review
steps are in [`tests/golden/README.md`](tests/golden/README.md).

### Performance runs

`tests/perf/` times a full run on four datasets (titanic, the adversarial frame, and the
kitchen-sink frame at 5,000 and 50,000 rows) and fails when a stage, a task, the peak memory
or a dashboard payload grows by more than 25% against `tests/perf/baseline.json`. The suite
takes about fifteen minutes, so `pytest` skips it. Run it with `uv run pytest -m perf`, and add
the `perf` label to a pull request to run it in CI, where the baseline was recorded. Local
numbers from another machine are only roughly comparable. How to read the results and update
the baseline is in [`tests/perf/README.md`](tests/perf/README.md).

### Test isolation

The suite never touches your own runs. `tests/conftest.py` points `DSBF_DB_PATH` at a
temporary database, turns off the metadata export (`DSBF_AUTO_EXPORT_METADATA=0`), and
runs the whole session from a temporary working directory. Anything DSBF writes relative
to the working directory (`dsbf/outputs/`, `dsbf_run.json`) lands there and is deleted
afterwards. Both variables are forced, so values you export are ignored while tests run.
Write tests that don't assume the working directory is the repository root: build paths
from `Path(dsbf.__file__)` or `tmp_path`.

---

## Project structure

```
src/dsbf/
├── core/               # BaseTask, TaskResult, execution graph
├── eda/
│   ├── tasks/          # All EDA task implementations (one file per task)
│   └── task_registry.py
├── utils/              # Shared utilities (backend, reco_engine, etc.)
├── config/
│   ├── default_config.yaml
│   └── task_metadata.yaml
└── dashboard/frontend/ # Vue 3 + Vite app

dsbf/outputs/           # Generated at runtime — gitignored

tests/
├── generators/         # Dataset generators for the validation suite
├── golden/             # Golden-run snapshots and the code that compares them
├── perf/               # Performance benchmark and its committed baseline
├── validation/
│   ├── assertions/     # Per-dataset assertion modules
│   └── run_validation.py
└── unit/               # Pytest unit tests

dashboard/
├── src/
│   ├── views/          # Tab components (OverviewTab, QualityTab, etc.)
│   └── components/     # Shared components
└── ...
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

**Vue / JavaScript:** follow the patterns established in the existing components.
Tailwind utility classes where applicable; custom CSS in scoped `<style>` blocks.

---

## Writing a new task

Every task is a Python class in `src/dsbf/eda/tasks/`. Here is the minimal template:

```python
# src/dsbf/eda/tasks/my_new_task.py

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result


@register_task(
    display_name="My New Task",
    description="One sentence describing what this task detects.",
    depends_on=["infer_types"],  # tasks that must run first
    profiling_depth="standard",  # basic | standard | full
    stage="eda",  # eda | modeling
    domain="core",
    runtime_estimate="fast",  # fast | medium | slow
    expected_semantic_types=["continuous"],  # filters matched_cols
)
class MyNewTask(BaseTask):
    """
    Detailed docstring explaining what this task computes,
    what the output data structure looks like, and any known
    edge cases or limitations.
    """

    def run(self) -> None:
        try:
            df = self.input_data
            matched_cols, excluded = self.get_columns_by_intent()

            # Guard: return early if nothing to process
            if not matched_cols:
                self.output = TaskResult(
                    name=self.name,
                    status="success",
                    summary={"message": "No eligible columns found."},
                    data={},
                    metadata={},
                )
                return

            # ... your logic here ...

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={"message": "..."},
                data={},
                metadata={},
            )

        except Exception as e:
            if self.context:
                raise
            self.output = make_failure_result(self.name, e)
```

### Key conventions

**Use `get_columns_by_intent()`** — never iterate `df.columns` or call
`df.select_dtypes()` directly. This applies the semantic type filter so that, for
example, a boolean stored as `int64` doesn't enter outlier or VIF computation.

**Declare `depends_on` correctly** — missing dependencies cause tasks to run before
their source data is available. This is a load-bearing declaration.

**Return gracefully when there are no eligible columns** — every task that filters
by semantic type must handle the zero-column case with `status="success"` and empty
output, not an error. The all-categorical and single-column validation datasets
specifically test this.

**Register in `task_metadata.yaml`** — after writing the task, add an entry to
`src/dsbf/config/task_metadata.yaml` so it appears in the task registry.

**Write a unit test** — add `tests/unit/test_my_new_task.py` covering at minimum:
the happy path, the zero-eligible-columns case, and any numerical edge cases.

---

## Adding a validation dataset

The validation suite lives in `tests/validation/`. Each dataset has three parts:

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

**2. An assertion module** in `tests/validation/assertions/my_dataset.py`:

```python
def validate(report: dict) -> None:
    _no_unexpected_failures(report)
    _core_tasks_succeeded(report)
    # ... specific assertions ...
    print("✓  my_dataset: all assertions passed")
```

Assertions should read the actual task output structure from `report.json` before
writing — the data structures are not always what you'd expect. Dry-run against a
real report before committing.

**3. Registration** in `tests/validation/run_validation.py` — add to
`ASSERTION_MODULES` and the manual dashboard checklist.

**Assertion writing principles:**
- Check both directions where relevant (e.g. skewed columns flagged AND clean columns not flagged)
- Use ranges not exact values for seeded floats (±5pp is typical)
- Never use `{ {comprehension} }` inside f-strings — extract to a named variable first (Ruff E201/E202)
- The `detect_high_cardinality` task returns `{col: count}` — flagged columns are dict keys, not a list

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
| `feat` | New task, new dashboard feature, new API endpoint |
| `fix` | Bug fix — include what was wrong and what changed |
| `refactor` | Code restructuring with no behaviour change |
| `test` | New or updated tests / validation datasets |
| `docs` | README, docstrings, CHANGELOG |
| `chore` | Dependency updates, config changes, CI |
| `style` | Formatting, linting — no logic change |

**Scope** (optional but helpful): `task`, `engine`, `dashboard`, `api`, `config`, `test`

**Examples:**

```
feat(task): add detect_bimodal_distribution with Hartigan's dip test

fix(engine): detect_collinear_features — add add_constant to VIF regression

Without an intercept term, features with nonzero means produce spuriously
high VIF (10–35) even when pairwise correlations are near zero.

test(validation): add wide dataset (100 columns) — passes first try

chore: bump ruff to 0.4.0, update pre-commit hooks
```

Keep the subject line under 72 characters. Use the body to explain *why*, not *what* —
the diff shows what changed.

---

## Pull request process

1. **Branch from `main`** — use a descriptive branch name:
   `feat/detect-bimodal`, `fix/vif-add-constant`, `test/wide-dataset`

2. **Keep PRs focused** — one logical change per PR. A PR that adds a new task,
   refactors the engine, and updates the dashboard is three PRs.

3. **Before opening the PR:**
   ```bash
   uv run ruff check .
   uv run pytest -m "unit and not slow"
   uv run pytest
   uv run python tests/validation/run_validation.py
   ```

4. **PR description** — explain what changed, why, and how to verify it. Link to
   any relevant issues.

5. **Checklist before merge:**
   - [ ] Tests pass (`pytest -m "unit and not slow"` and `pytest`), and new tests carry a tier marker
   - [ ] Validation suite passes (`run_validation.py`)
   - [ ] Ruff clean
   - [ ] Docstrings on new public classes and methods
   - [ ] `task_metadata.yaml` updated (if adding a task)
   - [ ] CHANGELOG entry added

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

**`depends_on` is load-bearing:** Missing a dependency declaration causes the scorer
or downstream task to run before its source data exists. If a task's output is
unexpectedly empty, check `depends_on` first.

**`data_quality_scorer._level()` thresholds:** The quality scorer uses a four-state
system (green / blue / amber / red). `info`-only findings below 10% of columns
produce `blue`. Any `warn`-or-above finding produces minimum `amber` regardless of
proportion. Validation assertions for quality dimensions must account for this.

**`detect_high_cardinality` output structure:** Returns `{col: cardinality_count}`
where the presence of a column name as a key means it was flagged. It does not
return `{"high_cardinality_columns": [...]}`. Several other tasks follow the same
`{col: value}` pattern — always inspect the actual report output before writing
assertions.

**Semantic type filtering:** Tasks must use `get_columns_by_intent()` and operate
only on `matched_cols`, never `df.columns` or `df.select_dtypes()`. The
`detect_high_cardinality` and `detect_id_columns` bugs both stemmed from iterating
`df.columns` directly, bypassing the semantic type filter entirely.

**VIF requires `add_constant`:** `variance_inflation_factor` from statsmodels
regresses each feature against all others without an intercept by default. Without
`add_constant`, features with nonzero means produce spuriously high VIF regardless
of actual collinearity. Always use:
```python
from statsmodels.tools import add_constant

numeric_df_with_const = add_constant(numeric_df, has_constant="add")
```
