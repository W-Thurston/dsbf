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
poetry install
poetry run pre-commit install
```

Run the test suite to confirm everything is working:

```bash
poetry run pytest
poetry run python tests/validation/run_validation.py
```

---

## Project structure

```
dsbf/
├── core/               # BaseTask, TaskResult, execution graph
├── eda/
│   ├── tasks/          # All EDA task implementations (one file per task)
│   └── task_registry.py
├── utils/              # Shared utilities (backend, reco_engine, etc.)
├── config/
│   ├── default_config.yaml
│   └── task_metadata.yaml
└── outputs/            # Generated at runtime — gitignored

tests/
├── generators/         # Dataset generators for the validation suite
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
ruff check .
ruff format .
```

All Python files must have:
- Module-level docstring explaining the file's purpose
- Docstrings on every public class and method
- Type annotations on all function signatures

**Vue / JavaScript:** follow the patterns established in the existing components.
Tailwind utility classes where applicable; custom CSS in scoped `<style>` blocks.

---

## Writing a new task

Every task is a Python class in `dsbf/eda/tasks/`. Here is the minimal template:

```python
# dsbf/eda/tasks/my_new_task.py

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result


@register_task(
    display_name="My New Task",
    description="One sentence describing what this task detects.",
    depends_on=["infer_types"],          # tasks that must run first
    profiling_depth="standard",          # basic | standard | full
    stage="eda",                         # eda | modeling
    domain="core",
    runtime_estimate="fast",             # fast | medium | slow
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
`dsbf/config/task_metadata.yaml` so it appears in the task registry.

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
   ruff check .
   poetry run pytest
   poetry run python tests/validation/run_validation.py
   ```

4. **PR description** — explain what changed, why, and how to verify it. Link to
   any relevant issues.

5. **Checklist before merge:**
   - [ ] Tests pass (`pytest`)
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
