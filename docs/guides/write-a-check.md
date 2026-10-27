# Write a check

A check is a function that reads a dataset's profile and values and raises findings. You can add one
without changing DSBF: save it in a directory, and load that directory as a plugin. The examples
here use the customer table `data.csv` from [Use DSBF in your pipeline](pipeline.md).

## The smallest check

```python
from pathlib import Path

Path("my_checks").mkdir(exist_ok=True)
Path("my_checks/negative_values.py").write_text('''\
from dsbf.engine import Requires, SemanticTypes, check
from dsbf.model import (
    Confidence,
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
)
def negative_values(ctx, params):
    """Flag numeric columns that hold a negative value."""
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
''')
```

Load the directory, and add the check by id:

```python
import dsbf

run = dsbf.profile("data.csv", plugins=["my_checks"], checks=["+negative_values"])
assert run.results["negative_values"].status == "ok"
print(len(run.findings.filter(check="negative_values")), "columns with negative values")
```

`dsbf profile data.csv --plugin my_checks --include negative_values` does the same, and
`dsbf plan` lists the check among the ones it would run. A plugin that fails to import becomes a
note on the run, not a crash.

## What a check declares

The decorator is the check's contract. The engine uses it to decide whether a check applies, in what
order checks run, and what to show about it.

| Argument | Meaning |
| --- | --- |
| `id`, `version` | The id is the name users select and write rules against. Raise `version` when a change can move a finding, so a cached result is not reused. |
| `kind` | `rule` for a test of the profile, which cannot read values; `analysis` for a check that reads the values through `ctx.values` and may report a measurement even when nothing is wrong. The engine raises a contract error when a `rule` reads values. |
| `applies_to` | The semantic types it reads (`numeric`, `categorical`, `boolean`, `datetime`, `text`, `id`); the planner gives it only the columns of those types. Leave it out for a check about the whole dataset. |
| `dimensions` | The quality dimensions its findings belong to. |
| `requires` | `profile` parts it reads, `checks` whose results it needs, `reference=True` for a comparison with a second dataset, and `params` that must be set (such as `target`). When a need is missing the check is `not_applicable` with the reason, never an error. |
| `params` | A `CheckParams` model: typed parameters with defaults, bounds and descriptions. |
| `output` | A `CheckOutputModel` subclass, when the check returns more than findings. |
| `cost` | `cheap`, `moderate` or `expensive`; the run's `depth` selects checks by it. |
| `extras` | The optional install groups it needs; without them it ends `unavailable` with the install line. |

## Parameters

A check declares the settings a user may change, with bounds, and DSBF validates them before the
check runs and lists them in the [check reference](../CHECK_REFERENCE.md):

```python
from pathlib import Path

Path("param_checks").mkdir(exist_ok=True)
Path("param_checks/low_values.py").write_text('''\
from pydantic import Field

from dsbf.engine import SemanticTypes, check
from dsbf.model import (
    CheckParams,
    Confidence,
    Dimension,
    Evidence,
    Severity,
    Subject,
)


class LowValuesParams(CheckParams):
    floor: float = Field(0.0, description="Values below this are low.")


@check(
    id="low_values",
    version="1.0.0",
    kind="analysis",
    applies_to=SemanticTypes("numeric"),
    dimensions=[Dimension.VALIDITY],
    params=LowValuesParams,
)
def low_values(ctx, params):
    """Flag numeric columns with values below a floor."""
    for name in ctx.columns:
        values = ctx.values(name)
        low = int((values < params.floor).sum())
        if low:
            ctx.finding(
                code="low_values",
                subject=Subject.column(name),
                severity=Severity.NOTICE,
                confidence=Confidence.HIGH,
                title=f"{name!r} has {low} values below {params.floor}",
                message=f"{low} values of {name!r} are below {params.floor}.",
                evidence=Evidence(
                    metrics={"low": low},
                    n_used=int(values.size),
                    n_total=ctx.profile.n_rows,
                    method="count_below_floor",
                ),
            )
''')

run = dsbf.profile(
    "data.csv",
    plugins=["param_checks"],
    checks=["+low_values"],
    params={"low_values": {"floor": 30}},
)
assert run.results["low_values"].status == "ok"
print(len(run.findings.filter(check="low_values")), "columns with values below 30")
```

DSBF imports a plugin directory once per process, so put each check in a directory you load
before the run. A parameter the check does not declare, or a value outside its bounds, is a `PlanError` before any
check runs.

## Conventions the core checks follow

Follow them and your check behaves like the ones DSBF ships:

- Write the docstring first line as a one-sentence summary; the check reference and `dsbf plan` print it.
- Read columns through `ctx.columns`, the profile and `ctx.values`; never pick columns by dtype.
- State `n_used` and the method on every finding's evidence, so a reader can judge it.
- Say what the evidence cannot show. A method that fails on small samples or on some shapes of data
  states that in its docstring and in the finding text.
- Leave recommendations to the tips DSBF attaches after the check returns.
- Never put a NaN or an infinity in an output or in a metric.
- Call `ctx.checkpoint()` once per iteration of a long loop, the only way a timeout can interrupt it.
- Never call `warnings.catch_warnings()`: the engine records a warning a check raises as a note.

[`dsbf.checks`](../../src/dsbf/checks/__init__.py) lists the rest of the conventions, and the files
beside it are the 47 core checks to copy from. Run `dsbf docs export` to put your check in the
reference.
