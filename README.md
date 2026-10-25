# DSBF — Data Scientist's Best Friend

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-22c55e)](LICENSE)
[![Code style: Ruff](https://img.shields.io/badge/code%20style-ruff-d97706)](https://github.com/astral-sh/ruff)
[![Type checked: Pyrefly](https://img.shields.io/badge/type%20checked-pyrefly-2563eb)](https://pyrefly.org/)

**DSBF is a statistically grounded, workflow-aware profiling engine designed to be a reliable co-pilot at every phase of the data science process.**

The goal is simple: the exploratory and preparatory portions of data science work are largely reproducible. The same questions get asked of every new dataset — what's missing, what's skewed, what's redundant, what's suspicious — and they deserve a standardized, rigorous answer every time. DSBF provides that answer through a structured task engine, a five-dimension quality framework, and an interactive dashboard that turns raw data into actionable insight within minutes.

The first phase — **Exploratory Data Analysis** — is what DSBF does today. It gives any data scientist, regardless of experience level, an immediately reproducible structure for understanding a new dataset: consistent findings, plain-English guidance grounded in statistical theory, and a dashboard designed to accelerate insight rather than replace judgment. From there, a dataset can be understood on its own terms before any modeling assumptions are introduced.

The broader vision is a tool that grows with the workflow — covering data preparation, modeling, inference, and tradeoff analysis under the same statistically honest, reproducible framework. That work is ongoing.

---

## Standing on the shoulders of giants

Tools like [ydata-profiling](https://github.com/ydataai/ydata-profiling) and [Great Expectations](https://github.com/great-expectations/great-expectations) are genuinely excellent and were a direct inspiration for this project. ydata-profiling in particular demonstrated how much value a single well-designed profiling report could deliver to a data scientist's workflow.

DSBF takes a different set of design bets:

**On scoring:** most tools surface an aggregate quality score. DSBF deliberately does not — a single number creates false anchoring and masks the specific dimensions that actually matter. Five independently-scored dimensions (Completeness, Validity, Usability, Redundancy, Leakage) replace it, each with its own traffic-light indicator and findings.

**On language:** DSBF treats epistemic honesty as a design constraint. Where the evidence is ambiguous — missingness mechanism classification is a good example — DSBF uses hedged language ("consistent with MAR") and caps confidence deliberately. It will never claim to confirm what the data can only suggest.

**On workflow scope:** rather than producing a static report, DSBF is designed as a growing co-pilot. The EDA phase is what exists today. Modeling preparation, modeling, inference, and tradeoff analysis are the phases being built toward — all under the same statistically rigorous, reproducible framework.

---

## Dashboard

<img src="docs/assets/Overview_Tab.png" alt="DSBF Dashboard" width="100%">

The Vue 3 dashboard in `frontend/` read the results of the old engine through an API that has been removed. It does not run until the store, the API and the dashboard shell are rebuilt on the `Run` (Phase 6 in [`ROADMAP.md`](ROADMAP.md)); the source stays in the repository. Until then, `dsbf profile` writes the whole `Run` to `run.json` and prints a health summary, and `dsbf.profile()` returns the `Run` in memory.

The screenshot shows what the dashboard presented: five traffic-light quality indicators (Completeness, Validity, Usability, Redundancy, Leakage), per-column distributions, pairwise associations, severity-ranked findings, five ML-preparation dimensions with a gate, and time series diagnostics. `run.results["quality_dimensions"]` and `run.results["ml_readiness"]` already carry the quality and readiness levels.

---

## Architecture

```
CSV / Parquet / DataFrame
      │
      ▼
 dsbf.data.load ──► Dataset ──► profile_dataset ──► DatasetProfile
                                                         │
                                                         ▼
┌─────────────────────────────────────────────────────────────┐
│  Check engine  (plan → execute)                             │
│  47 core checks · selection by depth, pack and id ·         │
│  dependency order · limits · cache · failures become notes  │
└────────────────────────────┬────────────────────────────────┘
                             │  CheckResult objects (findings, output, notes)
                             ▼
                        Run  ──►  run.json
```

Every check is a function with a declared spec: the semantic types it applies to, what it needs from the profile or from other checks, its cost, and the model of its parameters. The planner selects and orders the checks, the executor runs them (in parallel where it can) under time limits, and a check that raises ends with a status and a note instead of stopping the run. The result is one `Run`, a validated model whose JSON Schema is committed as `schema/run.schema.json`.

---

## Key design principles

**No single aggregate scores.** Both a numeric DQ score and column-level ML Readiness scores were designed and then scrapped — single numbers create false anchoring and ambiguous meaning across domains. Five independently-scored dimensions replaced them.

**Epistemic honesty as a first-class constraint.** Missingness mechanism analysis (MCAR / MAR / MNAR classification) deliberately caps confidence at "moderate" and uses `consistent_with` language throughout. DSBF will never claim to *confirm* MNAR — only to report evidence consistent with it.

**Severity-aware traffic lights.** A four-state system (🟢 green / 🔵 blue / 🟡 amber / 🔴 red) maps both *severity* and *proportion* to color. A single `warn`-level finding on a 500-column dataset is always amber — it never hides behind a green dot just because it affects 0.2% of columns.

**Semantic type filtering.** A check names the semantic types it applies to, and the planner gives it only the columns of those types, not the columns of a storage dtype. A boolean stored as `int64` will not enter VIF computation or outlier detection.

---

## Installation

```bash
git clone https://github.com/W-Thurston/dsbf.git
cd dsbf
uv sync
```

**Extras.** The package declares optional extras for the heavier parts: `ml`, `timeseries`, `text`, `pandas`, `sql`, `server`, and `all` for every one of them. Today the base install still carries most of what the checks import, so the extras change nothing for a normal `uv sync`; they become the way to opt in as those dependencies move out of the base install. The exception is `text`: `fuzzy_duplicates`, which runs at the `deep` depth, needs `rapidfuzz`, and without it ends `unavailable` with an install hint.

**Requirements:** Python 3.12+ · [uv](https://docs.astral.sh/uv/) · Node.js 24 (only for the dashboard source in `frontend/`)

---

## Quickstart

### CLI — profile a dataset

```bash
uv run dsbf profile data.csv
uv run dsbf profile data.parquet --depth deep --target survived --out results/
uv run dsbf profile data.csv --time-index date --param null_rate.notice_share=0.1
```

`dsbf profile` runs `dsbf.profile()`, writes the whole `Run` to `run.json` and prints the health summary with the path of that file. The file is written to `--out` (made absolute), or to `<home>/outputs/<run id>/` without it. `--depth` is `quick | standard | deep` and defaults to `standard`. `--pack`, `--plugin` and `--param check.key=value` can be repeated; a parameter named here wins over `--target` and `--time-index`. The exit code is 0 once a run completes, whatever its findings or failed checks (the summary says which), and 2 when the run cannot be set up (an unreadable file, an unknown check or parameter, a target that is not a column). `uv run dsbf plan data.csv` shows what would run, without running it.

### Python library — `dsbf.profile()`

```python
import dsbf
from dsbf.model import Severity

run = dsbf.profile("data.csv", depth="quick", target="survived")
print(run.render_health())  # check counts, and why any check did not complete
worst = [f for f in run.findings if f.severity >= Severity.WARN]
```

`dsbf.profile()` takes a pandas, Polars or Arrow table or a file path, and returns a `Run` in memory. It writes nothing to disk, and no check can stop it: a check that fails ends with a status and a note in `run.results`. `run.findings` lists the findings, highest severity first. A Python warning raised inside a check doesn't reach stderr; it becomes a `python_warning` note on that check, one per category and message, with a `count`. While the call lasts, the process's warning filters show every repeat of a warning (your `ignore` and `error` filters still apply), so your other threads may print a warning more than once in that time.

| Argument | Meaning |
|---|---|
| `depth` | `quick`, `standard` (the default) or `deep`: the most expensive kind of check to run |
| `packs`, `checks` | The packs to draw checks from (`core`, the checks DSBF ships, by default); `checks=["-outliers", "+zeros"]` removes and adds checks by id |
| `target`, `time_index` | The columns the checks that need a target or a time axis use |
| `sample` | `dsbf.Sample(rows=100_000, strategy="random", seed=1)`, or a row count for the first rows; the `Run` records how the rows were chosen |
| `params` | Parameters of a check, by id, which win over `target` and `time_index`: `{"null_rate": {"notice_share": 0.1}}`; an unknown check or parameter is a `PlanError` |
| `limits`, `workers` | `LimitsConfig(check_timeout_seconds=…, run_budget_seconds=…)`, and how many checks run at once: one per core, at most 8 |
| `cache` | `True` keeps results under the DSBF home, so a second call re-runs only the checks whose inputs changed; off by default |
| `hooks`, `plugins` | Functions that receive an event as each check starts and finishes; directories of check plugins |
| `config` | The path of a settings file with the same names. An argument here overrides the file, and `DSBF__SECTION__KEY` environment variables sit between the two. No file is read unless you name it |

`run.results["quality_dimensions"].output` reports completeness, validity, usability, redundancy and leakage from the other checks' findings, each with a level, the columns it flags and a coverage (the share of its input checks that ran); a dimension no check could assess, such as leakage without a `target`, has no level rather than a green one. `run.results["ml_readiness"].output` does the same for what to do before modelling and gives a gate (`ready`, `needs_work`, `not_ready`).

A `Run` can be megabytes on data with many columns, because a check's output is kept whole: the titanic `Run` is 0.56 MB (the largest outputs are `group_differences` at 52 KB and `contingency` at 44 KB). It was 8.9 MB until `fuzzy_duplicates`, which lists its 20 closest pairs and counts the rest, replaced `fuzzy_duplicate_detection`'s whole pair list.

`dsbf plan data.csv` and `dsbf schema export` are the other commands; `dsbf --help` lists them.

---

## Profiling depth

Each depth includes the ones above it. The 47 core checks are 30 `quick`, 16 `standard` and 1 `deep`.

| Depth | Adds | Best for |
|---|---|---|
| `quick` | The cheap checks: shape, nulls, types, duplicates, constant columns, value counts, distributions, outliers, encoding and imputation advice, time series rules, the quality and ML-readiness summaries | Quick triage, large datasets |
| `standard` | + the moderate checks: associations, group differences, collinearity, normality and bimodality, missingness mechanism, time series tests, target-aware and reference checks | Typical EDA workflow |
| `deep` | + `fuzzy_duplicates`, near-duplicate rows by fuzzy matching | Pre-modeling audit |

`dsbf profile` runs at `standard` unless you pass `--depth`; `dsbf plan data.csv --depth deep` lists exactly what each depth would run on your data.

---

## Output artifacts

`dsbf profile` writes one file, `run.json`, to `--out` or to `<home>/outputs/<run id>/`, whatever the working directory. The home is `$DSBF_HOME` if set, otherwise the per-user data directory (`~/.local/share/dsbf` on Linux). `dsbf.profile()` writes nothing unless you pass `cache=True`, which keeps check results under `<home>/cache/`.

| File | Description |
|---|---|
| `run.json` | The whole `Run`: dataset identity and sampling, the profile, every check's status, findings and output, notes and timing. It validates against `schema/run.schema.json` |

---

## Configuration

`dsbf.profile()` takes every setting as an argument (see the table above), or from a settings file you name with `config=` (the same names; `DSBF__SECTION__KEY` environment variables sit between the file and the arguments). Nothing is read from the working directory unless you name it.

```yaml
depth: deep
target: survived
sample:
  rows: 100000
  strategy: random
  seed: 1
params:
  null_rate:
    notice_share: 0.1
limits:
  check_timeout_seconds: 120
```

---

## Extending DSBF

A check is a function registered with a decorator. Save this as `my_checks/negative_values.py`:

```python
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
```

Then load the directory with `dsbf.profile(df, plugins=["my_checks"], checks=["+negative_values"])`; `dsbf profile data.csv --plugin my_checks` loads it too, and `dsbf plan` lists it. A plugin that fails to import becomes a note on the run, not an error. `tests/test_engine/test_plugins.py` holds working examples, and `docs/REFACTOR_PLAN.md` describes the check contract.

---

## Validation and calibration

DSBF ships with two suites that check its checks against data whose truth is known.

The **validation suite** (`tests/validation/`) profiles eight dataset archetypes in memory
and states what the core checks must say about each:

| Dataset | Purpose |
|---|---|
| `clean` | Baseline — no warning anywhere, every quality dimension green |
| `tiny` | 25-row graceful degradation, low-N reliability warnings |
| `near_clean` | Mild issues detected, and the dataset lands at amber, not red |
| `all_categorical` | Zero numeric columns — numeric checks step aside with a reason |
| `high_missingness` | Structured MAR patterns, extreme null rates |
| `severe_multicollinearity` | Four VIF failure modes including scale-mismatch regression test |
| `single_column` | Pair-based checks degrade gracefully with one column |
| `wide` | 100-column scale — every column reported, none truncated |

```bash
uv run pytest tests/validation
```

The **calibration suite** (`tests/calibration/`) generates frames with a planted flaw (or
none) from a seed and measures, per check and per version, how often it finds the flaw
and how often it raises an alarm nobody planted. The rates are committed in
`tests/calibration/baseline.json` and compared nightly in CI.

```bash
uv run pytest -m calibration                 # all scenarios, against the baseline
uv run pytest -m calibration -k leakage      # the leakage scenarios
```

---

## Roadmap

The plan for what comes next, phase by phase, is in [`ROADMAP.md`](ROADMAP.md). The design behind it is in [`docs/REFACTOR_PLAN.md`](docs/REFACTOR_PLAN.md), and [`docs/architecture.md`](docs/architecture.md) describes how the code works today.

DSBF is designed to eventually cover the full data science workflow — EDA today, then modeling preparation, modeling, inference, and tradeoff analysis — all under the same statistically honest, reproducible framework. Each phase will follow the same principle: the reproducible parts of data science work should be standardized so practitioners can focus their judgment on the parts that actually require it.

---

## Contributing

Contributions are welcome. Please read [`CONTRIBUTING.md`](CONTRIBUTING.md) for conventions on task structure, commit style, and the review process.

```bash
# Run the fast gate (about 30 seconds), then the full suite (about 5 minutes, needs network)
uv run pytest -m "unit and not slow"
uv run pytest

# Lint and format
uv run ruff check .
uv run ruff format --check .
```

The validation suite is described above.

All contributions must include docstrings, pass Ruff linting (`max-line-length = 88`), and include or update relevant validation assertions.

---

## License

MIT — see [`LICENSE`](LICENSE) for details.
