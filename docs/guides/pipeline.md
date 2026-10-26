# Use DSBF in your pipeline

DSBF profiles a table, returns the result as one `Run`, and can judge that run by a policy you
write. This guide covers the three ways to run it in a pipeline step: from Python, from the command
line, and from a CI workflow.

The examples read `data.csv`, a table of customers (`age`, `income`, `purchase_amount`, `region`,
`plan_type`, `is_churned` and a few more columns). The test suite runs every `python` and `bash`
block here against a copy of it.

## From Python

```python
import dsbf

run = dsbf.profile("data.csv", depth="quick")
print(run.render_health())
```

`dsbf.profile()` takes a path, or a pandas, Polars or Arrow table, and returns the `Run` in memory.
It writes nothing and no check can stop it: a check that fails ends with a status and a note. Pick
the findings you care about with `run.findings`, which answers the same questions as a policy
rule's `when`:

```python
from dsbf.model import Severity

serious = run.findings.filter(severity=">=warn")
print(len(serious), "findings at warn or above")
print(serious.to_polars().select("check_id", "code", "severity", "columns"))
```

Write the run when another step needs it:

```python
paths = run.write("out", sinks=["json", "ndjson", "parquet", "html"])
print(sorted(path.name for path in paths.values()))
```

and read it back, in this process or another, with `Run.read`:

```python
from dsbf import Run

again = Run.read("out")
assert again.run_id == run.run_id
```

`sinks` takes `json` (`run.json`, the whole run), `ndjson` (`findings.ndjson`), `parquet`
(`findings.parquet`, `profile.parquet` and `results.parquet`), `html` (a self-contained
`report.html`) and the single Parquet files by name (`findings_parquet`, `profile_parquet`,
`results_parquet`).

## Judge a run with a policy

A run reports; it does not fail by itself. A [policy](policy.md) turns findings into a decision:

```python
run = dsbf.profile(
    "data.csv",
    depth="quick",
    policy={
        "rules": [
            {"name": "no-heavy-tails", "when": {"code": "heavy_tails"}, "then": "fail"}
        ]
    },
)
print(run.policy.failed, run.policy.exit_code)
```

`run.policy` is absent when no policy was given. Pass `previous=` (a `Run`, or the path of an earlier
`run.json`) to let a rule select only findings that are `new_since: previous_run`.

## From the command line

`dsbf check` profiles a dataset, writes the outputs and exits with a code a pipeline can act on.

```bash
dsbf check data.csv --depth quick --out results --sink json --sink parquet
```

| Exit code | Meaning |
| --- | --- |
| 0 | The run completed, and no policy rule with a `fail` action matched (or there was no policy). |
| 1 | A rule with a `fail` action matched. |
| 2 | The run could not start: the input, the settings or the policy is invalid. |

Add `--policy policy.yaml` to apply a policy, and `--previous results/run.json` to compare with an
earlier run. `dsbf profile` takes the same options and always exits 0 on a completed run. The
generated [command-line and settings reference](../CLI_REFERENCE.md) lists every option.

Settings can sit in a file, with `DSBF__SECTION__KEY` environment variables on top and
command-line options over both:

```bash
dsbf plan data.csv --depth quick
```

`dsbf plan` lists what a run would do without running it: the checks it would run, and why any would
not.

## Limits, caching and speed

- `limits=LimitsConfig(check_timeout_seconds=..., run_budget_seconds=...)` (or `--check-timeout`
  and `--run-budget`) ends a slow check with a `timeout` status and a note instead of stopping the
  run.
- `workers` is how many checks run at once: one per core, at most 8.
- `cache=True` keeps check results under the DSBF home (`$DSBF_HOME`, else the per-user data
  directory), so a second call re-runs only the checks whose inputs changed. `dsbf cache clear` empties it.
- `sample=dsbf.Sample(rows=100_000, strategy="random", seed=1)` profiles a sample, and the `Run`
  records how the rows were chosen.

## Logging

A library call prints nothing. DSBF logs to the `dsbf` logger with a `NullHandler`, so unless your
application configures `logging`, nothing appears. The library logs warnings only (a cache it could
not prune, a hook that raised), so the default `logging.WARNING` setting shows all of it. To see
progress of a run, pass `hooks=`, functions that receive an event as each check starts and finishes:

```python
events = []
dsbf.profile("data.csv", depth="quick", hooks=[events.append])
print(events[0].kind, events[-1].kind)
```

The command line configures logging for you with `--verbosity`, which accepts `quiet`, `error`,
`warn`, `stage`, `info` (the default) and `debug`.

## In a GitHub Actions workflow

[`examples/github-actions/dsbf-check.yml`](../../examples/github-actions/dsbf-check.yml) installs
DSBF, runs `dsbf check` and uploads the Parquet findings as a workflow artifact, so every run of the
pipeline leaves a table you can load into a warehouse. It shows the exit codes in action: a policy
`fail` marks the job failed after the upload. To use it, copy it to `.github/workflows/`, and point
`DATA` at your file. [Write a policy](policy.md) explains the rules in `policy.yaml`.
