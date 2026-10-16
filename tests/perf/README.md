# Performance benchmark

`pytest -m perf` profiles four datasets end to end, each in a fresh interpreter, and compares
what it measured with `baseline.json`. It is deselected by default (kitchen50k alone takes
over ten minutes), and `.github/workflows/perf.yml` runs it nightly, on demand, and on a pull
request labelled `perf`. Add that label to any pull request that is meant to change speed or
memory, or that touches the engine or a slow task.

| Scenario | Input | Runs |
|---|---|---|
| `titanic` | `tests/golden/data/titanic.csv` (891 x 15) | 3, median kept |
| `adversarial` | `tests/fixtures/adversarial.py` (200 x 14) | 3, median kept |
| `kitchen5k` | `DSBFTestDataGenerator(n_rows=5000, random_seed=42)`, `normal_date` as the time series | 1 |
| `kitchen50k` | the same generator at 50,000 rows (52,500 x 60 after the generator's own additions) | 1 |

## Run it

```bash
uv run pytest -m perf                       # all four, compare with the baseline
uv run pytest -m perf -k titanic            # one scenario
uv run pytest -m perf --perf-no-compare     # measure only
uv run pytest -m perf -k titanic --perf-cprofile   # also write run.prof (slower; diagnosis only)
```

Each run writes `perf-results/perf.json` (and a Markdown table next to it, `perf.md`) in the
directory you ran from; `--perf-results PATH` changes that. The directory is git-ignored.

The baseline was recorded on GitHub's `ubuntu-latest` runner. A local run on a different
machine is only comparable roughly: a faster machine passes, a slower one can fail without
anything having regressed. Use `--perf-no-compare` for exploration, and trust the CI job.

## What is measured

`perf.json` has an `environment` block (Python, platform, CPU count, commit, time) and one
entry per scenario:

- `total_seconds`: wall time of `ProfileEngine(...).run()`, without the interpreter start-up
  and without `cProfile`, which would inflate every number.
- `import_seconds`: time to import `dsbf`.
- `stages`: wall time of each function `ProfileEngine.run()` calls (loading, sampling,
  building the graph, `graph.run (all tasks)`, the report, persisting the run).
- `tasks`: `task_durations` for every task.
- `peak_rss_mb`: the process's peak resident memory (`ru_maxrss`).
- `output_bytes`, `report_bytes`: size of the output directory and of `report.json`.
- `api`: status, bytes and milliseconds of the dashboard requests that open a run (run
  detail, tasks, figures, data quality, ML readiness, associations, sample), made through the
  API test client against the run's own database.
- `task_outcomes` and `failed_tasks`: a scenario fails if any task fails, because the timings
  of a crashed run describe less work than the baseline did.

## What fails

A value regresses when it is more than 25% above the baseline **and** above a floor for its
kind, so shared-runner noise on small values doesn't fail the job:

| Value | Floor |
|---|---|
| total, stage, task | 1 s |
| peak memory | 100 MB |
| API payload bytes | 10 KB |

Endpoints that answered 200 in the baseline and no longer do also fail. API milliseconds are
recorded but not gated. A task or stage that is not in the baseline is not compared.

`PYTHONHASHSEED=0` is pinned for every run: `fuzzy_duplicate_detection` walks sets of strings,
and its amount of work (and so its time) depends on the iteration order.

## Update the baseline

A deliberate change in speed or memory moves the baseline in the same pull request:

1. Label the pull request `perf`, and let the `Performance` workflow finish.
2. Download the `perf-results` artifact and copy its `perf.json` to `tests/perf/baseline.json`.
3. Review `git diff tests/perf/baseline.json` and describe the change in the commit body.

`uv run pytest -m perf --update-perf-baseline` writes the baseline directly from a local run
(it merges the scenarios that ran into the file), but a local baseline makes the CI job
compare against another machine. Don't commit one.

Never edit `baseline.json` by hand.
