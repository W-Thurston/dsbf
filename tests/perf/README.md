# Performance benchmark

`pytest -m perf` times the profile pass on two datasets and `dsbf.profile()` on six scenarios,
each in a fresh interpreter. It compares what it measured with `baseline.json`, and holds the
profile pass to absolute budgets. It is deselected by default (the slowest scenarios take
minutes), and `.github/workflows/perf.yml` runs it nightly, on demand, and on a pull request
labelled `perf`. Add that label to any pull request that is meant to change speed or memory,
or that touches the engine or a slow check.

The profile scenarios time only `dsbf.profiling` (type inference, `profile_dataset`, stage
inference), after the file is loaded:

| Scenario | Input | Runs | Budget |
|---|---|---|---|
| `profile-kitchen50k` | the `kitchen50k` frame (52,500 x 60), as CSV | 3, median kept | pass under 2 s; memory under 2.5x |
| `profile-wide1m` | 1,000,000 x 50 floats (`numpy` seed 42), as Parquet | 3, median kept | memory under 2.5x; time compared with the baseline only |

The engine scenarios time one `dsbf.profile()` call (loading the file, the profile pass,
planning and executing the checks of the default selection, the `core` pack), in a fresh
interpreter. Each writes the
wall time of every check as its `tasks`, so the executor's effect shows per check as well
as in the total. The `engine-*` pairs run every check (`depth="deep"`), serial and
parallel. The `standard-*` scenarios run what `dsbf.profile()` runs when no depth is
given, which is what the Phase 4 targets are about (REFACTOR_PLAN section 11). The summary
lists each target the run misses, and nothing fails on one. The baseline comparison
applies to them as to every scenario.

An engine entry's `peak_rss_mb` is the child's `VmHWM` from `/proc/self/status`. Linux keeps
`ru_maxrss` across `exec`, so a child started after the harness had generated the
`profile-wide1m` frame reported the harness's 959 MB whatever it did.

| Scenario | Input | Runs | `depth` | `workers` | Target (reported) |
|---|---|---|---|---|---|
| `engine-titanic-serial` | titanic | 3, median kept | `deep` | 1 | |
| `engine-titanic-parallel` | titanic | 3, median kept | `deep` | the executor's default (one per core, at most 8) | |
| `engine-kitchen5k-serial` | kitchen5k, `normal_date` as the time index | 1 | `deep` | 1 | |
| `engine-kitchen5k-parallel` | kitchen5k, `normal_date` as the time index | 1 | `deep` | one per core | |
| `standard-titanic` | titanic | 3, median kept | `standard` | `dsbf.profile()`'s choice (one per core) | under 5 s |
| `standard-kitchen50k` | kitchen50k, `normal_date` as the time index | 1 | `standard` | `dsbf.profile()`'s choice | under 60 s and 1 GB peak |

## Run it

```bash
uv run pytest -m perf                       # every scenario, compare with the baseline
uv run pytest -m perf -k titanic            # the titanic scenarios
uv run pytest -m perf -k profile            # the profile pass only (about 30 s)
uv run pytest -m perf -k engine             # dsbf.profile() at deep, serial and parallel
uv run pytest -m perf -k standard           # dsbf.profile() at the standard depth
uv run pytest -m perf --perf-no-compare     # measure only
```

Each run writes `perf-results/perf.json` (and a Markdown table next to it, `perf.md`) in the
directory you ran from; `--perf-results PATH` changes that. The directory is git-ignored.

The baseline was recorded on GitHub's `ubuntu-latest` runner. A local run on a different
machine is only comparable roughly: a faster machine passes, a slower one can fail without
anything having regressed. Use `--perf-no-compare` for exploration, and trust the CI job.

## What is measured

`perf.json` has an `environment` block (Python, platform, CPU count, commit, time) and one
entry per scenario. An engine entry has:

- `total_seconds`: wall time of the `dsbf.profile()` call, without the interpreter start-up.
- `import_seconds`: time to import `dsbf`.
- `tasks`: the wall time of every check.
- `peak_rss_mb`: the process's peak resident memory.
- `run_bytes`, `n_checks`, `statuses`: the size of the `Run` as JSON and how its checks ended.
- `failed_checks`: a scenario fails if any check errors or times out, because the timings of
  a crashed run describe less work than the baseline did.

### The profile pass

`measure_profile.py` records the same `total_seconds`, `stages` and `peak_rss_mb`, with these
differences:

- `total_seconds` is the pass alone. `stages.load` is timed too, but is not in the total.
- `data_mb` is `Dataset.frame.estimated_size()`. The child resets the peak resident size
  after loading (`/proc/self/clear_refs`), so `pass_overhead_mb` is what the pass added to
  the loaded frame, and `memory_ratio = 1 + pass_overhead_mb / data_mb`: the frame plus what
  the pass adds, over the frame. Where the peak can't be reset (not Linux) `peak_isolated`
  is `false`, the figure includes the load, and the memory budget is not applied.
- There are no `tasks`.

## What fails

The budgets in `compare.PROFILE_BUDGETS` hold with or without a baseline, and also when the
baseline is being rewritten, so a pass that misses them can't become the reference.
Beyond that, a profile entry is compared with the baseline as below.

A value regresses when it is more than 25% above the baseline **and** above a floor for its
kind, so shared-runner noise on small values doesn't fail the job:

| Value | Floor |
|---|---|
| total, stage, check | 1 s |
| peak memory | 100 MB |

A check or stage that is not in the baseline is not compared.

`dsbf.profile()` has no check whose work depends on the order of a set of strings, so the hash
seed is not pinned.

## Update the baseline

A deliberate change in speed or memory moves the baseline in the same pull request:

1. Label the pull request `perf`, and let the `Performance` workflow finish.
2. Download the `perf-results` artifact and copy its `perf.json` to `tests/perf/baseline.json`.
3. Review `git diff tests/perf/baseline.json` and describe the change in the commit body.

`uv run pytest -m perf --update-perf-baseline` writes the baseline directly from a local run
(it merges the scenarios that ran into the file), but a local baseline makes the CI job
compare against another machine. Don't commit one.

Never edit `baseline.json` by hand.
