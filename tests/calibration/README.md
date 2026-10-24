# Calibration suite

A check can pass every unit test and still be wrong in use: too eager (it flags clean data)
or too timid (it misses a flaw that is there). The calibration suite measures that. It
generates frames with a *known* truth, profiles them over many seeds, and reports for each
check how often it finds what was planted and how often it raises an alarm nobody planted.
The rates are committed in `baseline.json`, so a change that makes a check worse shows up
as a failing scenario, and a change that makes it better shows up as a diff to accept.

It is deselected by default (several minutes), and `.github/workflows/calibration.yml` runs
it nightly, on demand, and on a pull request labelled `calibration`. Add that label to any
pull request that changes a threshold, a method or a check's version. The run uploads
`calibration-results/` (the JSON and a Markdown table of every check against its baseline).

## Run it

```bash
uv run pytest -m calibration                          # every scenario, against the baseline
uv run pytest -m calibration -k leakage               # the scenarios whose name matches
uv run pytest -m calibration --update-calibration-baseline   # accept the measured rates

# A quick look without pytest: rates per check, or every finding of one frame
uv run python -m tests.calibration.measure --scenario noise_numeric --seeds 5 --workers 4
uv run python -m tests.calibration.measure --scenario leakage_flipped --explain 0
```

`--explain SEED` prints every finding of one frame marked `OK` (explained by a plant), `~~`
(allowed), `!!` (a false alarm) or blank (not a claim), followed by the plants. It is the
fastest way to see why a rate is what it is.

## What is measured

A **scenario** (`scenarios/*.py`) is a function from a seed to a frame and its truth. Frames
come from `numpy.random.default_rng(seed)` alone, so a seed gives the same frame on every
machine, and the scenario's `seeds` count says how many are run. Every scenario states the
truth it plants:

- a **plant** says that a check must report something about these columns (optionally with
  a given finding code, a least severity, or a test on the finding);
- an **allowed** finding is a consequence of the plants (outliers also make a column
  non-normal), so it counts neither as a hit nor as a false alarm;
- a **probe** reads a number off a check's result when a rate cannot say enough (the share
  of planted near-duplicate pairs found, whether a traffic-light level is the expected one).

Scoring (`scoring.py`):

- A check is *scored* on a frame only when it ended `ok`. One that was not applicable is
  counted under `unscored`, never as a miss. A scenario that plants a flaw for a check that
  could not run fails.
- An **alarm** is a finding of severity `warn` or `error`. Three groups differ, because their
  findings mean something else: the checks that only ever report at `info` (`INFO_CLAIMS`)
  claim with every finding, `reference_drift` claims from `notice`, and the advice checks
  (`ADVICE_CHECKS`) advise on every column, so they are scored only where a scenario says
  which advice is right.
- A plant is **detected** when any finding of its check, at the severity it asks for, is about
  the planted columns. A **false alarm** is an alarm that no plant and no allowance explains.
- `detection_rate` is the share of plants detected. `false_alarm_rate` is the share of scored
  trials with at least one false alarm. Each has a 95% Wilson interval in the JSON.

A scenario fails when a rate in it got worse than its baseline by more than the baseline's
interval (at most 0.15 of it) plus 0.02, when a probe's mean moved the wrong way by more
than 0.05, or when a check's version differs from the baseline's. The runs are seeded, so
unchanged code gives unchanged rates; the slack absorbs a library upgrade that moves a
borderline case.

## Scenarios

| File | Plants |
|---|---|
| `noise.py` | Nothing: independent columns at 150, 3 000 and 30 000 rows. Every alarm is false. |
| `distributions.py` | Outliers (common, rare, joint), skew, heavy tails, two modes, normality at 8 000 rows. |
| `rules.py` | Nulls, a constant, a frozen column, zeros, a dominant value, high cardinality, an identifier, copies, damaged text. |
| `strings.py` | UUIDs, digests, base64, tokens; mixed types; case, padding and hidden characters; a format rule and mixed date formats. |
| `duplicates.py` | Exact repeats; near-duplicate records with one or two edited fields, at 800 records (every pair scored) and 2 300 (rare-token blocking); the same people in two systems with different keys, with and without `exclude_columns`; look-alike records that are distinct. |
| `relationships.py` | Correlated, categorical and collinear columns; group differences, including unequal variances and sizes. |
| `leakage.py` | A target with copies, recodes, flipped labels, honest strong features, and leaks through nulls, an identifier's prefix and a timestamp. |
| `missingness.py` | Nulls that are random, follow other columns, or follow the column itself. |
| `series.py` | White noise, AR(1), a random walk, a ramp, a level shift, a weekly cycle, holes in the index. |
| `drift.py` | A reference that is the same population, and one that moved. |
| `advice.py` | Columns where one transform, binning, encoding or imputation is clearly right. |
| `contracts.py` | Bounds, a schema, class imbalance, unparsed dates, too few rows. |
| `synthesis.py` | Tables whose flaws should read as a given level on the quality traffic light. |

## Add a scenario

1. Write `@scenario("name", seeds=20)` over a function that returns a `Planted` (a frame, its
   `plants`, any `allowed` findings, the `checks` to run, and the `target`, `time_index`,
   `reference` or per-check `params` the checks need). Its docstring is its summary.
2. Look at it with `--explain 0`. Every plant should be `OK` or its miss should be the thing
   you mean to measure; a `!!` on a column you did not plant is either a false alarm to
   record or an allowance you forgot.
3. Run `pytest -m calibration -k name --update-calibration-baseline` and commit the
   baseline. `test_scoring.py` checks that every scenario is reproducible and names real
   checks.

Plants state what *should* be true, not what the check says today. A scenario may plant what
a check misses so that the baseline records the miss, and the change that fixes the check
moves the rate in the diff.

## Calibrating a check

1. Measure the current rate (`--scenario ... --seeds 40`) and add the scenario that shows the
   defect if there is none.
2. Change the threshold or method, and raise the check's version.
3. Re-measure the scenarios that name the check, and every null scenario (a fix for one
   direction often costs the other).
4. Regenerate the golden snapshots the change touches, update the baseline, and record the
   rates before and after in the commit.
