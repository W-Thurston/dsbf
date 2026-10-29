# DSBF command-line and settings reference

> **Generated file.** Do not edit it. `dsbf docs export` writes it from the Typer app
> and from `ProfileConfig`, and `dsbf docs export --check` fails when it is stale.

Every command exits 0 on success, 2 when the run cannot start (bad input, settings or
policy), and, for `dsbf check`, 1 when a policy rule with a `fail` action matched.

## Commands

### `dsbf`

```
dsbf [OPTIONS] COMMAND [ARGS]...
```

DSBF: Data Scientist's Best Friend - EDA Profiling CLI

| Option | Description |
| --- | --- |
| `-v, --verbosity <str>` | How much to log: quiet, error, warn, stage, info, debug. [default: info] |
| `--install-completion` | Install completion for the current shell. |
| `--show-completion` | Show completion for the current shell, to copy it or customize the installation. |

### `dsbf profile`

```
dsbf profile [OPTIONS] DATA
```

Profile a dataset, write run.json and print the health summary.

Exits 0 when the run completes, whatever a policy says, and 2 when it cannot start.

| Option | Description |
| --- | --- |
| `-o, --out <path>` | Directory for the output files (default: <DSBF home>/outputs/<run id>). |
| `-s, --sink <str>` | Output to write (repeatable, default json): json, ndjson, findings_parquet, profile_parquet, results_parquet, html, parquet, store. `store` keeps the run in the run store (needs dsbf[server]) and, alone, writes no directory. |
| `-c, --config <path>` | A settings file (YAML), such as dsbf.yaml. |
| `--set <str>` | Set any setting: a.b=c, the value read as YAML (repeatable). Wins over the options and the settings file. |
| `--policy <path>` | A policy.yaml whose rules judge the run. |
| `--previous <path>` | The run.json of an earlier run, for rules that say new_since: previous_run. |
| `-p, --pack <str>` | Run only these packs. |
| `-i, --include <str>` | Add a check by id, whatever the filters say. |
| `-x, --exclude <str>` | Drop a check. |
| `-d, --depth <quick\|standard\|deep>` | The most expensive checks to allow. |
| `--target <str>` | The column to predict, for the checks that need one. |
| `--time-index <str>` | The datetime column the time series checks use. |
| `--plugin <path>` | A directory of check plugins to load. |
| `--param <str>` | Set a check parameter: check.key=value (repeatable). |
| `--workers <int range>` | Checks to run at once (default: one per core). [x>=1] |
| `--check-timeout <float range>` | Seconds one check may take. [x>=0] |
| `--run-budget <float range>` | Seconds the whole run may take. [x>=0] |
| `--cache / --no-cache` | Reuse results of unchanged checks kept under the DSBF home (default: the cache.enabled setting, off). |
| `--dataset <str>` | Groups this run with others of the same dataset. |

### `dsbf check`

```
dsbf check [OPTIONS] DATA
```

Profile a dataset and judge it by a policy, for a pipeline step.

Exits 0 when the policy passes the run (or there is none), 1 when a rule with a
`fail` action matched, and 2 when the run cannot start.

| Option | Description |
| --- | --- |
| `-o, --out <path>` | Directory for the output files (default: <DSBF home>/outputs/<run id>). |
| `-s, --sink <str>` | Output to write (repeatable, default json): json, ndjson, findings_parquet, profile_parquet, results_parquet, html, parquet, store. `store` keeps the run in the run store (needs dsbf[server]) and, alone, writes no directory. |
| `-c, --config <path>` | A settings file (YAML), such as dsbf.yaml. |
| `--set <str>` | Set any setting: a.b=c, the value read as YAML (repeatable). Wins over the options and the settings file. |
| `--policy <path>` | A policy.yaml whose rules judge the run. |
| `--previous <path>` | The run.json of an earlier run, for rules that say new_since: previous_run. |
| `-p, --pack <str>` | Run only these packs. |
| `-i, --include <str>` | Add a check by id, whatever the filters say. |
| `-x, --exclude <str>` | Drop a check. |
| `-d, --depth <quick\|standard\|deep>` | The most expensive checks to allow. |
| `--target <str>` | The column to predict, for the checks that need one. |
| `--time-index <str>` | The datetime column the time series checks use. |
| `--plugin <path>` | A directory of check plugins to load. |
| `--param <str>` | Set a check parameter: check.key=value (repeatable). |
| `--workers <int range>` | Checks to run at once (default: one per core). [x>=1] |
| `--check-timeout <float range>` | Seconds one check may take. [x>=0] |
| `--run-budget <float range>` | Seconds the whole run may take. [x>=0] |
| `--cache / --no-cache` | Reuse results of unchanged checks kept under the DSBF home (default: the cache.enabled setting, off). |
| `--dataset <str>` | Groups this run with others of the same dataset. |

### `dsbf dashboard`

```
dsbf dashboard [OPTIONS]
```

Serve the stored runs and the dashboard over the local API.

| Option | Description |
| --- | --- |
| `--host <str>` | Local address to listen on. [default: 127.0.0.1] |
| `--port <int range>` | Port to use. [default: 8765; 1<=x<=65535] |
| `--open / --no-open` | Open the browser once the server starts. [default: open] |

### `dsbf plan`

```
dsbf plan [OPTIONS] DATA
```

Show which checks would run on a dataset, and what each is expected to do.

| Option | Description |
| --- | --- |
| `-p, --pack <str>` | Run only these packs. |
| `-i, --include <str>` | Add a check by id, whatever the filters say. |
| `-x, --exclude <str>` | Drop a check. |
| `-d, --depth <quick\|standard\|deep>` | The most expensive checks to allow. [default: standard] |
| `--max-cost <cheap\|moderate\|heavy>` | Lower the cost ceiling set by --depth. |
| `--dimension <completeness\|validity\|usability\|redundancy\|leakage\|distribution\|relationship\|temporal>` | Keep checks of these dimensions. |
| `--tag <str>` | Keep checks with these tags. |
| `--plugin <path>` | A directory of check plugins to load. |
| `--entry-points / --no-entry-points` | Load checks that installed packages offer. [default: entry-points] |
| `--param <str>` | Set a check parameter: check.key=value (repeatable). |

### `dsbf version`

```
dsbf version [OPTIONS]
```

Print DSBF version and exit.

### `dsbf schema`

```
dsbf schema [OPTIONS] COMMAND [ARGS]...
```

Publish and guard the result schema.

### `dsbf schema export`

```
dsbf schema export [OPTIONS]
```

Write the JSON Schemas and the Parquet column listing; optionally TypeScript.

| Option | Description |
| --- | --- |
| `-o, --out <path>` | Directory for the JSON Schemas. [default: schema] |
| `--types <path>` | Also write TypeScript types to this file, with the API types and the rank tables beside it (needs pnpm and the frontend). |
| `--frontend <path>` | Where json2ts is installed. [default: frontend] |
| `--check` | Write nothing; fail if the files are out of date. |

### `dsbf schema check`

```
dsbf schema check [OPTIONS]
```

Fail if the schema changed in a way its version number doesn't announce.

| Option | Description |
| --- | --- |
| `--base <str>` | Git revision holding the published schema. [default: origin/main] |
| `--schema <path>` | Schema to check. [default: schema/run.schema.json] |

### `dsbf runs`

```
dsbf runs [OPTIONS] COMMAND [ARGS]...
```

Browse, compare and clean the stored runs.

### `dsbf runs list`

```
dsbf runs list [OPTIONS]
```

The stored runs, newest first.

| Option | Description |
| --- | --- |
| `--dataset <str>` | Only runs of this dataset key. |
| `--limit <int range>` | Runs to show. [default: 20; 1<=x<=500] |
| `--json` | Print JSON instead of text. |

### `dsbf runs show`

```
dsbf runs show [OPTIONS] RUN
```

A run's health summary and policy verdict.

| Option | Description |
| --- | --- |
| `--json` | Print JSON instead of text. |

### `dsbf runs diff`

```
dsbf runs diff [OPTIONS] BASE HEAD
```

What the head run shows that the base run did not.

| Option | Description |
| --- | --- |
| `--tolerance <float range>` | Relative change a metric must exceed to count as changed. [default: 0.05; x>=0] |
| `--json` | Print JSON instead of text. |

### `dsbf runs delete`

```
dsbf runs delete [OPTIONS] RUN
```

Remove a run and its stored files.

### `dsbf runs gc`

```
dsbf runs gc [OPTIONS]
```

Delete runs no rule keeps, and stored files no run owns.

A run is kept when any given rule keeps it. With neither rule, only orphaned
files are removed.

| Option | Description |
| --- | --- |
| `--keep <int range>` | Keep the newest N runs of each dataset. [x>=0] |
| `--older-than <float range>` | Keep the runs younger than this many days. [x>=0] |
| `--dry-run` | Report what would go; remove nothing. |
| `--json` | Print JSON instead of text. |

### `dsbf runs import`

```
dsbf runs import [OPTIONS] PATH
```

Store a run written by the json sink, artifacts included.

### `dsbf cache`

```
dsbf cache [OPTIONS] COMMAND [ARGS]...
```

Manage the results kept between runs.

### `dsbf cache clear`

```
dsbf cache clear [OPTIONS]
```

Remove every result kept under <DSBF home>/cache/results.

### `dsbf docs`

```
dsbf docs [OPTIONS] COMMAND [ARGS]...
```

Publish the command-line and settings reference.

### `dsbf docs export`

```
dsbf docs export [OPTIONS]
```

Write the generated command-line, settings and check references.

| Option | Description |
| --- | --- |
| `-o, --out <path>` | File to write the command-line and settings reference to. [default: docs/CLI_REFERENCE.md] |
| `--checks-out <path>` | File to write the check reference to. [default: docs/CHECK_REFERENCE.md] |
| `--check` | Write nothing; fail if a file is out of date. |

## Settings

The settings file (`--config`), `DSBF__SECTION__KEY` environment variables, `--set a.b=c` and the options of `dsbf profile` all fill these settings, later layers winning in that order. `dsbf.profile()` takes the same names as arguments.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `dataset` | str | unset | The key that groups this run with earlier runs of the same dataset. Unset, a file's absolute path is the key, and an in-memory table has none. |
| `packs` | tuple[str, ...] | `('core',)` | The packs whose checks are candidates. |
| `include` | tuple[str, ...] | `()` | Checks to add by id, whatever the filters say. |
| `exclude` | tuple[str, ...] | `()` | Checks to remove by id. |
| `depth` | Depth | `'standard'` | The most expensive checks to allow. |
| `target` | str | unset | The column to predict, for the checks that need one. |
| `time_index` | str | unset | The datetime column the time series checks use. |
| `sample` | SampleConfig | unset | Profile a sample of the rows rather than all of them. |
| `limits` | LimitsConfig | see below | How long a check, and the whole run, may take; unset leaves a limit off. |
| `workers` | int | unset | Checks to run at once; unset is one per core. |
| `check_settings` | dict[str, CheckSettings] | `{}` | A timeout or `runs_in` for single checks, by check id. |
| `cache` | CacheConfig | see below | Keep the results of checks between runs under the DSBF home. |
| `store` | StoreConfig | see below | What the run store keeps beside a run. |
| `plugins` | tuple[str, ...] | `()` | Directories of check plugins to load. |
| `params` | dict[str, dict[str, Any]] | `{}` | Parameters of checks, by check id; they win over `target` and `time_index`. |

### `sample`

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `sample.rows` | int | required | How many rows to keep. |
| `sample.strategy` | SampleStrategy | `'head'` | How the rows are chosen. |
| `sample.seed` | int | unset | Seeds a random or stratified sample. |
| `sample.stratify_by` | tuple[str, ...] | `()` | The columns whose groups a stratified sample keeps. |

### `limits`

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `limits.check_timeout_seconds` | float | unset | Seconds one check may take. |
| `limits.run_budget_seconds` | float | unset | Seconds the whole run may take. |

### `cache`

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `cache.enabled` | bool | `False` | Reuse results of checks whose inputs are unchanged. |
| `cache.max_megabytes` | int | `500` | Evict the least recently used entries past this size; unset keeps all. |
| `cache.max_age_days` | float | `90` | Evict entries not used for this long; unset keeps all. |

### `store`

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `store.snapshot_rows` | int | `10000` | Rows of the analysed data to keep as a snapshot, sampled (seed 0) above this; 0 keeps none. |
