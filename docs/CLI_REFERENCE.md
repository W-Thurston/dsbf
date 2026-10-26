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
| `-s, --sink <str>` | Output to write (repeatable, default json): json, ndjson, findings_parquet, profile_parquet, results_parquet, html, parquet. |
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
| `-s, --sink <str>` | Output to write (repeatable, default json): json, ndjson, findings_parquet, profile_parquet, results_parquet, html, parquet. |
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
| `--types <path>` | Also write TypeScript types to this file (needs pnpm and the frontend). |
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

### `dsbf docs`

```
dsbf docs [OPTIONS] COMMAND [ARGS]...
```

Publish the command-line and settings reference.

### `dsbf docs export`

```
dsbf docs export [OPTIONS]
```

Write the generated command-line and settings reference.

| Option | Description |
| --- | --- |
| `-o, --out <path>` | File to write. [default: docs/CLI_REFERENCE.md] |
| `--check` | Write nothing; fail if the file is out of date. |

## Settings

The settings file (`--config`), `DSBF__SECTION__KEY` environment variables, `--set a.b=c` and the options of `dsbf profile` all fill these settings, later layers winning in that order. `dsbf.profile()` takes the same names as arguments.

| Setting | Type | Default | Description |
| --- | --- | --- | --- |
| `packs` | tuple[str, ...] | `('core',)` | The packs whose checks are candidates. |
| `include` | tuple[str, ...] | `()` | Checks to add by id, whatever the filters say. |
| `exclude` | tuple[str, ...] | `()` | Checks to remove by id. |
| `depth` | Depth | `'standard'` | The most expensive checks to allow. |
| `target` | str | unset | The column to predict, for the checks that need one. |
| `time_index` | str | unset | The datetime column the time series checks use. |
| `sample` | SampleConfig | unset | Profile a sample of the rows rather than all of them. |
| `limits` | LimitsConfig | see below | How long a check, and the whole run, may take; unset leaves a limit off. |
| `workers` | int | unset | Checks to run at once; unset is one per core. |
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
