# dsbf/interfaces/cli.py

import sys
from pathlib import Path
from typing import Annotated, Any, NoReturn

import typer
import yaml

from dsbf.checks import core_pack
from dsbf.config import ConfigError, ProfileConfig, load_config
from dsbf.data import SamplingError, SourceError, load
from dsbf.engine import (
    Depth,
    DiskCache,
    PlanError,
    Registry,
    Selection,
    format_plan,
    make_plan,
)
from dsbf.interfaces import (
    check_reference,
    openapi_export,
    reference,
    schema_export,
)
from dsbf.interfaces.runs_cli import runs_app
from dsbf.library import profile_with
from dsbf.model import Cost, Dimension, Run
from dsbf.paths import default_outputs_dir
from dsbf.profiling import profile_dataset
from dsbf.sinks import RunReadError, get_sinks, known_sinks
from dsbf.utils.logging_utils import console_logging
from dsbf.utils.versioning import get_dsbf_version

app = typer.Typer(help="DSBF: Data Scientist's Best Friend - EDA Profiling CLI")
schema_app = typer.Typer(help="Publish and guard the result schema.")
app.add_typer(schema_app, name="schema")
app.add_typer(runs_app, name="runs")

VERBOSITIES = ("quiet", "error", "warn", "stage", "info", "debug")


def tolerate_narrow_encodings() -> None:
    """Print ``?`` for a character the terminal cannot encode, not a traceback.

    The summary uses symbols such as ``⦸``; a Windows console in cp1252 cannot encode
    them, and a run that has finished should not fail while printing.
    """
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


@app.callback()
def main(
    context: typer.Context,
    verbosity: str = typer.Option(
        "info",
        "--verbosity",
        "-v",
        help="How much to log: " + ", ".join(VERBOSITIES) + ".",
    ),
) -> None:
    """DSBF: Data Scientist's Best Friend - EDA Profiling CLI."""
    if verbosity not in VERBOSITIES:
        raise typer.BadParameter(
            f"{verbosity!r} is not one of {', '.join(VERBOSITIES)}",
            param_hint="--verbosity",
        )
    tolerate_narrow_encodings()
    context.with_resource(console_logging(verbosity))


Data = Annotated[Path, typer.Argument(help="Dataset to profile (CSV, Parquet, ...).")]
Out = Annotated[
    Path | None,
    typer.Option(
        "--out",
        "-o",
        help="Directory for the output files (default: <DSBF home>/outputs/<run id>).",
    ),
]
STORE_SINK = "store"
Sinks = Annotated[
    list[str] | None,
    typer.Option(
        "--sink",
        "-s",
        help="Output to write (repeatable, default json): "
        + ", ".join((*known_sinks(), STORE_SINK))
        + ". `store` keeps the run in the run store (needs dsbf[server]) and, "
        "alone, writes no directory.",
    ),
]
ConfigFile = Annotated[
    Path | None,
    typer.Option("--config", "-c", help="A settings file (YAML), such as dsbf.yaml."),
]
Overrides = Annotated[
    list[str] | None,
    typer.Option(
        "--set",
        help="Set any setting: a.b=c, the value read as YAML (repeatable). "
        "Wins over the options and the settings file.",
    ),
]
PolicyFile = Annotated[
    Path | None,
    typer.Option("--policy", help="A policy.yaml whose rules judge the run."),
]
Previous = Annotated[
    Path | None,
    typer.Option(
        "--previous",
        help="The run.json of an earlier run, for rules that say new_since: "
        "previous_run.",
    ),
]
Packs = Annotated[
    list[str] | None, typer.Option("--pack", "-p", help="Run only these packs.")
]
Include = Annotated[
    list[str] | None,
    typer.Option(
        "--include", "-i", help="Add a check by id, whatever the filters say."
    ),
]
Exclude = Annotated[
    list[str] | None, typer.Option("--exclude", "-x", help="Drop a check.")
]
DepthOption = Annotated[
    Depth | None,
    typer.Option("--depth", "-d", help="The most expensive checks to allow."),
]
Target = Annotated[
    str | None,
    typer.Option(
        "--target", help="The column to predict, for the checks that need one."
    ),
]
TimeIndex = Annotated[
    str | None,
    typer.Option(
        "--time-index", help="The datetime column the time series checks use."
    ),
]
DatasetKey = Annotated[
    str | None,
    typer.Option("--dataset", help="Groups this run with others of the same dataset."),
]
Plugins = Annotated[
    list[Path] | None,
    typer.Option("--plugin", help="A directory of check plugins to load."),
]
Params = Annotated[
    list[str] | None,
    typer.Option(
        "--param", help="Set a check parameter: check.key=value (repeatable)."
    ),
]
Workers = Annotated[
    int | None,
    typer.Option(
        "--workers", min=1, help="Checks to run at once (default: one per core)."
    ),
]
CheckTimeout = Annotated[
    float | None,
    typer.Option("--check-timeout", min=0, help="Seconds one check may take."),
]
RunBudget = Annotated[
    float | None,
    typer.Option("--run-budget", min=0, help="Seconds the whole run may take."),
]
UseCache = Annotated[
    bool | None,
    typer.Option(
        "--cache/--no-cache",
        help="Reuse results of unchanged checks kept under the DSBF home "
        "(default: the cache.enabled setting, off).",
    ),
]


@app.command()
def profile(
    data: Data,
    out: Out = None,
    sink: Sinks = None,
    config: ConfigFile = None,
    set_: Overrides = None,
    policy: PolicyFile = None,
    previous: Previous = None,
    pack: Packs = None,
    include: Include = None,
    exclude: Exclude = None,
    depth: DepthOption = None,
    target: Target = None,
    time_index: TimeIndex = None,
    plugin: Plugins = None,
    param: Params = None,
    workers: Workers = None,
    check_timeout: CheckTimeout = None,
    run_budget: RunBudget = None,
    cache: UseCache = None,
    dataset: DatasetKey = None,
) -> None:
    """Profile a dataset, write run.json and print the health summary.

    Exits 0 when the run completes, whatever a policy says, and 2 when it cannot start.
    """
    _run(
        data,
        out=out,
        sink=sink,
        config=config,
        overrides=set_,
        policy=policy,
        previous=previous,
        flags=_flags(
            pack,
            include,
            exclude,
            depth,
            target,
            time_index,
            plugin,
            param,
            workers,
            check_timeout,
            run_budget,
            cache,
            dataset,
        ),
        fail_on_policy=False,
    )


@app.command()
def check(
    data: Data,
    out: Out = None,
    sink: Sinks = None,
    config: ConfigFile = None,
    set_: Overrides = None,
    policy: PolicyFile = None,
    previous: Previous = None,
    pack: Packs = None,
    include: Include = None,
    exclude: Exclude = None,
    depth: DepthOption = None,
    target: Target = None,
    time_index: TimeIndex = None,
    plugin: Plugins = None,
    param: Params = None,
    workers: Workers = None,
    check_timeout: CheckTimeout = None,
    run_budget: RunBudget = None,
    cache: UseCache = None,
    dataset: DatasetKey = None,
) -> None:
    """Profile a dataset and judge it by a policy, for a pipeline step.

    Exits 0 when the policy passes the run (or there is none), 1 when a rule with a
    `fail` action matched, and 2 when the run cannot start.
    """
    _run(
        data,
        out=out,
        sink=sink,
        config=config,
        overrides=set_,
        policy=policy,
        previous=previous,
        flags=_flags(
            pack,
            include,
            exclude,
            depth,
            target,
            time_index,
            plugin,
            param,
            workers,
            check_timeout,
            run_budget,
            cache,
            dataset,
        ),
        fail_on_policy=True,
    )


def _flags(  # noqa: PLR0913
    pack: list[str] | None,
    include: list[str] | None,
    exclude: list[str] | None,
    depth: Depth | None,
    target: str | None,
    time_index: str | None,
    plugin: list[Path] | None,
    param: list[str] | None,
    workers: int | None,
    check_timeout: float | None,
    run_budget: float | None,
    cache: bool | None,
    dataset: str | None,
) -> dict[str, Any]:
    """The options that were given, as the nested mapping of the ``cli`` layer."""
    given: dict[str, Any] = {
        "packs": pack,
        "include": include,
        "exclude": exclude,
        "depth": depth.value if depth else None,
        "target": target,
        "time_index": time_index,
        "dataset": dataset,
        "plugins": [str(path) for path in plugin or []],
        "params": _parse_params(param or []),
        "workers": workers,
        "limits": {
            "check_timeout_seconds": check_timeout,
            "run_budget_seconds": run_budget,
        },
        "cache": {"enabled": cache},
    }
    for section in ("limits", "cache"):
        given[section] = {k: v for k, v in given[section].items() if v is not None}
    return {key: value for key, value in given.items() if value}


def _run(  # noqa: PLR0913
    data: Path,
    *,
    out: Path | None,
    sink: list[str] | None,
    config: Path | None,
    overrides: list[str] | None,
    policy: Path | None,
    previous: Path | None,
    flags: dict[str, Any],
    fail_on_policy: bool,
) -> None:
    requested = tuple(dict.fromkeys(sink or ())) or ("json",)
    sinks = tuple(name for name in requested if name != STORE_SINK)
    to_store = STORE_SINK in requested
    try:
        get_sinks(sinks)
    except ValueError as error:
        _fail(error)
    if to_store:
        _require_store()
    layer = _merge_settings(flags, _parse_settings(overrides or []))
    try:
        settings = load_config(ProfileConfig, project_file=config, cli=layer)
        run = profile_with(
            data, settings, policy=policy, previous=previous if policy else None
        )
    except PlanError as error:
        for problem in error.problems:
            typer.echo(f"error: {problem}", err=True)
        raise typer.Exit(2) from error
    except (SourceError, ConfigError, SamplingError, RunReadError) as error:
        _fail(error)
    pointer = None
    if sinks:
        directory = (
            (out or default_outputs_dir() / str(run.run_id)).expanduser().resolve()
        )
        written = run.write(directory, sinks=sinks)
        pointer = str(written.get("json", next(iter(written.values()))))
    if to_store:
        pointer = _store_run(run, settings.store.snapshot_rows, pointer)
    typer.echo(run.render_health(pointer=pointer))
    if run.policy is not None:
        typer.echo(run.policy.render())
        if fail_on_policy and run.policy.exit_code:
            raise typer.Exit(run.policy.exit_code)


def _require_store() -> None:
    """Exit 2 naming the extra when the run store is not installed."""
    try:
        import dsbf.store  # noqa: F401, PLC0415 (the base install has no store)
    except ImportError as error:
        _fail(error)


def _store_run(run: Run, snapshot_rows: int, pointer: str | None) -> str:
    """Keep ``run`` in the store; the pointer to show: its files, else its id."""
    from dsbf.store import StoreError, save  # noqa: PLC0415

    try:
        save(run, snapshot_rows=snapshot_rows)
    except StoreError as error:
        _fail(error)
    return pointer or f"dsbf runs show {run.run_id}"


def _fail(error: Exception) -> NoReturn:
    typer.echo(f"error: {error}", err=True)
    raise typer.Exit(2) from error


def _parse_settings(assignments: list[str]) -> dict[str, Any]:
    """``a.b=c`` options as a nested mapping; values are YAML."""
    settings: dict[str, Any] = {}
    for assignment in assignments:
        path, equals, text = assignment.partition("=")
        keys = path.split(".")
        if not (equals and all(keys)):
            raise typer.BadParameter(
                f"{assignment!r} is not a.b=c, such as limits.run_budget_seconds=60",
                param_hint="--set",
            )
        node = settings
        for key in keys[:-1]:
            child = node.setdefault(key, {})
            if not isinstance(child, dict):
                raise typer.BadParameter(
                    f"{path!r} sets {key!r} to a value and to sub-keys",
                    param_hint="--set",
                )
            node = child
        node[keys[-1]] = yaml.safe_load(text)
    return settings


def _merge_settings(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    """``top`` over ``base``, mappings key by key."""
    merged = dict(base)
    for key, value in top.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_settings(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_params(assignments: list[str]) -> dict[str, dict[str, Any]]:
    """``check.key=value`` options as ``{check: {key: value}}``; values are YAML."""
    params: dict[str, dict[str, Any]] = {}
    for assignment in assignments:
        target, equals, text = assignment.partition("=")
        check_id, dot, key = target.partition(".")
        if not (equals and dot and check_id and key):
            raise typer.BadParameter(
                f"{assignment!r} is not check.key=value, such as null_share.min=0.5"
            )
        params.setdefault(check_id, {})[key] = yaml.safe_load(text)
    return params


@app.command()
def dashboard(
    host: str = typer.Option("127.0.0.1", "--host", help="Local address to listen on."),
    port: int = typer.Option(8765, "--port", min=1, max=65535, help="Port to use."),
    open_: bool = typer.Option(
        True, "--open/--no-open", help="Open the browser once the server starts."
    ),
) -> None:
    """Serve the stored runs over the local API (and the dashboard, when built)."""
    try:
        from dsbf.service.serve import NotLocalError, serve  # noqa: PLC0415
    except ImportError as error:
        _fail(error)
    typer.echo(f"serving the runs in the store on http://{host}:{port}/ (Ctrl+C stops)")
    try:
        serve(host, port, open_browser=open_)
    except NotLocalError as error:
        _fail(error)


@app.command("plan")
def plan_command(
    data: Path = typer.Argument(..., help="Dataset to plan for (CSV, Parquet, ...)."),
    pack: list[str] = typer.Option([], "--pack", "-p", help="Run only these packs."),
    include: list[str] = typer.Option(
        [], "--include", "-i", help="Add a check by id, whatever the filters say."
    ),
    exclude: list[str] = typer.Option([], "--exclude", "-x", help="Drop a check."),
    depth: Depth = typer.Option(
        Depth.STANDARD, "--depth", "-d", help="The most expensive checks to allow."
    ),
    max_cost: Cost | None = typer.Option(
        None, "--max-cost", help="Lower the cost ceiling set by --depth."
    ),
    dimension: list[Dimension] = typer.Option(
        [], "--dimension", help="Keep checks of these dimensions."
    ),
    tag: list[str] = typer.Option([], "--tag", help="Keep checks with these tags."),
    plugin: list[Path] = typer.Option(
        [], "--plugin", help="A directory of check plugins to load."
    ),
    entry_points: bool = typer.Option(  # noqa: FBT001
        True,
        "--entry-points/--no-entry-points",
        help="Load checks that installed packages offer.",
    ),
    param: list[str] = typer.Option(
        [], "--param", help="Set a check parameter: check.key=value (repeatable)."
    ),
) -> None:
    """Show which checks would run on a dataset, and what each is expected to do."""
    params = _parse_params(param)
    try:
        dataset = load(data)
        registry = Registry()
        registry.register_pack(core_pack())
        plan = make_plan(
            registry,
            Selection(
                packs=pack,
                include=include,
                exclude=exclude,
                depth=depth,
                max_cost=max_cost,
                dimensions=dimension,
                tags=tag,
                plugins=plugin,
                entry_points=entry_points,
            ),
            params=params,
            profile=profile_dataset(dataset),
        )
    except SourceError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(2) from error
    except PlanError as error:
        for problem in error.problems:
            typer.echo(f"error: {problem}", err=True)
        raise typer.Exit(2) from error
    shape = f"{dataset.n_rows:,} × {dataset.n_columns}"
    typer.echo(format_plan(plan, dataset=f"{dataset.name} ({shape})"))


@schema_app.command("export")
def export_schema(
    out: Path = typer.Option(
        Path("schema"), "--out", "-o", help="Directory for the JSON Schemas."
    ),
    types: Path | None = typer.Option(
        None,
        "--types",
        help="Also write TypeScript types to this file (needs pnpm and the frontend).",
    ),
    frontend: Path = typer.Option(
        Path("frontend"), "--frontend", help="Where json2ts is installed."
    ),
    check: bool = typer.Option(
        False, "--check", help="Write nothing; fail if the files are out of date."
    ),
) -> None:
    """Write the JSON Schemas and the Parquet column listing; optionally TypeScript."""
    try:
        expected = schema_export.export_schema(out, check=check)
        expected[out / openapi_export.OPENAPI_FILE] = openapi_export.openapi_text()
        if not check:
            (out / openapi_export.OPENAPI_FILE).write_text(
                expected[out / openapi_export.OPENAPI_FILE], encoding="utf-8"
            )
        if types is not None:
            run_text = expected[out / schema_export.SCHEMA_FILE]
            expected[types] = schema_export.generate_typescript(run_text, frontend)
    except schema_export.SchemaToolError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(2) from error
    if check:
        stale = schema_export.stale_files(expected)
        stale += schema_export.orphaned_check_schemas(out, expected)
        for path in stale:
            typer.echo(f"out of date: {path}", err=True)
        if stale:
            typer.echo("run `dsbf schema export` and commit the result", err=True)
            raise typer.Exit(1)
        typer.echo("schema files are up to date")
        return
    if types is not None:
        types.parent.mkdir(parents=True, exist_ok=True)
        types.write_text(expected[types], encoding="utf-8")
    for path in expected:
        typer.echo(f"wrote {path}")


@schema_app.command("check")
def check_schema(
    base: str = typer.Option(
        "origin/main", "--base", help="Git revision holding the published schema."
    ),
    schema_file: Path = typer.Option(
        Path("schema") / schema_export.SCHEMA_FILE, "--schema", help="Schema to check."
    ),
) -> None:
    """Fail if the schema changed in a way its version number doesn't announce."""
    try:
        comparison = schema_export.compare_with_base(base, schema_file)
        check_comparisons = schema_export.compare_check_schemas(
            base, schema_file.parent
        )
        tables = schema_export.compare_tables_with_base(
            base, schema_file.parent / schema_export.TABLES_SCHEMA_FILE
        )
        policy = schema_export.compare_policy_with_base(
            base, schema_file.parent / schema_export.POLICY_SCHEMA_FILE
        )
        api = openapi_export.compare_openapi_with_base(
            base, schema_file.parent / openapi_export.OPENAPI_FILE
        )
    except schema_export.SchemaToolError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(2) from error
    failed = False
    if comparison is None:
        typer.echo(f"{base} has no {schema_file}; nothing to compare")
    else:
        for change in comparison.changes:
            typer.echo(str(change))
        if comparison.problems:
            failed = True
            for problem in comparison.problems:
                typer.echo(f"error: {problem}", err=True)
            typer.echo(
                "bump SCHEMA_VERSION in dsbf/model/run.py, then run "
                "`dsbf schema export`",
                err=True,
            )
        else:
            typer.echo(
                f"schema {comparison.base_version} to {comparison.version}: "
                f"{len(comparison.changes)} change(s), announced"
            )
    if tables is not None:
        for change in tables.changes:
            typer.echo(f"tables: {change}")
        for problem in tables.problems:
            failed = True
            typer.echo(f"error: tables: {problem}", err=True)
        if tables.problems:
            typer.echo(
                "bump TABLES_SCHEMA_VERSION in dsbf/sinks/tables.py, then run "
                "`dsbf schema export`",
                err=True,
            )
    if policy is not None:
        for change in policy.changes:
            typer.echo(f"policy: {change}")
        for problem in policy.problems:
            failed = True
            typer.echo(f"error: policy: {problem}", err=True)
        if policy.problems:
            typer.echo(
                "bump Policy.version in dsbf/model/policy.py, then run "
                "`dsbf schema export`",
                err=True,
            )
    if api is not None:
        for change in api.changes:
            typer.echo(f"openapi: {change}")
        for problem in api.problems:
            failed = True
            typer.echo(f"error: openapi: {problem}", err=True)
        if api.problems:
            typer.echo(
                "bump API_VERSION in dsbf/service/app.py, then run "
                "`dsbf schema export`",
                err=True,
            )
    for check_id, found in check_comparisons:
        for change in found.changes:
            typer.echo(f"{check_id}: {change}")
        for problem in found.problems:
            failed = True
            typer.echo(f"error: {check_id}: {problem}", err=True)
    if any(found.problems for _, found in check_comparisons):
        typer.echo("bump the check's version, then run `dsbf schema export`", err=True)
    if failed:
        raise typer.Exit(1)


cache_app = typer.Typer(help="Manage the results kept between runs.")
app.add_typer(cache_app, name="cache")


@cache_app.command("clear")
def clear_cache() -> None:
    """Remove every result kept under <DSBF home>/cache/results."""
    cache = DiskCache.default()
    try:
        cleared = cache.clear()
    except OSError as error:
        _fail(error)
    typer.echo(
        f"removed {cleared.removed} cached result{'' if cleared.removed == 1 else 's'} "
        f"({cleared.freed_bytes:,} bytes) from {cache.root}"
    )


docs_app = typer.Typer(help="Publish the command-line and settings reference.")
app.add_typer(docs_app, name="docs")


@docs_app.command("export")
def export_reference(
    out: Path = typer.Option(
        Path("docs") / reference.REFERENCE_FILE,
        "--out",
        "-o",
        help="File to write the command-line and settings reference to.",
    ),
    checks_out: Path = typer.Option(
        Path("docs") / check_reference.CHECK_REFERENCE_FILE,
        "--checks-out",
        help="File to write the check reference to.",
    ),
    check: bool = typer.Option(
        False, "--check", help="Write nothing; fail if a file is out of date."
    ),
) -> None:
    """Write the generated command-line, settings and check references."""
    files = {
        out: reference.render_reference(app),
        checks_out: check_reference.render_check_reference(),
    }
    if check:
        stale = [
            path
            for path, text in files.items()
            if not path.is_file() or path.read_text(encoding="utf-8") != text
        ]
        for path in stale:
            typer.echo(f"out of date: {path}", err=True)
        if stale:
            typer.echo("run `dsbf docs export` and commit the result", err=True)
            raise typer.Exit(1)
        typer.echo("the references are up to date")
        return
    for path, text in files.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        typer.echo(f"wrote {path}")


@app.command()
def version() -> None:
    """Print DSBF version and exit."""
    typer.echo(f"DSBF version: {get_dsbf_version()}")
