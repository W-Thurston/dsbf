# dsbf/interfaces/cli.py

from pathlib import Path
from typing import Any

import typer
import yaml

from dsbf.checks import core_pack
from dsbf.config import ConfigError
from dsbf.data import SamplingError, SourceError, load
from dsbf.engine import Depth, PlanError, Registry, Selection, format_plan, make_plan
from dsbf.interfaces import schema_export
from dsbf.library import profile as dsbf_profile
from dsbf.model import Cost, Dimension
from dsbf.paths import default_outputs_dir
from dsbf.profiling import profile_dataset
from dsbf.utils.logging_utils import console_logging
from dsbf.utils.versioning import get_dsbf_version

RUN_FILE = "run.json"

app = typer.Typer(help="DSBF: Data Scientist's Best Friend - EDA Profiling CLI")
schema_app = typer.Typer(help="Publish and guard the result schema.")
app.add_typer(schema_app, name="schema")


@app.command()
def profile(
    data: Path = typer.Argument(..., help="Dataset to profile (CSV, Parquet, ...)."),
    out: Path | None = typer.Option(
        None,
        "--out",
        "-o",
        help="Directory for run.json (default: <DSBF home>/outputs/<run id>).",
    ),
    pack: list[str] = typer.Option([], "--pack", "-p", help="Run only these packs."),
    depth: Depth = typer.Option(
        Depth.STANDARD, "--depth", "-d", help="The most expensive checks to allow."
    ),
    target: str | None = typer.Option(
        None, "--target", help="The column to predict, for the checks that need one."
    ),
    time_index: str | None = typer.Option(
        None, "--time-index", help="The datetime column the time series checks use."
    ),
    plugin: list[Path] = typer.Option(
        [], "--plugin", help="A directory of check plugins to load."
    ),
    param: list[str] = typer.Option(
        [], "--param", help="Set a check parameter: check.key=value (repeatable)."
    ),
) -> None:
    """Profile a dataset, write run.json and print the health summary."""
    params = _parse_params(param)
    try:
        with console_logging("info"):
            run = dsbf_profile(
                data,
                packs=pack or None,
                depth=depth,
                target=target,
                time_index=time_index,
                plugins=plugin,
                params=params,
            )
    except (SourceError, ConfigError, SamplingError) as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(2) from error
    except PlanError as error:
        for problem in error.problems:
            typer.echo(f"error: {problem}", err=True)
        raise typer.Exit(2) from error
    directory = (out or default_outputs_dir() / str(run.run_id)).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    run_file = directory / RUN_FILE
    run_file.write_text(run.model_dump_json(indent=2), encoding="utf-8")
    typer.echo(run.render_health(pointer=str(run_file)))


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
    """Write the JSON Schemas of a run and a policy, and optionally TypeScript types."""
    try:
        expected = schema_export.export_schema(out, check=check)
        if types is not None:
            run_text = expected[out / schema_export.SCHEMA_FILE]
            expected[types] = schema_export.generate_typescript(run_text, frontend)
    except schema_export.SchemaToolError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(2) from error
    if check:
        stale = schema_export.stale_files(expected)
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
    except schema_export.SchemaToolError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(2) from error
    if comparison is None:
        typer.echo(f"{base} has no {schema_file}; nothing to compare")
        return
    for change in comparison.changes:
        typer.echo(str(change))
    if comparison.problems:
        for problem in comparison.problems:
            typer.echo(f"error: {problem}", err=True)
        typer.echo(
            "bump SCHEMA_VERSION in dsbf/model/run.py, then run `dsbf schema export`",
            err=True,
        )
        raise typer.Exit(1)
    typer.echo(
        f"schema {comparison.base_version} to {comparison.version}: "
        f"{len(comparison.changes)} change(s), announced"
    )


@app.command()
def version() -> None:
    """Print DSBF version and exit."""
    typer.echo(f"DSBF version: {get_dsbf_version()}")
