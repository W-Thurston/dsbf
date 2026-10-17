# dsbf/interfaces/cli.py

import json
from pathlib import Path
from typing import Any

import typer
import yaml

from dsbf.config import load_default_config
from dsbf.eda.profile_engine import ProfileEngine
from dsbf.eda.task_registry import load_task_group
from dsbf.interfaces import schema_export
from dsbf.utils.task_utils import build_task_metadata, write_task_metadata
from dsbf.utils.versioning import get_dsbf_version

app = typer.Typer(help="DSBF: Data Scientist's Best Friend - EDA Profiling CLI")
tasks_app = typer.Typer(help="Inspect the registered tasks.")
app.add_typer(tasks_app, name="tasks")
schema_app = typer.Typer(help="Publish and guard the result schema.")
app.add_typer(schema_app, name="schema")


def _load_config(config_path: str) -> dict:
    with Path.open(config_path, "r") as f:
        return yaml.safe_load(f)


@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c", help="Path to config YAML file."),
    strict: bool = typer.Option(  # noqa: FBT001
        False,
        "--strict",
        help="Enable strict mode.",
    ),
    visualize_dag: bool = typer.Option(  # noqa: FBT001
        False,
        "--visualize-dag",
        help="Save DAG image.",
    ),
    no_report: bool = typer.Option(  # noqa: FBT001
        False,
        "--no-report",
        help="Skip writing output report.",
    ),
) -> None:
    """Run profiling using full config."""
    cfg: dict = _load_config(config)
    if strict:
        cfg.setdefault("safety", {})["strict_mode"] = True
    if visualize_dag:
        cfg.setdefault("metadata", {})["visualize_dag"] = True
    if no_report:
        cfg.setdefault("metadata", {})["disable_report"] = (
            True  # you can handle this flag in report_utils
        )

    engine = ProfileEngine(cfg)
    engine.run()


@app.command()
def profile(
    data: str = typer.Argument(..., help="Path to dataset CSV file."),
    depth: str = typer.Option(
        "standard",
        "--depth",
        "-d",
        help="Profiling depth: basic | standard | full",
    ),
    name: str = typer.Option(
        None,
        "--name",
        "-n",
        help="Dataset name shown in the dashboard and stored in the run database. "
        "Defaults to the CSV filename stem (e.g. 'my_data' from 'my_data.csv').",
    ),
) -> None:
    """Profile a single dataset using default config."""
    cfg: dict[str, Any] = load_default_config()
    cfg["metadata"]["dataset_path"] = data
    cfg["metadata"]["profiling_depth"] = depth
    # Use the provided name, or fall back to the CSV filename stem so the
    # dashboard always shows a meaningful name rather than the default_config value.
    cfg["metadata"]["dataset_name"] = name or Path(data).stem
    engine = ProfileEngine(cfg)
    engine.run()


@app.command()
def quickstart(
    dataset: str = typer.Argument(
        "iris",
        help="Built-in dataset name (e.g., iris, titanic).",
    ),
) -> None:
    """Run quick profiling using built-in dataset (e.g., sklearn or seaborn)."""
    cfg: dict[str, Any] = load_default_config()
    cfg["metadata"]["dataset_name"] = dataset
    # Explicitly clear dataset_path so writer.py never stores a stale file path
    # from default_config.yaml as the source for a built-in dataset.
    cfg["metadata"]["dataset_path"] = None
    engine = ProfileEngine(cfg)
    engine.run()


@tasks_app.command("export")
def export_tasks(
    output: Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write the JSON to this file instead of printing it.",
    ),
) -> None:
    """Export the static metadata of every registered task as JSON."""
    load_task_group("core")
    if output is None:
        typer.echo(json.dumps(build_task_metadata(), indent=2))
    else:
        write_task_metadata(str(output))
        typer.echo(f"Task metadata written to {output}")


@schema_app.command("export")
def export_schema(
    out: Path = typer.Option(
        Path("schema"), "--out", "-o", help="Directory for the JSON Schema."
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
    """Write the JSON Schema of a run, and optionally its TypeScript types."""
    try:
        schema_file, text = schema_export.export_schema(out, check=check)
        expected = {schema_file: text}
        if types is not None:
            expected[types] = schema_export.generate_typescript(text, frontend)
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
