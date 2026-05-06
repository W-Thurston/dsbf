# dsbf/interfaces/cli.py

from pathlib import Path
from typing import Any

import typer
import yaml

from dsbf.config import load_default_config

# from dsbf.dashboard.render_dashboard import render_and_show
from dsbf.eda.profile_engine import ProfileEngine
from dsbf.utils.versioning import get_dsbf_version

app = typer.Typer(help="DSBF: Data Scientist's Best Friend - EDA Profiling CLI")


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


@app.command()
def version() -> None:
    """Print DSBF version and exit."""
    typer.echo(f"DSBF version: {get_dsbf_version()}")


# @app.command()
# def render_dashboard(
#     output_dir: str = typer.Argument(
#         ...,
#         help="Path to output folder with report.json",
#     ),
#     save_html: bool = typer.Option(
#         False,
#         "--save-html",
#         help="Export to standalone HTML instead of serving.",
#     ),
# ) -> None:
#     """Render the interactive dashboard from a completed DSBF run."""
#     render_and_show(output_dir=output_dir, save_html=save_html)
