# dsbf/interfaces/cli.py

import typer
import yaml

from dsbf.config import load_default_config
from dsbf.eda.profile_engine import ProfileEngine

app = typer.Typer(help="DSBF: Data Scientist's Best Friend — EDA Profiling CLI")


def _load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


@app.command()
def run(
    config: str = typer.Option(..., "--config", "-c", help="Path to config YAML file."),
    strict: bool = typer.Option(False, "--strict", help="Enable strict mode."),
    visualize_dag: bool = typer.Option(
        False, "--visualize-dag", help="Save DAG image."
    ),
    no_report: bool = typer.Option(
        False, "--no-report", help="Skip writing output report."
    ),
):
    """Run profiling using full config."""
    cfg = _load_config(config)
    if strict:
        cfg.setdefault("safety", {})["strict_mode"] = True
    if visualize_dag:
        cfg.setdefault("metadata", {})["visualize_dag"] = True
    if no_report:
        cfg.setdefault("metadata", {})[
            "disable_report"
        ] = True  # you can handle this flag in report_utils

    engine = ProfileEngine(cfg)
    engine.run()


@app.command()
def profile(
    data: str = typer.Argument(..., help="Path to dataset CSV file."),
    depth: str = typer.Option(
        "standard", "--depth", "-d", help="Profiling depth: basic | standard | full"
    ),
):
    """Profile a single dataset using default config."""
    cfg = load_default_config()
    cfg["metadata"]["dataset_path"] = data
    cfg["metadata"]["profiling_depth"] = depth
    engine = ProfileEngine(cfg)
    engine.run()


@app.command()
def quickstart(
    dataset: str = typer.Argument(
        "iris", help="Built-in dataset name (e.g., iris, titanic)."
    )
):
    """Run quick profiling using built-in dataset (e.g., sklearn or seaborn)."""
    cfg = load_default_config()
    cfg["metadata"]["dataset_name"] = dataset
    engine = ProfileEngine(cfg)
    engine.run()


@app.command()
def version():
    """Print DSBF version and exit."""
    from dsbf.utils.versioning import get_dsbf_version

    typer.echo(f"DSBF version: {get_dsbf_version()}")
