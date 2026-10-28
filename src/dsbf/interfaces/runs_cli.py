# dsbf/interfaces/runs_cli.py
"""``dsbf runs``: browse, compare and clean the run store."""

import json
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any, NoReturn
from uuid import UUID

import typer

from dsbf.comparison import DEFAULT_TOLERANCE, compare
from dsbf.model import Run
from dsbf.sinks import RunReadError

runs_app = typer.Typer(help="Browse, compare and clean the stored runs.")

AsJson = Annotated[bool, typer.Option("--json", help="Print JSON instead of text.")]


def _fail(error: object) -> NoReturn:
    typer.echo(f"error: {error}", err=True)
    raise typer.Exit(2)


def _repository() -> Any:
    try:
        from dsbf.store import open_repository  # noqa: PLC0415 (needs the extra)
    except ImportError as error:
        _fail(error)
    return open_repository()


def _load(reference: str, repository: Any = None) -> Run:
    """A run from a ``run.json`` (or its directory), else from the store by id."""
    path = Path(reference).expanduser()
    if path.exists():
        try:
            return Run.read(path)
        except RunReadError as error:
            _fail(error)
    try:
        run_id = UUID(reference)
    except ValueError:
        _fail(f"{reference!r} is neither a run.json path nor a run id")
    from dsbf.store import StoreError  # noqa: PLC0415

    repository = repository or _repository()
    try:
        return repository.get(run_id)
    except StoreError as error:
        _fail(error)


def _echo_json(value: object) -> None:
    typer.echo(json.dumps(value, indent=2, ensure_ascii=False))


@runs_app.command("list")
def list_runs(
    dataset: Annotated[
        str | None, typer.Option("--dataset", help="Only runs of this dataset key.")
    ] = None,
    limit: Annotated[int, typer.Option(min=1, max=500, help="Runs to show.")] = 20,
    as_json: AsJson = False,
) -> None:
    """The stored runs, newest first."""
    repository = _repository()
    try:
        page = repository.list_runs(dataset_key=dataset, limit=limit)
    finally:
        repository.close()
    if as_json:
        _echo_json(page.model_dump(mode="json"))
        return
    for item in page.items:
        severity = item.max_severity.value if item.max_severity else "-"
        typer.echo(
            f"{item.run_id}  {item.started_at:%Y-%m-%d %H:%M}  {item.dataset_name}  "
            f"{item.n_rows:,}x{item.n_columns}  {item.n_findings} findings  {severity}"
        )
    shown = len(page.items)
    typer.echo(f"{shown} of {page.total} runs" if page.total else "no stored runs")


@runs_app.command("show")
def show(
    run: Annotated[str, typer.Argument(help="A run id or a run.json path.")],
    as_json: AsJson = False,
) -> None:
    """A run's health summary and policy verdict."""
    loaded = _load(run)
    if as_json:
        _echo_json(
            {
                "run_id": str(loaded.run_id),
                "dataset": loaded.dataset.model_dump(mode="json"),
                "health": loaded.health.model_dump(mode="json"),
                "policy": loaded.policy.model_dump(mode="json")
                if loaded.policy
                else None,
            }
        )
        return
    typer.echo(loaded.render_health())
    if loaded.policy is not None:
        typer.echo(loaded.policy.render())


@runs_app.command("diff")
def diff(
    base: Annotated[str, typer.Argument(help="The earlier run: an id or run.json.")],
    head: Annotated[str, typer.Argument(help="The later run: an id or run.json.")],
    tolerance: Annotated[
        float,
        typer.Option(
            min=0, help="Relative change a metric must exceed to count as changed."
        ),
    ] = DEFAULT_TOLERANCE,
    as_json: AsJson = False,
) -> None:
    """What the head run shows that the base run did not."""
    result = compare(_load(base), _load(head), tolerance=tolerance)
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(result.render())


@runs_app.command("delete")
def delete(
    run: Annotated[str, typer.Argument(help="The run id.")],
) -> None:
    """Remove a run and its stored files."""
    from dsbf.store import StoreError  # noqa: PLC0415

    try:
        run_id = UUID(run)
    except ValueError:
        _fail(f"{run!r} is not a run id")
    repository = _repository()
    try:
        repository.delete(run_id)
    except StoreError as error:
        _fail(error)
    finally:
        repository.close()
    typer.echo(f"deleted {run_id}")


@runs_app.command("gc")
def gc(
    keep: Annotated[
        int | None,
        typer.Option(min=0, help="Keep the newest N runs of each dataset."),
    ] = None,
    older_than: Annotated[
        float | None,
        typer.Option(
            "--older-than",
            min=0,
            help="Keep the runs younger than this many days.",
        ),
    ] = None,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Report what would go; remove nothing.")
    ] = False,
    as_json: AsJson = False,
) -> None:
    """Delete runs no rule keeps, and stored files no run owns.

    A run is kept when any given rule keeps it. With neither rule, only orphaned
    files are removed.
    """
    repository = _repository()
    try:
        result = repository.gc(
            keep=keep,
            older_than=None if older_than is None else timedelta(days=older_than),
            dry_run=dry_run,
        )
    finally:
        repository.close()
    if as_json:
        _echo_json(result.model_dump(mode="json"))
        return
    verb = "would remove" if dry_run else "removed"
    typer.echo(
        f"{verb} {len(result.runs)} runs and {len(result.orphans)} orphaned file sets"
    )


@runs_app.command("import")
def import_run(
    path: Annotated[
        Path, typer.Argument(help="A run.json or the directory the json sink wrote.")
    ],
) -> None:
    """Store a run written by the json sink, artifacts included."""
    from dsbf.store import StoreError  # noqa: PLC0415

    run = _load(str(path))
    repository = _repository()
    try:
        repository.save(run)
    except StoreError as error:
        _fail(error)
    finally:
        repository.close()
    typer.echo(f"imported {run.run_id}")
