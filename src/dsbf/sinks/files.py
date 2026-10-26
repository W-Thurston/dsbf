# dsbf/sinks/files.py
"""The JSON and NDJSON sinks, and reading ``run.json`` back."""

from collections.abc import Iterable
from pathlib import Path

from pydantic import ValidationError

from dsbf.model import Run
from dsbf.sinks.registry import RunReadError, Sink, get_sinks, register_sink

RUN_FILE = "run.json"
FINDINGS_FILE = "findings.ndjson"


def _write_json(run: Run, path: Path) -> None:
    path.write_text(run.model_dump_json(indent=2), encoding="utf-8")


def _write_ndjson(run: Run, path: Path) -> None:
    lines = (finding.model_dump_json() + "\n" for finding in run.findings)
    path.write_text("".join(lines), encoding="utf-8")


register_sink(Sink("json", RUN_FILE, _write_json))
register_sink(Sink("ndjson", FINDINGS_FILE, _write_ndjson))


def write_run(
    run: Run, directory: str | Path, sinks: Iterable[str] = ("json",)
) -> dict[str, Path]:
    """Write ``run`` into ``directory`` through each sink; return the file per sink.

    Every sink name is checked before anything is written.
    """
    chosen = get_sinks(tuple(sinks))
    target = Path(directory).expanduser()
    target.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for sink in chosen:
        written[sink.name] = target / sink.filename
        sink.write(run, written[sink.name])
    return written


def read_run(path: str | Path) -> Run:
    """Read a ``run.json``, given its path or the directory that holds it."""
    file = Path(path).expanduser()
    if file.is_dir():
        file = file / RUN_FILE
    try:
        text = file.read_text(encoding="utf-8")
    except OSError as error:
        raise RunReadError(
            f"cannot read run {file}: {error.strerror or error}"
        ) from error
    except UnicodeDecodeError as error:
        raise RunReadError(f"{file} is not UTF-8 text") from error
    try:
        return Run.model_validate_json(text)
    except ValidationError as error:
        first = error.errors()[0]
        where = ".".join(map(str, first["loc"])) or "document"
        raise RunReadError(
            f"{file} is not a readable run ({error.error_count()} problem(s); "
            f"first at {where}: {first['msg']})"
        ) from error
