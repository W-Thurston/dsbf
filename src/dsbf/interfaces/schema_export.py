# dsbf/interfaces/schema_export.py
"""Write, verify and compare the published schema files (behind ``dsbf schema``)."""

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from dsbf.model.schema import (
    Change,
    compare_schemas,
    run_schema,
    schema_text,
    schema_version,
    typescript_schema,
    unannounced,
)

SCHEMA_FILE = "run.schema.json"
TYPESCRIPT_BANNER = (
    "/* eslint-disable */\n"
    "/**\n"
    " * Generated from schema/run.schema.json by `dsbf schema export`.\n"
    " * Do not edit by hand: change the models in dsbf.model and regenerate.\n"
    " */"
)


class SchemaToolError(Exception):
    """A schema command couldn't do its job; the message says what to fix."""


@dataclass(frozen=True)
class SchemaComparison:
    """What changed against a base revision and whether the version announces it."""

    base_version: str
    version: str
    changes: list[Change]
    problems: list[str]


def generate_typescript(schema_json: str, frontend_dir: Path) -> str:
    """Render the TypeScript types for a schema with the frontend's generator."""
    pnpm = shutil.which("pnpm")
    if pnpm is None:
        raise SchemaToolError(
            "TypeScript types need pnpm (and Node, see .nvmrc); install them, then "
            f"run `pnpm install` in {frontend_dir}"
        )
    if not (frontend_dir / "node_modules").is_dir():
        raise SchemaToolError(f"run `pnpm install` in {frontend_dir} first")
    with tempfile.TemporaryDirectory() as scratch:
        schema_file = Path(scratch) / SCHEMA_FILE
        schema_file.write_text(
            json.dumps(typescript_schema(json.loads(schema_json))), encoding="utf-8"
        )
        command = [
            pnpm,
            "--dir",
            str(frontend_dir),
            "exec",
            "json2ts",
            "--unreachableDefinitions",
            "--input",
            str(schema_file),
            "--bannerComment",
            TYPESCRIPT_BANNER,
        ]
        done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise SchemaToolError(f"json2ts failed:\n{done.stderr.strip()}")
    return done.stdout


def export_schema(out_dir: Path, *, check: bool) -> tuple[Path, str]:
    """The schema file's path and fresh text; written unless ``check`` is set."""
    path = out_dir / SCHEMA_FILE
    text = schema_text(run_schema())
    if not check:
        out_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return path, text


def stale_files(expected: dict[Path, str]) -> list[Path]:
    """The files whose content on disk is not the expected text."""
    return [
        path
        for path, text in expected.items()
        if not path.is_file() or path.read_text(encoding="utf-8") != text
    ]


def schema_at_ref(ref: str, path: Path) -> str | None:
    """The schema file as committed at ``ref``; None if the file didn't exist."""
    verify = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    if verify.returncode != 0:
        raise SchemaToolError(
            f"{ref!r} is not a git revision here (a CI checkout needs the base branch "
            "fetched, for example `fetch-depth: 0`)"
        )
    shown = subprocess.run(
        ["git", "show", f"{ref}:./{path.as_posix()}"],
        capture_output=True,
        text=True,
        check=False,
    )
    return shown.stdout if shown.returncode == 0 else None


def compare_with_base(ref: str, schema_file: Path) -> SchemaComparison | None:
    """Compare the schema file with its committed version at ``ref``.

    None means the base has no schema yet, so there is nothing to break.
    """
    if not schema_file.is_file():
        raise SchemaToolError(f"{schema_file} doesn't exist; run `dsbf schema export`")
    base_text = schema_at_ref(ref, schema_file)
    if base_text is None:
        return None
    old, new = json.loads(base_text), json.loads(schema_file.read_text("utf-8"))
    changes = compare_schemas(old, new)
    old_version, new_version = schema_version(old), schema_version(new)
    problems = unannounced(old_version, new_version, changes)
    return SchemaComparison(old_version, new_version, changes, problems)
