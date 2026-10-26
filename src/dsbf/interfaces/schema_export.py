# dsbf/interfaces/schema_export.py
"""Write, verify and compare the published schema files (behind ``dsbf schema``)."""

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dsbf.checks import core_pack
from dsbf.engine import Registry
from dsbf.model.schema import (
    JSON_SCHEMA_DIALECT,
    Change,
    compare_schemas,
    model_schema,
    policy_schema,
    run_schema,
    schema_text,
    schema_version,
    typescript_schema,
    unannounced,
)

SCHEMA_FILE = "run.schema.json"
POLICY_SCHEMA_FILE = "policy.schema.json"
CHECKS_DIR = "checks"
CHECK_SCHEMA_SUFFIX = ".schema.json"
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


def check_schemas() -> dict[str, dict[str, Any]]:
    """Each core check's parameters and output as JSON Schema, keyed by check id.

    The document names the check and its version, which is what a change to either
    schema has to announce (``compare_check_schemas``).
    """
    registry = Registry()
    registry.register_pack(core_pack())
    schemas: dict[str, dict[str, Any]] = {}
    for check in sorted(registry, key=lambda c: c.spec.id):
        spec = check.spec
        schemas[spec.id] = {
            "$schema": JSON_SCHEMA_DIALECT,
            "title": spec.id,
            "check_id": spec.id,
            "check_version": spec.version,
            "params": model_schema(spec.params, "validation"),
            "output": (
                None
                if spec.output is None
                else model_schema(spec.output, "serialization")
            ),
        }
    return schemas


def check_schema_path(out_dir: Path, check_id: str) -> Path:
    return out_dir / CHECKS_DIR / f"{check_id}{CHECK_SCHEMA_SUFFIX}"


def export_schema(out_dir: Path, *, check: bool) -> dict[Path, str]:
    """The JSON Schema files and their fresh text; written unless ``check`` is set."""
    files = {
        out_dir / SCHEMA_FILE: schema_text(run_schema()),
        out_dir / POLICY_SCHEMA_FILE: schema_text(policy_schema()),
    }
    for check_id, schema in check_schemas().items():
        files[check_schema_path(out_dir, check_id)] = schema_text(schema)
    if not check:
        for path, text in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        for path in orphaned_check_schemas(out_dir, files):
            path.unlink()
    return files


def orphaned_check_schemas(out_dir: Path, expected: dict[Path, str]) -> list[Path]:
    """Check schema files on disk that no core check exports any more."""
    directory = out_dir / CHECKS_DIR
    if not directory.is_dir():
        return []
    found = directory.glob(f"*{CHECK_SCHEMA_SUFFIX}")
    return sorted(path for path in found if path not in expected)


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


def compare_check_schemas(
    ref: str, out_dir: Path
) -> list[tuple[str, SchemaComparison]]:
    """Compare each check's schema file with its committed version at ``ref``.

    A check's params and output are compared as two schemas under the check's
    version; a check whose file is gone is a breaking change that needs a major
    version, so it is reported against the version it had. A base without the
    check schemas yields nothing.
    """
    directory = out_dir / CHECKS_DIR
    comparisons: list[tuple[str, SchemaComparison]] = []
    for check_id, base_text in base_check_schemas(ref, directory).items():
        path = directory / f"{check_id}{CHECK_SCHEMA_SUFFIX}"
        old = json.loads(base_text)
        if not path.is_file():
            change = Change("#", True, "check removed")
            version = old["check_version"]
            problems = unannounced(version, version, [change])
            comparisons.append(
                (check_id, SchemaComparison(version, version, [change], problems))
            )
            continue
        new = json.loads(path.read_text("utf-8"))
        changes: list[Change] = []
        for part in ("params", "output"):
            changes += [
                Change(f"{part}{c.path[1:]}", c.breaking, c.message)
                for c in compare_schemas(old[part] or {}, new[part] or {})
            ]
        problems = unannounced(old["check_version"], new["check_version"], changes)
        comparisons.append(
            (
                check_id,
                SchemaComparison(
                    old["check_version"], new["check_version"], changes, problems
                ),
            )
        )
    return comparisons


def base_check_schemas(ref: str, directory: Path) -> dict[str, str]:
    """The check schema files committed at ``ref``, by check id."""
    schema_at_ref(ref, directory / "_")  # raises if ``ref`` is not a revision
    listed = subprocess.run(
        ["git", "ls-tree", "--name-only", f"{ref}:./{directory.as_posix()}/"],
        capture_output=True,
        text=True,
        check=False,
    )
    if listed.returncode != 0:
        return {}
    texts: dict[str, str] = {}
    for name in listed.stdout.split():
        if name.endswith(CHECK_SCHEMA_SUFFIX):
            text = schema_at_ref(ref, directory / name)
            if text is not None:
                texts[name.removesuffix(CHECK_SCHEMA_SUFFIX)] = text
    return texts
