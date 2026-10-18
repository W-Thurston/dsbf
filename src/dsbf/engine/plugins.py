# dsbf/engine/plugins.py
"""Finding checks and packs outside the core: entry points and local directories.

Nothing here registers anything. Each function returns what a source contributes,
and the registry decides what to keep.
"""

import hashlib
import importlib
import importlib.machinery
import importlib.util
import re
import sys
import traceback
from collections.abc import Iterable
from dataclasses import dataclass, field
from importlib.metadata import EntryPoint
from importlib.metadata import entry_points as installed_entry_points
from pathlib import Path

from dsbf.engine.check import Check
from dsbf.engine.pack import Pack
from dsbf.model import Note, Severity

CHECKS_GROUP = "dsbf.checks"
PACKS_GROUP = "dsbf.packs"

Contribution = Check | Pack


@dataclass(frozen=True)
class Discovery:
    """What one source contributed, and what went wrong reading it."""

    contributions: tuple[Contribution, ...] = ()
    notes: tuple[Note, ...] = field(default=())


def _failure(source: str, error: BaseException) -> Note:
    return Note(
        code="plugin_load_failed",
        message=f"{source} failed to load: {type(error).__name__}: {error}",
        severity=Severity.WARN,
        hint="Fix the error in the plugin, or remove it from the plugin list; "
        "its checks were not registered.",
        exception_type=type(error).__name__,
        detail="".join(
            traceback.format_exception(type(error), error, error.__traceback__)
        ),
    )


def _invalid(source: str, problem: str, hint: str) -> Note:
    return Note(
        code="plugin_invalid",
        message=f"{source} {problem}",
        severity=Severity.WARN,
        hint=hint,
    )


def _contributions(namespace: dict[str, object]) -> tuple[Contribution, ...]:
    """The public ``Check`` and ``Pack`` objects of a module, in definition order."""
    found = (
        value
        for name, value in namespace.items()
        if not name.startswith("_") and isinstance(value, Check | Pack)
    )
    return tuple(dict.fromkeys(found))


def package_name(directory: Path) -> str:
    """A module name unique to ``directory``, so two plugin folders never share one."""
    digest = hashlib.sha256(str(directory).encode()).hexdigest()[:8]
    slug = re.sub(r"\W+", "_", directory.name).strip("_").lower() or "plugins"
    return f"_dsbf_plugin_{slug}_{digest}"


def _import_package(directory: Path, name: str) -> None:
    """Make ``directory`` importable as the package ``name``, once per process."""
    if name in sys.modules:
        return
    init = directory / "__init__.py"
    if init.is_file():
        spec = importlib.util.spec_from_file_location(
            name, init, submodule_search_locations=[str(directory)]
        )
    else:
        spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
        spec.submodule_search_locations = [str(directory)]
    if spec is None:
        raise ImportError(f"cannot import {directory} as a package")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        if spec.loader is not None:
            spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name, None)
        raise


def discover_directory(path: str | Path) -> Discovery:
    """Import every public ``*.py`` in a plugin directory and collect its checks.

    The directory becomes a package with a name of its own, so its files can import
    each other as ``from .utils import helper`` and never collide with another
    plugin's ``utils.py``. A module already imported under that name is reused, not
    run again. Files starting with ``_`` are skipped. A file may be a helper that
    declares no checks; a directory that yields none at all is reported.
    """
    directory = Path(path).expanduser().resolve()
    source = f"plugin directory {directory}"
    if not directory.is_dir():
        reason = "does not exist" if not directory.exists() else "is not a directory"
        return Discovery(
            notes=(
                _invalid(
                    source,
                    reason,
                    "Point the plugin list at a directory of check modules.",
                ),
            )
        )
    name = package_name(directory)
    try:
        _import_package(directory, name)
    except Exception as error:
        return Discovery(notes=(_failure(source, error),))

    contributions: list[Contribution] = []
    notes: list[Note] = []
    for file in sorted(directory.glob("*.py")):
        if file.name.startswith("_"):
            continue
        module_name = f"{name}.{file.stem}"
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            sys.modules.pop(module_name, None)
            notes.append(_failure(f"{source}, file {file.name},", error))
            continue
        contributions.extend(_contributions(vars(module)))
    if not contributions and not notes:
        notes.append(
            Note(
                code="plugin_empty",
                message=f"{source} declares no checks.",
                severity=Severity.NOTICE,
                hint="Declare a check with @check at module level in a file whose "
                "name does not start with '_'.",
            )
        )
    return Discovery(tuple(dict.fromkeys(contributions)), tuple(notes))


def entry_points(group: str) -> tuple[EntryPoint, ...]:
    """The installed entry points of ``group``, by name."""
    return tuple(
        sorted(installed_entry_points(group=group), key=lambda point: point.name)
    )


def discover_entry_point(point: EntryPoint, group: str) -> Discovery:
    """Load one entry point.

    ``dsbf.checks`` expects a ``Check`` or an iterable of them; ``dsbf.packs``
    expects a ``Pack``.
    """
    source = f"entry point {group}:{point.name}"
    try:
        loaded = point.load()
    except Exception as error:
        return Discovery(notes=(_failure(source, error),))

    if group == PACKS_GROUP:
        items: tuple[object, ...] = (loaded,)
        expected, kind = Pack, "a Pack"
    else:
        items = (
            tuple(loaded)
            if isinstance(loaded, Iterable) and not isinstance(loaded, Check)
            else (loaded,)
        )
        expected, kind = Check, "a Check or an iterable of Checks"
    wrong = [item for item in items if not isinstance(item, expected)]
    if wrong:
        return Discovery(
            notes=(
                _invalid(
                    source,
                    f"loaded {type(wrong[0]).__name__}, expected {kind}.",
                    "Point the entry point at the object itself, "
                    "such as 'my_plugin.checks:null_rate'.",
                ),
            )
        )
    return Discovery(tuple(items))  # type: ignore[arg-type]
