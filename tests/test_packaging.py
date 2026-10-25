# tests/test_packaging.py
"""The base install and the extras in pyproject.toml match what the code imports."""

import ast
import sys
import tomllib
from collections.abc import Iterator
from importlib.metadata import packages_distributions
from pathlib import Path

import pytest
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from dsbf.engine.extras import EXTRA_MODULES, PRELOAD

pytestmark = pytest.mark.unit

EXTRAS = {"ml", "timeseries", "text", "excel", "pandas", "sql", "server"}
SOURCE = Path(__file__).resolve().parents[1] / "src" / "dsbf"


@pytest.fixture(scope="module")
def project():
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    return tomllib.loads(pyproject.read_text())["project"]


def _names(requirements: list[str]) -> set[str]:
    return {canonicalize_name(Requirement(r).name) for r in requirements}


def _imports(statements: list[ast.stmt], *, module_level: bool) -> Iterator[str]:
    """The top-level package of each third-party import in ``statements``.

    With ``module_level``, imports inside a function or a class are left out, and so is
    a block that only type checkers read.
    """
    for node in statements:
        if isinstance(node, ast.Import):
            modules = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules = [node.module]
        else:
            modules = []
        for module in modules:
            top = module.split(".")[0]
            if top not in sys.stdlib_module_names and top != "dsbf":
                yield top
        nested = [
            child
            for field, value in ast.iter_fields(node)
            if field in ("body", "orelse", "finalbody", "handlers")
            for child in (value if isinstance(value, list) else [value])
            if isinstance(child, ast.stmt | ast.ExceptHandler)
        ]
        skipped = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        if module_level and isinstance(node, skipped):
            continue
        if module_level and _is_type_checking(node):
            continue
        for child in nested:
            body = child.body if isinstance(child, ast.ExceptHandler) else [child]
            yield from _imports(body, module_level=module_level)


def _is_type_checking(node: ast.stmt) -> bool:
    return isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test)


def _packages_imported(*, module_level: bool) -> dict[str, set[str]]:
    """The distribution of each third-party package ``src`` imports, with its files."""
    distributions = packages_distributions()
    found: dict[str, set[str]] = {}
    for path in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for top in _imports(tree.body, module_level=module_level):
            owners = distributions.get(top)
            assert owners, f"{top} (imported by {path.name}) is not installed"
            found.setdefault(canonicalize_name(owners[0]), set()).add(path.name)
    return found


def test_extras_are_the_documented_set_plus_all(project):
    assert set(project["optional-dependencies"]) == EXTRAS | {"all"}


def test_all_is_the_union_of_the_other_extras(project):
    (requirement,) = (Requirement(r) for r in project["optional-dependencies"]["all"])

    assert requirement.name == project["name"]
    assert requirement.extras == EXTRAS


def test_no_extra_repeats_a_base_package(project):
    base = _names(project["dependencies"])
    optional = {
        extra: _names(requirements)
        for extra, requirements in project["optional-dependencies"].items()
        if extra != "all"
    }

    for extra, names in optional.items():
        assert not names & base, f"the {extra!r} extra repeats {names & base}"


def test_a_package_imported_at_module_level_is_in_the_base_install(project):
    base = _names(project["dependencies"])

    missing = {
        package: sorted(files)
        for package, files in _packages_imported(module_level=True).items()
        if package not in base
    }

    assert not missing, (
        f"imported when dsbf loads, so the base install must declare it: {missing}"
    )


def test_every_package_the_code_imports_is_declared(project):
    declared = _names(project["dependencies"]).union(
        *(
            _names(requirements)
            for extra, requirements in project["optional-dependencies"].items()
            if extra != "all"
        )
    )

    missing = {
        package: sorted(files)
        for package, files in _packages_imported(module_level=False).items()
        if package not in declared
    }

    assert not missing, (
        f"imported but declared by neither the base nor an extra: {missing}"
    )


def test_every_base_package_is_imported_by_the_code(project):
    imported = set(_packages_imported(module_level=False))
    not_imported = _names(project["dependencies"]) - imported

    assert not not_imported, (
        f"declared in the base install but never imported: {not_imported}"
    )


def _extra_modules_imported() -> set[str]:
    """The modules of the extras' packages that ``src`` imports.

    ``from rapidfuzz import fuzz`` counts as the module ``rapidfuzz.fuzz``.
    """
    packages = {module for modules in EXTRA_MODULES.values() for module in modules}
    found: set[str] = set()
    for path in sorted(SOURCE.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
                if node.module in packages:
                    names = [f"{node.module}.{alias.name}" for alias in node.names]
            else:
                continue
            found |= {name for name in names if name.split(".")[0] in packages}
    return found


def test_the_preload_table_lists_exactly_the_extra_modules_the_code_imports():
    preloaded = {module for modules in PRELOAD.values() for module in modules}

    assert _extra_modules_imported() == preloaded
