import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from dsbf.config import load_default_config
from dsbf.eda.profile_engine import ProfileEngine

# Exactly one of these tiers per test; slow and network are optional modifiers.
PRIMARY_TIERS = ("unit", "integration", "golden", "perf")

# Runtime state the suite must never share with a developer's own runs. Each
# variable is forced for the whole session, whatever the caller exported.
_ISOLATED_ENV_KEYS = ("DSBF_HOME", "DSBF_DB_PATH")
_RUNTIME_ROOT_KEY = pytest.StashKey[Path]()
_SAVED_ENV_KEY = pytest.StashKey[dict[str, str | None]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite the snapshots in tests/golden/snapshots instead of comparing.",
    )
    parser.addoption(
        "--perf-results",
        default="perf-results/perf.json",
        help="Where `-m perf` writes its results JSON (from the invocation directory).",
    )
    parser.addoption(
        "--update-perf-baseline",
        action="store_true",
        default=False,
        help="Merge the measured scenarios into tests/perf/baseline.json.",
    )
    parser.addoption(
        "--perf-no-compare",
        action="store_true",
        default=False,
        help="Measure and write results without failing on a regression.",
    )
    parser.addoption(
        "--perf-cprofile",
        action="store_true",
        default=False,
        help="Also run each perf scenario under cProfile (slower; for diagnosis only).",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Point the DSBF home and the database at a throwaway root.

    This runs at configure time, not in a fixture, so the paths are in place before
    any test module is collected or imported.
    """
    root = Path(tempfile.mkdtemp(prefix="dsbf-tests-")).resolve()
    (root / "cwd").mkdir()
    config.stash[_RUNTIME_ROOT_KEY] = root
    config.stash[_SAVED_ENV_KEY] = {k: os.environ.get(k) for k in _ISOLATED_ENV_KEYS}
    os.environ["DSBF_HOME"] = str(root / "home")
    os.environ["DSBF_DB_PATH"] = str(root / "dsbf.db")


def pytest_unconfigure(config: pytest.Config) -> None:
    saved = config.stash.get(_SAVED_ENV_KEY, {})
    for key, previous in saved.items():
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous
    root = config.stash.get(_RUNTIME_ROOT_KEY, None)
    if root is not None:
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(scope="session", autouse=True)
def isolated_working_directory(pytestconfig: pytest.Config) -> Iterator[Path]:
    """Run the whole session from a temporary directory.

    DSBF writes under ``DSBF_HOME``, which the session points at a temporary root.
    The temporary CWD keeps any stray relative write, and any code that wrongly
    depends on the repository layout, from touching the repository.
    """
    cwd = pytestconfig.stash[_RUNTIME_ROOT_KEY] / "cwd"
    previous = Path.cwd()
    os.chdir(cwd)
    try:
        yield cwd
    finally:
        os.chdir(previous)


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Fail collection for any test without exactly one primary tier marker.

    Runs before ``-m`` deselects, so the check sees every collected test.
    """
    problems: list[str] = []
    for item in items:
        tiers = [t for t in PRIMARY_TIERS if item.get_closest_marker(t)]
        if len(tiers) != 1:
            found = ", ".join(tiers) if tiers else "none"
            problems.append(f"  {item.nodeid} (tiers: {found})")
    if problems:
        raise pytest.UsageError(
            f"Every test needs exactly one of {', '.join(PRIMARY_TIERS)} "
            "(see markers in pyproject.toml):\n" + "\n".join(problems)
        )


@pytest.fixture
def clean_engine_run(tmp_path):
    """Return a callable that runs the engine on titanic into ``tmp_path``.

    The engine honours ``output_dir`` (top-level or under ``metadata``); it ignores
    ``engine.output_path``. The database and any CWD-relative writes are already
    isolated for the whole session, so there is nothing to clean up afterwards.
    """

    def run_engine_with_temp_config(extra_config=None) -> Path:
        config: dict[str, Any] = load_default_config()
        # Override the sentinel dataset name with a real seaborn dataset.
        # default_config.yaml uses "default_dataset_name" deliberately so
        # callers that forget to set a dataset fail loudly — conftest must
        # always supply a real one.
        config.setdefault("metadata", {})["dataset_name"] = "titanic"
        config.setdefault("metadata", {})["dataset_source"] = "seaborn"
        if extra_config:
            config.update(extra_config)
        config["output_dir"] = str(tmp_path)

        engine = ProfileEngine(config=config)
        engine.run()

        report_path: Path = Path(engine.output_dir) / "report.json"
        assert report_path.exists(), f"Report not found at {report_path}"
        return report_path

    return run_engine_with_temp_config


@pytest.fixture
def minimal_valid_config() -> dict[
    str, dict[str, bool] | dict[str, dict] | dict[str, str]
]:
    """A minimal config with one valid task and strict mode on."""
    return {
        "tasks": {"dummy_task": {}},
        "safety": {"strict_mode": True},
        "metadata": {"profiling_depth": "basic"},
    }


@pytest.fixture
def config_with_unknown_task() -> dict[str, dict[str, bool] | dict[str, dict]]:
    """Includes a task that does not exist in the registry."""
    return {
        "tasks": {"not_a_real_task": {}},
        "safety": {"strict_mode": False},
    }


@pytest.fixture
def config_with_cycle() -> dict[str, dict[str, bool] | dict[str, dict]]:
    """
    Tasks 'a' and 'b' will be registered in the test with cyclic deps.
    This config sets strict mode to true to test validation error.
    """
    return {
        "tasks": {"a": {}, "b": {}},
        "safety": {"strict_mode": True},
    }


@pytest.fixture
def config_with_schema_validation() -> dict[
    str, dict[str, bool | dict[str, list] | str] | dict[str, dict] | dict[str, str]
]:
    """Enables schema validation with one required column."""
    return {
        "schema_validation": {
            "enable_schema_validation": True,
            "fail_or_warn": "warn",
            "schema": {"required_columns": ["A"]},
        },
        "tasks": {"schema_validation": {}},
        "metadata": {"dataset_name": "custom"},
    }


@pytest.fixture
def config_with_strict_plugin_failure() -> dict[str, dict[str, bool] | dict]:
    """Tests strict-mode behavior when plugin warnings are present."""
    return {
        "tasks": {},
        "safety": {"strict_mode": True},
    }
