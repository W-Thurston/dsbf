import os
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest

# Exactly one of these tiers per test; slow and network are optional modifiers.
PRIMARY_TIERS = ("unit", "integration", "golden", "perf", "calibration")

# Runtime state the suite must never share with a developer's own runs. Each
# variable is forced for the whole session, whatever the caller exported.
_ISOLATED_ENV_KEYS = ("DSBF_HOME",)
_RUNTIME_ROOT_KEY = pytest.StashKey[Path]()
_SAVED_ENV_KEY = pytest.StashKey[dict[str, str | None]]()


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Rewrite the committed snapshots (golden runs, `dsbf plan`) "
        "instead of comparing.",
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
        "--calibration-results",
        default="calibration-results/calibration.json",
        help="Where `-m calibration` writes its results JSON (from the invocation "
        "directory).",
    )
    parser.addoption(
        "--update-calibration-baseline",
        action="store_true",
        default=False,
        help="Merge the measured rates into tests/calibration/baseline.json.",
    )
    parser.addoption(
        "--perf-no-compare",
        action="store_true",
        default=False,
        help="Measure and write results without failing on a regression.",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Point the DSBF home at a throwaway root.

    This runs at configure time, not in a fixture, so the paths are in place before
    any test module is collected or imported.
    """
    root = Path(tempfile.mkdtemp(prefix="dsbf-tests-")).resolve()
    (root / "cwd").mkdir()
    config.stash[_RUNTIME_ROOT_KEY] = root
    config.stash[_SAVED_ENV_KEY] = {k: os.environ.get(k) for k in _ISOLATED_ENV_KEYS}
    os.environ["DSBF_HOME"] = str(root / "home")


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
