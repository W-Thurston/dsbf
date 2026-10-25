# tests/test_extras.py
"""The core runs without an extra and reports the checks that need it as unavailable.

The first group hides an installed extra from the importer, so a module-level import
of it in the core would fail here. The second group reads the environment as it is,
which the CI install matrix varies one extra at a time.
"""

import sys
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

import dsbf
from dsbf.checks import core_pack
from dsbf.engine import executor
from dsbf.engine.check import Extra
from dsbf.engine.events import CheckStarted
from dsbf.engine.extras import (
    EXTRA_MODULES,
    PRELOAD,
    install_hint,
    missing_modules,
)
from dsbf.model import CheckStatus

pytestmark = pytest.mark.integration

# The extras that gate a check; the others gate a source or nothing yet.
CHECK_EXTRAS: tuple[Extra, ...] = ("ml", "timeseries", "text")

NEEDING = {
    extra: sorted(
        check.id for check in core_pack().checks if extra in check.spec.extras
    )
    for extra in CHECK_EXTRAS
}


def small_table() -> pl.DataFrame:
    rng = np.random.default_rng(11)
    n = 240
    start = datetime(2024, 1, 1)
    return pl.DataFrame(
        {
            "when": [start + timedelta(hours=i) for i in range(n)],
            "x": rng.normal(size=n),
            "y": rng.normal(size=n).cumsum(),
            "kind": rng.choice(["a", "b", "c"], size=n).tolist(),
            "label": [f"record number {i % 7}" for i in range(n)],
            "target": rng.integers(0, 2, size=n),
        }
    )


def unavailable_ids(run) -> set[str]:
    return {
        check_id
        for check_id, result in run.results.items()
        if result.status is CheckStatus.UNAVAILABLE
    }


def needing_a_missing_extra() -> set[str]:
    """The checks that need an extra this process cannot import."""
    return {
        check_id
        for extra in CHECK_EXTRAS
        if missing_modules(extra)
        for check_id in NEEDING[extra]
    }


def profile_deep(frame: pl.DataFrame):
    return dsbf.profile(frame, depth="deep", target="target", time_index="when")


@pytest.fixture
def hidden(monkeypatch):
    """Make the modules of one extra impossible to import, loaded or not."""

    def hide(extra: Extra) -> None:
        for module in EXTRA_MODULES[extra]:
            monkeypatch.setitem(sys.modules, module, None)
            for loaded in [m for m in sys.modules if m.startswith(f"{module}.")]:
                monkeypatch.setitem(sys.modules, loaded, None)

    return hide


@pytest.mark.parametrize("extra", CHECK_EXTRAS)
def test_every_extra_gates_at_least_one_check(extra):
    assert NEEDING[extra]


@pytest.mark.parametrize("extra", CHECK_EXTRAS)
def test_without_an_extra_its_checks_are_unavailable_and_the_rest_run(extra, hidden):
    hidden(extra)
    assert missing_modules(extra)

    run = profile_deep(small_table())

    unavailable = unavailable_ids(run)
    assert unavailable == needing_a_missing_extra()
    assert set(NEEDING[extra]) <= unavailable
    for check_id in NEEDING[extra]:
        assert run.results[check_id].notes[0].hint == install_hint([extra])
    assert not [
        check_id
        for check_id, result in run.results.items()
        if result.status in (CheckStatus.ERROR, CheckStatus.TIMEOUT)
    ]
    assert run.results["null_rate"].status is CheckStatus.OK


def test_without_every_extra_the_base_install_still_profiles(hidden):
    for extra in CHECK_EXTRAS:
        hidden(extra)

    run = profile_deep(small_table())

    assert unavailable_ids(run) == {c for ids in NEEDING.values() for c in ids}
    assert run.health.counts.ok > 0
    assert run.health.counts.error == 0


@pytest.mark.parametrize("extra", sorted(EXTRA_MODULES))
def test_a_hidden_extra_names_what_is_missing_and_how_to_install_it(extra, hidden):
    hidden(extra)

    assert missing_modules(extra) == EXTRA_MODULES[extra]
    assert install_hint([extra]) == f"Install it with `pip install 'dsbf[{extra}]'`."


@pytest.mark.parametrize("extra", CHECK_EXTRAS)
def test_the_installed_extras_decide_which_checks_run(extra):
    installed = not missing_modules(extra)

    run = profile_deep(small_table())

    for check_id in NEEDING[extra]:
        status = run.results[check_id].status
        if installed:
            assert status in (CheckStatus.OK, CheckStatus.NOT_APPLICABLE), check_id
        else:
            assert status is CheckStatus.UNAVAILABLE, check_id


def test_the_extras_a_plan_needs_are_imported_before_any_check_starts(monkeypatch):
    seen: list[object] = []
    real = executor.preload_extras

    def spy(extras):
        extras = list(extras)
        seen.append(("preload", sorted(set(extras))))
        real(extras)

    monkeypatch.setattr(executor, "preload_extras", spy)

    dsbf.profile(
        small_table(),
        depth="deep",
        target="target",
        time_index="when",
        workers=4,
        hooks=[lambda event: seen.append(event)],
    )

    installed = sorted(extra for extra in CHECK_EXTRAS if not missing_modules(extra))
    assert seen[0] == ("preload", installed)
    assert isinstance(seen[1], CheckStarted)


@pytest.mark.parametrize("extra", sorted(PRELOAD))
def test_preloading_an_extra_imports_every_module_it_lists(extra, monkeypatch):
    loaded: list[str] = []
    monkeypatch.setattr(
        "dsbf.engine.extras.import_module", lambda name: loaded.append(name)
    )

    executor.preload_extras([extra, extra])

    expected = [] if missing_modules(extra) else list(PRELOAD[extra])
    assert loaded == expected
