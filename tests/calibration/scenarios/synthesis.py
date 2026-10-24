"""Tables whose flaws should read as a given level on the quality traffic light.

The levels follow one rule, written here so a threshold can be judged against it:

- green: nothing found;
- blue: an issue worth knowing about that needs no action, such as a few columns with
  a little missingness;
- amber: something to act on before relying on the table, such as a warning on a
  minority of columns;
- red: do not rely on it: an error, or a warning on most of the columns.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from dsbf.model import CheckResult
from tests.calibration.scenario import scenario
from tests.calibration.scoring import Planted, Probe, plant


def level_is(dimension: str, *expected: str) -> Callable[[CheckResult, Planted], float]:
    """1 when the quality level of ``dimension`` is one of ``expected``, else 0."""

    def measure(result: CheckResult, planted: Planted) -> float:
        assert result.output is not None
        found = result.output["dimensions"][dimension]["level"]
        return 1.0 if found in expected else 0.0

    return measure


def _nulls(rng: np.random.Generator, n: int, share: float) -> np.ndarray:
    values = rng.normal(0, 1, n)
    values[rng.random(n) < share] = np.nan
    return values


def _table(
    rng: np.random.Generator, columns: int, flawed: int, share: float, n: int = 1000
) -> pd.DataFrame:
    """``columns`` normal columns, the first ``flawed`` of them ``share`` null."""
    return pd.DataFrame(
        {
            f"col_{i:02d}": _nulls(rng, n, share if i < flawed else 0.0)
            for i in range(columns)
        }
    )


def _expect(frame: pd.DataFrame, level: str, name: str, flawed: int) -> Planted:
    return Planted(
        frame,
        # The nulls are the point of the table, so `null_rate` reporting them is right.
        allowed=tuple(plant("null_rate", f"col_{i:02d}") for i in range(flawed)),
        probes=(
            Probe(
                "quality_dimensions",
                name,
                level_is("completeness", level),
            ),
        ),
    )


@scenario("quality_green", seeds=3)
def quality_green(seed: int) -> Planted:
    """Twelve complete columns."""
    return _expect(_table(np.random.default_rng(seed), 12, 0, 0.0), "green", "level", 0)


@scenario("quality_warning_one_of_two", seeds=3)
def quality_warning_one_of_two(seed: int) -> Planted:
    """Two columns, one 30% null."""
    return _expect(_table(np.random.default_rng(seed), 2, 1, 0.3), "amber", "level", 1)


@scenario("quality_warning_one_of_twenty", seeds=3)
def quality_warning_one_of_twenty(seed: int) -> Planted:
    """Twenty columns, one 30% null."""
    return _expect(_table(np.random.default_rng(seed), 20, 1, 0.3), "amber", "level", 1)


@scenario("quality_warning_most", seeds=3)
def quality_warning_most(seed: int) -> Planted:
    """Ten columns, seven of them 30% null."""
    return _expect(_table(np.random.default_rng(seed), 10, 7, 0.3), "red", "level", 7)


@scenario("quality_error_one_of_twenty", seeds=3)
def quality_error_one_of_twenty(seed: int) -> Planted:
    """Twenty columns, one 95% null."""
    return _expect(_table(np.random.default_rng(seed), 20, 1, 0.95), "red", "level", 1)


@scenario("quality_notice_one_of_twenty", seeds=3)
def quality_notice_one_of_twenty(seed: int) -> Planted:
    """Twenty columns, one 8% null."""
    return _expect(_table(np.random.default_rng(seed), 20, 1, 0.08), "blue", "level", 1)


@scenario("quality_notice_three_of_ten", seeds=3)
def quality_notice_three_of_ten(seed: int) -> Planted:
    """Ten columns, three 8% null."""
    return _expect(_table(np.random.default_rng(seed), 10, 3, 0.08), "blue", "level", 3)


@scenario("quality_notice_most", seeds=3)
def quality_notice_most(seed: int) -> Planted:
    """Ten columns, seven 8% null: a pattern, but a mild one."""
    return _expect(
        _table(np.random.default_rng(seed), 10, 7, 0.08), "amber", "level", 7
    )
