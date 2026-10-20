"""The differences between a core check and its legacy task that DSBF accepts.

Each entry names the check, the golden dataset, the column (or none), the kind of
difference and its exact detail, and gives the reason the core check is right. A
difference that is not listed fails ``test_migration_parity``, and so does a listed one
that no longer occurs. Reasons are written by hand.
"""

from __future__ import annotations

from tests.golden.migration.harness import Accepted

CODED_COLUMN = (
    "The profile types an integer column categorical when it has two values, or at "
    "most 20 distinct values under 5% of the rows, so it is a label or a small count "
    "code. `detect_zeros` read every numeric dtype and called its zeros a "
    "concentration; `zeros` reads only numeric columns, where a zero is a "
    "measurement, so a zero in a code is a category and not a finding."
)
CONSTANT_COLUMN = (
    "A column with one value is typed categorical, not numeric, so `zeros` doesn't "
    "read it; constant columns are reported by their own check."
)
FEW_VALUES_COLUMN = (
    "A float column of 11 distinct values under 5% of the rows is typed categorical, "
    "so `zeros` doesn't read it. It has no zeros, so nothing is lost here."
)


def _legacy_only(
    dataset: str, column: str, reason: str, finding: str | None = None
) -> list[Accepted]:
    """The task read ``column`` (and raised a finding about it); the check does not."""
    entries = [Accepted("zeros", dataset, column, "column_read", "legacy only", reason)]
    if finding:
        detail = f"legacy only ({finding})"
        entries.append(Accepted("zeros", dataset, column, "finding", detail, reason))
    return entries


ACCEPTED: tuple[Accepted, ...] = (
    *_legacy_only("titanic", "survived", CODED_COLUMN, "warn"),
    *_legacy_only("titanic", "pclass", CODED_COLUMN),
    *_legacy_only("titanic", "sibsp", CODED_COLUMN, "warn"),
    *_legacy_only("titanic", "parch", CODED_COLUMN, "warn"),
    *_legacy_only("kitchen5k", "balanced_target", CODED_COLUMN, "info"),
    *_legacy_only("kitchen5k", "imbalanced_target", CODED_COLUMN, "warn"),
    *_legacy_only("kitchen5k", "mostly_zeros", CODED_COLUMN, "warn"),
    *_legacy_only("kitchen5k", "multiclass_target", CODED_COLUMN, "warn"),
    *_legacy_only("kitchen5k", "perfect_leakage", CODED_COLUMN, "warn"),
    *_legacy_only("kitchen5k", "target_for_leakage", CODED_COLUMN, "warn"),
    *_legacy_only("kitchen5k", "constant_numeric", CONSTANT_COLUMN),
    *_legacy_only("kitchen5k", "near_zero_variance", FEW_VALUES_COLUMN),
)
