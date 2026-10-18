# dsbf/engine/expectation.py
"""What a plan expects of each check, judged before anything runs."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from dsbf.engine.check import Check
from dsbf.engine.extras import install_hint, missing_extras
from dsbf.model import ColumnProfile, DatasetProfile


class Expected(StrEnum):
    """How a check is expected to end, with the reason in its ``Expectation``."""

    WILL_RUN = "will_run"
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE = "unavailable"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class Expectation:
    """The expected status of one planned check, and why.

    ``columns`` are the columns the check would read, or None when the plan was made
    without a profile (or the check reads none).
    """

    status: Expected
    reason: str | None = None
    hint: str | None = None
    columns: tuple[str, ...] | None = None
    blocked_by: tuple[str, ...] = ()


# The profile features a column can lack: a constant column has no histogram, an
# all-null one no stats. "counts" is always there, so it needs no entry.
_COLUMN_FEATURES: dict[str, Callable[[ColumnProfile], bool]] = {
    "numeric_stats": lambda column: getattr(column, "stats", None) is not None,
    "quantiles": lambda column: bool(
        getattr(getattr(column, "stats", None), "quantiles", ())
    ),
    "histogram": lambda column: getattr(column, "histogram", None) is not None,
    "top_values": lambda column: bool(getattr(column, "top_values", ())),
    "text_shape": lambda column: getattr(column, "shape", None) is not None,
    "datetime": lambda column: getattr(column, "min", None) is not None,
}
_DATASET_FEATURES: dict[str, Callable[[DatasetProfile], bool]] = {
    "missingness": lambda profile: profile.missingness is not None,
    "duplicate_rows": lambda profile: profile.duplicate_rows is not None,
}


def expect(
    check: Check,
    profile: DatasetProfile | None,
    upstream: Mapping[str, Expectation],
) -> Expectation:
    """The status ``check`` is expected to end with.

    A missing extra comes first, then a hard requirement that cannot run (``upstream``
    holds the expectations of the checks already judged), then a profile with
    nothing for the check to read. Without a ``profile`` only the first two are
    judged.
    """
    spec = check.spec
    missing = missing_extras(spec.extras)
    if missing:
        modules = ", ".join(m for modules in missing.values() for m in modules)
        names = ", ".join(repr(extra) for extra in missing)
        return Expectation(
            Expected.UNAVAILABLE,
            reason=f"Needs {modules}, which is not installed ({names} extra).",
            hint=install_hint(missing),
        )
    blockers = {
        needed: upstream[needed].status
        for needed in spec.requires.checks
        if needed in upstream
        and upstream[needed].status in (Expected.UNAVAILABLE, Expected.SKIPPED)
    }
    if blockers:
        listed = ", ".join(f"{name!r} ({status})" for name, status in blockers.items())
        return Expectation(
            Expected.SKIPPED,
            reason=f"Requires {listed}, which cannot run.",
            blocked_by=tuple(blockers),
        )
    if profile is None:
        return Expectation(Expected.WILL_RUN)
    return _from_profile(check, profile)


def _from_profile(check: Check, profile: DatasetProfile) -> Expectation:
    spec = check.spec
    for feature in spec.requires.profile:
        if feature in _DATASET_FEATURES and not _DATASET_FEATURES[feature](profile):
            return Expectation(
                Expected.NOT_APPLICABLE,
                reason=f"The profile has no {feature}, which {spec.id} requires.",
            )
    if spec.applies_to is None:
        return Expectation(Expected.WILL_RUN)

    of_type = [c for c in profile.columns if spec.applies_to.selects(c.semantic_type)]
    wanted = [f for f in spec.requires.profile if f in _COLUMN_FEATURES]
    columns = tuple(
        c.name for c in of_type if all(_COLUMN_FEATURES[f](c) for f in wanted)
    )
    if columns:
        return Expectation(Expected.WILL_RUN, columns=columns)
    if of_type:
        return Expectation(
            Expected.NOT_APPLICABLE,
            reason=(
                f"No {spec.applies_to.describe()} column has "
                f"{', '.join(wanted)} in the profile, which {spec.id} requires."
            ),
            columns=(),
        )
    return Expectation(
        Expected.NOT_APPLICABLE,
        reason=(
            f"No column is of a type {spec.id} applies to "
            f"({spec.applies_to.describe()})."
        ),
        columns=(),
    )
