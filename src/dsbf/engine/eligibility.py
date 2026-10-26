# dsbf/engine/eligibility.py
"""Which columns a check can read from a profile, or why it has nothing to run on."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from dsbf.engine.check import CheckSpec
from dsbf.model import ColumnProfile, DatasetProfile

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
    "content_hash": lambda column: column.content_hash is not None,
}
_DATASET_FEATURES: dict[str, Callable[[DatasetProfile], bool]] = {
    "missingness": lambda profile: profile.missingness is not None,
    "duplicate_rows": lambda profile: profile.duplicate_rows is not None,
}


@dataclass(frozen=True)
class Eligibility:
    """What a check can read from a profile.

    ``columns`` are the columns it reads: None for a check that reads none, empty
    when it has no column to read. ``reason`` is set, with a machine-readable
    ``code`` (and a ``hint`` when a setting would help), when the profile leaves the
    check nothing to run on.
    """

    columns: tuple[str, ...] | None
    reason: str | None = None
    code: str | None = None
    hint: str | None = None

    @property
    def applicable(self) -> bool:
        return self.reason is None


def _unset(value: Any) -> bool:
    return value is None or (hasattr(value, "__len__") and len(value) == 0)


def eligibility(
    spec: CheckSpec,
    profile: DatasetProfile,
    *,
    has_reference: bool = False,
    params: Mapping[str, Any] | None = None,
) -> Eligibility:
    """The columns ``spec`` reads from ``profile``, or why it has none.

    A column is eligible when it is of a type the check applies to and the profile
    holds the column features the check requires. A dataset-level feature the profile
    lacks, or a reference the check requires and the run was not given, makes the
    whole check not applicable, as is a parameter the check requires (``params``)
    that is unset. A synthesis check reads every column's profile, so it
    can give advice about a column from other checks' results; a check about the
    dataset as a whole (``applies_to=None``, any other kind) reads none.
    """
    if spec.requires.reference and not has_reference:
        return Eligibility(
            None,
            reason=f"No reference dataset was given, which {spec.id} compares with.",
            code="no_reference",
        )
    for name in spec.requires.params:
        if _unset((params or {}).get(name)):
            return Eligibility(
                None,
                reason=f"The `{name}` parameter is not set, which {spec.id} requires.",
                code=f"no_{name}",
                hint=(
                    f"Set the `{name}` parameter: `--param {spec.id}.{name}=...` on "
                    f"the command line, or `params={{'{spec.id}': {{'{name}': ...}}}}` "
                    "to dsbf.profile()."
                ),
            )
    for feature in spec.requires.profile:
        if feature in _DATASET_FEATURES and not _DATASET_FEATURES[feature](profile):
            return Eligibility(
                None,
                reason=f"The profile has no {feature}, which {spec.id} requires.",
                code="profile_feature_missing",
            )
    if spec.applies_to is None:
        if spec.kind == "synthesis":
            return Eligibility(tuple(c.name for c in profile.columns))
        return Eligibility(None)

    of_type = [c for c in profile.columns if spec.applies_to.selects(c.semantic_type)]
    wanted = [f for f in spec.requires.profile if f in _COLUMN_FEATURES]
    columns = tuple(
        c.name for c in of_type if all(_COLUMN_FEATURES[f](c) for f in wanted)
    )
    if columns:
        return Eligibility(columns)
    if of_type:
        return Eligibility(
            (),
            reason=(
                f"No {spec.applies_to.describe()} column has "
                f"{', '.join(wanted)} in the profile, which {spec.id} requires."
            ),
            code="no_eligible_columns",
        )
    return Eligibility(
        (),
        reason=(
            f"No column is of a type {spec.id} applies to "
            f"({spec.applies_to.describe()})."
        ),
        code="no_eligible_columns",
    )
