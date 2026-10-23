# dsbf/checks/target.py
"""What the target-aware checks share: the target, a feature, and how they associate.

This module holds no check. ``target_leakage`` and ``mutual_information`` import it,
so both read the target and sample the rows the same way.
"""

from dataclasses import dataclass, replace
from typing import Literal, Self

import numpy as np
from scipy.stats import rankdata

from dsbf.checks.pairs import (
    Undefined,
    contingency_table,
    correlation_ratio,
    cramers_v,
    finite,
    level_codes,
    unit_column,
)
from dsbf.engine import CheckContext, CheckOutput
from dsbf.model import CheckStatus, Severity

Kind = Literal["class", "numeric"]
AssociationMetric = Literal["pearson_r", "spearman_r", "eta", "cramers_v"]

TARGET_TYPES = ("numeric", "categorical", "boolean")


@dataclass(frozen=True)
class Variable:
    """A column over the rows read: class codes, or finite floats with NaN gaps."""

    kind: Kind
    codes: np.ndarray  # class: the level of each row, -1 where missing
    values: np.ndarray  # numeric: float64, NaN where missing or infinite
    n_levels: int

    @property
    def present(self) -> np.ndarray:
        return self.codes >= 0 if self.kind == "class" else ~np.isnan(self.values)

    def take(self, positions: np.ndarray) -> Self:
        """The variable over the rows at ``positions``."""
        return replace(
            self,
            codes=self.codes[positions] if self.codes.size else self.codes,
            values=self.values[positions] if self.values.size else self.values,
        )


@dataclass(frozen=True)
class Target(Variable):
    name: str = ""
    levels: tuple[str, ...] = ()


@dataclass(frozen=True)
class Association:
    """How strongly a feature and the target move together, between 0 and 1."""

    metric: AssociationMetric
    value: float
    n_used: int


_EMPTY = np.empty(0)
_NO_CODES = np.empty(0, dtype=np.int64)


def numeric_variable(raw: np.ndarray) -> Variable:
    """A numeric column read with ``dropna=False``; infinities count as missing."""
    values = np.asarray(raw, dtype=np.float64)
    return Variable("numeric", _NO_CODES, np.where(finite(values), values, np.nan), 0)


def class_variable(raw: np.ndarray) -> Variable:
    """A categorical or boolean column read with ``dropna=False``."""
    codes, levels = level_codes(raw)
    return Variable("class", codes, _EMPTY, int(levels.size))


def null_mask(raw: np.ndarray) -> np.ndarray:
    """True where a column read with ``dropna=False`` holds a null."""
    kind = raw.dtype.kind
    if kind == "f":
        return np.isnan(raw)
    if kind in "Mm":
        return np.isnat(raw)
    if kind == "O":
        return np.fromiter(
            (v is None or v != v for v in raw), dtype=bool, count=raw.size
        )
    return np.zeros(raw.size, dtype=bool)


def sample_rows(
    ctx: CheckContext, max_rows: int, seed: int, *, rows_read: str = "rows"
) -> np.ndarray | None:
    """The sorted indices of a seeded sample of the rows, or None to read them all."""
    total = ctx.profile.n_rows
    if total <= max_rows:
        return None
    rows = np.sort(np.random.default_rng(seed).choice(total, max_rows, replace=False))
    ctx.note(
        "rows_sampled",
        f"Read {rows.size:,} of {total:,} {rows_read}, sampled with seed {seed}.",
    )
    return rows


def read(ctx: CheckContext, name: str, rows: np.ndarray | None) -> np.ndarray:
    """The column with its nulls kept, limited to the sampled ``rows``."""
    raw = ctx.values(name, dropna=False)
    return raw if rows is None else raw[rows]


def resolve_target(
    ctx: CheckContext, name: str | None, rows: np.ndarray | None, max_levels: int
) -> Target | CheckOutput:
    """The target as a class or numeric variable, or the output that ends the check.

    The check ends ``not_applicable`` without a target, with one that is not a
    numeric, categorical or boolean column of the data, with a class target of more
    than ``max_levels`` values, or with a target that has no spread.
    """
    if name is None:
        return _stopped(
            ctx,
            "no_target",
            "No target column is set, so there is nothing for a feature to leak "
            "or to predict.",
            "Pass target=... to dsbf.profile(), or set the `target` parameter.",
        )
    if name not in ctx.columns:
        return _stopped(
            ctx,
            "target_not_supported",
            f"The target {name!r} is not a numeric, categorical or boolean column "
            "of the data, so no feature can be measured against it.",
        )
    column = ctx.profile.column(name)
    if column.semantic_type not in TARGET_TYPES:
        return _stopped(
            ctx,
            "target_not_supported",
            f"The target {name!r} is a {column.semantic_type} column; only numeric, "
            "categorical and boolean targets are supported.",
        )
    raw = read(ctx, name, rows)
    if column.semantic_type == "numeric":
        variable = numeric_variable(raw)
        try:
            unit_column(variable.values[variable.present])
        except Undefined:
            return _stopped(
                ctx, "target_constant", f"The target {name!r} has no variation."
            )
        return Target(**variable.__dict__, name=name)
    variable = class_variable(raw)
    if variable.n_levels < 2:
        return _stopped(
            ctx,
            "target_constant",
            f"The target {name!r} has fewer than two values, so nothing can be "
            "associated with it.",
        )
    if variable.n_levels > max_levels:
        return _stopped(
            ctx,
            "target_not_supported",
            f"The target {name!r} has {variable.n_levels:,} classes, more than the "
            f"limit of {max_levels}.",
            "Raise `max_levels`, or choose a target with fewer classes.",
        )
    levels = level_codes(raw)[1]
    return Target(**variable.__dict__, name=name, levels=tuple(str(v) for v in levels))


def _stopped(
    ctx: CheckContext, code: str, message: str, hint: str | None = None
) -> CheckOutput:
    return ctx.stop(
        CheckStatus.NOT_APPLICABLE, code, message, severity=Severity.INFO, hint=hint
    )


def associate(feature: Variable, target: Variable, *, min_n: int) -> Association:
    """The association of a feature with the target, on the scale of its type pair.

    Numeric with numeric: the larger of |Pearson r| and |Spearman ρ|. Numeric with
    class, either way round: the correlation ratio η. Class with class: the
    bias-corrected Cramér's V. Raises ``Undefined`` when the pair has fewer than
    ``min_n`` complete rows or a column with no spread.
    """
    both = feature.present & target.present
    n_used = int(np.count_nonzero(both))
    if n_used < min_n:
        raise Undefined("too_few_rows")
    if feature.kind == "numeric" and target.kind == "numeric":
        x, y = feature.values[both], target.values[both]
        pearson = abs(float(np.dot(unit_column(x), unit_column(y))))
        spearman = abs(
            float(np.dot(unit_column(rankdata(x)), unit_column(rankdata(y))))
        )
        if spearman > pearson:
            return Association("spearman_r", min(spearman, 1.0), n_used)
        return Association("pearson_r", min(pearson, 1.0), n_used)
    if feature.kind == "class" and target.kind == "class":
        table, _, _ = contingency_table(feature.codes[both], target.codes[both])
        if min(table.shape) < 2:
            raise Undefined("too_few_groups")
        _, _, corrected = cramers_v(table)
        return Association("cramers_v", float(corrected), n_used)
    numeric, grouped = (
        (feature, target) if feature.kind == "numeric" else (target, feature)
    )
    value, used = correlation_ratio(
        numeric.values[both],
        grouped.codes[both],
        grouped.n_levels,
        min_n=min_n,
    )
    return Association("eta", value, used)
