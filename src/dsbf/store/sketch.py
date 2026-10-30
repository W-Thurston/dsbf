# dsbf/store/sketch.py
"""A column's sketch: the few figures and the small distribution a table row and its
inline chart need, cut from the stored profile."""

from dsbf.model.profile import (
    BooleanProfile,
    CategoricalProfile,
    ColumnProfile,
    NumericProfile,
    TextProfile,
    ValueCount,
)
from dsbf.store.repository import ColumnSketch

MAX_BINS = 12
MAX_TOP_VALUES = 5


def sketch_of(column: ColumnProfile) -> ColumnSketch:
    if isinstance(column, NumericProfile):
        return _numeric(column)
    if isinstance(column, CategoricalProfile):
        return _listed(column, column.top_values, column.mode_count)
    if isinstance(column, TextProfile):
        return _listed(column, column.top_values, None)
    if isinstance(column, BooleanProfile):
        return _boolean(column)
    return ColumnSketch()


def _numeric(column: NumericProfile) -> ColumnSketch:
    stats = column.stats
    if stats is None:
        return ColumnSketch()
    bins = ()
    if column.histogram is not None and stats.min is not None:
        bins = _recut(column.histogram.counts)
    return ColumnSketch(
        skewness=stats.skewness, min=stats.min, max=stats.max, bins=bins
    )


def _recut(counts: tuple[int, ...]) -> tuple[int, ...]:
    """At most ``MAX_BINS`` equal-width bins over the same range, by interpolating the
    cumulative count at the new edges; the total is kept."""
    if len(counts) <= MAX_BINS:
        return counts
    cumulative = [0]
    for count in counts:
        cumulative.append(cumulative[-1] + count)
    recut = [0]
    for i in range(1, MAX_BINS + 1):
        position = i * len(counts) / MAX_BINS
        whole = min(int(position), len(counts) - 1)
        fraction = position - whole
        value = cumulative[whole] + fraction * counts[whole]
        recut.append(round(value) if i < MAX_BINS else cumulative[-1])
    return tuple(b - a for a, b in zip(recut, recut[1:], strict=False))


def _listed(
    column: CategoricalProfile | TextProfile,
    top: tuple[ValueCount, ...],
    mode_count: int | None,
) -> ColumnSketch:
    non_null = column.counts.non_null
    count = mode_count if mode_count is not None else (top[0].count if top else None)
    return ColumnSketch(
        mode=str(top[0].value)
        if top and (mode_count in (None, top[0].count))
        else None,
        mode_share=None if count is None or not non_null else count / non_null,
        top_values=top[:MAX_TOP_VALUES],
    )


def _boolean(column: BooleanProfile) -> ColumnSketch:
    non_null = column.counts.non_null
    pairs = sorted(
        (
            ValueCount(value=value, count=n)
            for value, n in (
                (True, column.true_count),
                (False, column.false_count),
            )
            if n
        ),
        key=lambda pair: -pair.count,
    )
    if not pairs:
        return ColumnSketch()
    return ColumnSketch(
        mode=str(pairs[0].value).lower(),
        mode_share=pairs[0].count / non_null,
        top_values=tuple(pairs),
    )


__all__ = ["MAX_BINS", "MAX_TOP_VALUES", "sketch_of"]
