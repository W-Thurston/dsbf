# dsbf/data/sampling.py
"""Row sampling that records how the rows were chosen."""

from collections.abc import Sequence
from typing import Literal

import numpy as np
import polars as pl

from dsbf.data.errors import SamplingError
from dsbf.model.evidence import SamplingInfo

Strategy = Literal["head", "random", "stratified"]
_STRATEGIES = ("head", "random", "stratified")
_RANK = "dsbf rank"
_QUOTA = "dsbf quota"


def check_request(
    frame: pl.DataFrame,
    rows: int,
    strategy: str,
    seed: int | None,
    stratify_by: Sequence[str],
) -> None:
    """Raise :class:`SamplingError` for a request that cannot be drawn as asked."""
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise SamplingError(f"rows must be a positive integer, not {rows!r}")
    if strategy not in _STRATEGIES:
        raise SamplingError(
            f"Unknown sampling strategy {strategy!r}; "
            f"use one of {', '.join(_STRATEGIES)}"
        )
    if strategy != "stratified" and stratify_by:
        raise SamplingError(
            f"stratify_by applies to the stratified strategy, not {strategy!r}"
        )
    if strategy == "head" and seed is not None:
        raise SamplingError("A head sample is not random, so it has no seed")
    if strategy == "stratified":
        if not stratify_by:
            raise SamplingError("A stratified sample needs stratify_by")
        missing = [name for name in stratify_by if name not in frame.columns]
        if missing:
            raise SamplingError(
                f"Cannot stratify by missing column(s): {', '.join(missing)}"
            )
        if _RANK in stratify_by or _QUOTA in stratify_by:
            raise SamplingError(f"Cannot stratify by a column named {_RANK!r}")


def sample_frame(
    frame: pl.DataFrame,
    rows: int,
    *,
    strategy: Strategy,
    seed: int | None,
    stratify_by: Sequence[str],
) -> tuple[pl.DataFrame, SamplingInfo]:
    """Draw ``rows`` rows from ``frame``, keeping their original order.

    The request has passed :func:`check_request` and ``rows < frame.height``.
    """
    if strategy == "head":
        sample = frame.head(rows)
        return sample, _info(strategy, frame, sample, None, ())
    seed = 0 if seed is None else seed
    if strategy == "random":
        generator = np.random.default_rng(seed)
        picked = np.sort(generator.choice(frame.height, size=rows, replace=False))
        sample = frame[picked]
        return sample, _info(strategy, frame, sample, seed, ())
    strata = tuple(dict.fromkeys(stratify_by))
    sample = _stratified(frame, rows, seed, strata)
    return sample, _info(strategy, frame, sample, seed, strata)


def _info(
    strategy: Strategy,
    frame: pl.DataFrame,
    sample: pl.DataFrame,
    seed: int | None,
    strata: tuple[str, ...],
) -> SamplingInfo:
    return SamplingInfo(
        strategy=strategy,
        original_rows=frame.height,
        sampled_rows=sample.height,
        seed=seed,
        stratify_by=strata,
    )


def _stratified(
    frame: pl.DataFrame, rows: int, seed: int, strata: tuple[str, ...]
) -> pl.DataFrame:
    """Proportional allocation by largest remainder, then a seeded draw per group.

    The sample has exactly ``rows`` rows. A null is a group of its own. A group
    whose share of ``rows`` rounds to nothing gets no rows, so with fewer rows
    than groups some groups are absent.
    """
    keys = list(strata)
    quotas = _group_quotas(frame.select(keys), rows)
    generator = np.random.default_rng(seed)
    work = frame.select(keys).with_columns(
        pl.Series(_RANK, generator.random(frame.height))
    )
    # Within each group, the `quota` rows with the smallest random rank are kept.
    ranked = work.select(
        pl.col(_RANK).rank("ordinal").over(keys).alias(_RANK),
        *keys,
    ).join(quotas, on=keys, how="left", nulls_equal=True, maintain_order="left")
    return frame.filter(ranked[_RANK] <= ranked[_QUOTA])


def _group_quotas(keys: pl.DataFrame, rows: int) -> pl.DataFrame:
    """How many rows each group contributes, summing to exactly ``rows``."""
    columns = keys.columns
    groups = keys.group_by(columns).agg(pl.len().alias("size")).sort(columns)
    total = keys.height
    # Integer arithmetic, so there is no float rounding to disagree about.
    share = groups["size"].cast(pl.Int64) * rows
    whole = share // total
    remainder = share % total
    leftover = rows - int(whole.sum())
    # The largest remainders each get one more row; ties go to the sorted-first group.
    order = np.argsort(-remainder.to_numpy(), kind="stable")
    bonus = np.zeros(groups.height, dtype=np.int64)
    bonus[order[:leftover]] = 1
    return groups.select(*columns, (whole + pl.Series(bonus)).alias(_QUOTA))
