# dsbf/service/relationships.py
"""The association matrix and the pair plot, from stored results and the snapshot."""

from typing import Any, Literal

import pyarrow as pa
from pydantic import Field

from dsbf.model import CheckResult, CheckStatus, Interval
from dsbf.model.base import Contract
from dsbf.model.evidence import SamplingInfo
from dsbf.service.views import json_safe

Statistic = Literal[
    "pearson_r",
    "spearman_r",
    "kendalls_tau",
    "eta",
    "cramers_v_corrected",
    "omega_squared",
]

# Statistics are not on one scale, so the view groups by this order before it sorts
# by value: correlations, then η, then Cramér's V, then ω².
STATISTIC_ORDER: tuple[Statistic, ...] = (
    "pearson_r",
    "spearman_r",
    "kendalls_tau",
    "eta",
    "cramers_v_corrected",
    "omega_squared",
)
PAIR_CHECKS = ("associations", "contingency", "group_differences")
MAX_POINTS = 2_000


class RelationshipCell(Contract):
    """One tested pair and the statistic that measured it."""

    first: str
    second: str
    check_id: str
    statistic: Statistic
    value: float = Field(description="Signed for r, ρ and τ; otherwise 0 to 1.")
    strength: str | None = Field(
        description="The check's band for the value; null when it names none."
    )
    n_used: int = Field(ge=0)
    interval: Interval | None
    flagged: bool | None = Field(
        description="Whether the check flagged the pair, when it says."
    )


class PairCoverage(Contract):
    """How far one pairwise check got."""

    check_id: str
    status: CheckStatus
    pairs_tested: int = Field(ge=0)
    untested: dict[str, int] = Field(description="Pairs not tested, by reason.")


class RelationshipMatrix(Contract):
    """The tested pairs of a run; a pair absent from `cells` was not tested."""

    columns: tuple[str, ...] = Field(
        description="Columns that appear in a cell, in the dataset's order."
    )
    statistics: tuple[Statistic, ...] = Field(
        description="Statistics present, in the order the view groups them."
    )
    cells: tuple[RelationshipCell, ...]
    coverage: tuple[PairCoverage, ...]


class PairPoints(Contract):
    """Two columns of the snapshot, for the rows where both have a value."""

    first: str
    second: str
    values: tuple[tuple[Any, Any], ...] = Field(
        description="(first, second) per row, thinned evenly to at most `limit`."
    )
    rows_with_both: int = Field(
        ge=0, description="Snapshot rows where both columns have a value."
    )
    snapshot_rows: int = Field(ge=0)
    sampling: SamplingInfo | None = Field(
        description="How the snapshot's rows were drawn from the data."
    )


def build_matrix(
    results: dict[str, CheckResult | None], order: tuple[str, ...]
) -> RelationshipMatrix:
    cells: list[RelationshipCell] = []
    coverage: list[PairCoverage] = []
    for check_id in PAIR_CHECKS:
        result = results.get(check_id)
        if result is None:
            continue
        output = result.output or {}
        pairs = output.get("pairs") or []
        cells.extend(_cell(check_id, pair) for pair in pairs)
        coverage.append(
            PairCoverage(
                check_id=check_id,
                status=result.status,
                pairs_tested=len(pairs),
                untested=output.get("untested") or {},
            )
        )
    present = {name for cell in cells for name in (cell.first, cell.second)}
    position = {name: index for index, name in enumerate(order)}
    columns = tuple(sorted(present, key=lambda name: position.get(name, len(order))))
    used = {cell.statistic for cell in cells}
    return RelationshipMatrix(
        columns=columns,
        statistics=tuple(s for s in STATISTIC_ORDER if s in used),
        cells=tuple(cells),
        coverage=tuple(coverage),
    )


def _cell(check_id: str, pair: dict[str, Any]) -> RelationshipCell:
    interval = pair.get("interval")
    if check_id == "associations":
        statistic, value = pair["metric"], pair["value"]
        first, second = pair["first"], pair["second"]
        strength, flagged = pair.get("strength"), None
        interval = pair.get("interval")
    elif check_id == "contingency":
        statistic, value = "cramers_v_corrected", pair["cramers_v_corrected"]
        first, second = pair["first"], pair["second"]
        strength, flagged = pair.get("strength"), None
    else:
        statistic, value = "omega_squared", pair["omega_squared"]
        first, second = pair["numeric"], pair["categorical"]
        strength, flagged = None, pair.get("flagged")
        interval = pair.get("omega_interval")
    return RelationshipCell(
        first=first,
        second=second,
        check_id=check_id,
        statistic=statistic,
        value=value,
        strength=strength,
        n_used=pair["n_used"],
        interval=Interval.model_validate(interval) if interval else None,
        flagged=flagged,
    )


def pair_points(
    table: pa.Table,
    sampling: SamplingInfo | None,
    first: str,
    second: str,
    limit: int,
) -> PairPoints:
    both = table.select([first, second]).drop_null()
    kept = both.num_rows
    if kept > limit:
        step = kept / limit
        both = both.take([int(i * step) for i in range(limit)])
    columns = [[json_safe(v) for v in column.to_pylist()] for column in both.columns]
    return PairPoints(
        first=first,
        second=second,
        values=tuple(zip(*columns, strict=True)),
        rows_with_both=kept,
        snapshot_rows=table.num_rows,
        sampling=sampling,
    )
