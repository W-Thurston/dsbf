# dsbf/checks/fuzzy_duplicates.py
"""Rows that are one record written twice, differing by a typo, case or padding."""

import math
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl
from pydantic import Field

from dsbf.checks.strings import holds_strings
from dsbf.engine import (
    CheckContext,
    CheckOutput,
    CheckOutputModel,
    Requires,
    SemanticTypes,
    check,
)
from dsbf.model import (
    CheckParams,
    CheckStatus,
    Confidence,
    Contract,
    Cost,
    Dimension,
    Evidence,
    Severity,
    Subject,
)

# Pairs scored between two checks of the clock and of the executor's cancel flag.
_CHUNK = 20_000
_VALUE_WIDTH = 60


class FuzzyDuplicatesParams(CheckParams):
    threshold: float = Field(
        0.92,
        ge=0.5,
        le=1.0,
        description="Similarity from which two records are near-duplicates: the mean "
        "of the normalised Indel ratio of the two values, over the columns either has "
        "a value in, weighted by each column's entropy in bits, so one long shared "
        "field or a column that hardly varies cannot make unlike rows alike.",
    )
    number_cap: float = Field(
        0.7,
        ge=0,
        le=1,
        description="The most two values alike when both hold digits and the digits "
        "differ. Another number (an email's serial, a street number, a sequential "
        "code) is evidence of a different record, which a typo in letters is not; "
        "1 reads digits as letters, so a typo in a number is a typo.",
    )
    max_label_levels: int = Field(
        20,
        ge=1,
        description="A column with at most this many distinct values is a label and "
        "is left out of the row string: it says what kind of record a row is, not "
        "which one.",
    )
    max_chars: int = Field(
        200,
        ge=20,
        le=5000,
        description="Characters of a value that are compared; the rest is cut.",
    )
    max_rows: int = Field(
        50_000,
        ge=10,
        description="Most distinct records compared; more are sampled with `seed`.",
    )
    max_pairs: int = Field(
        2_000_000,
        ge=100,
        description="Most pairs scored. When every pair fits all are scored, "
        "otherwise only the pairs that share a rare token.",
    )
    max_block_size: int = Field(
        50,
        ge=2,
        le=1000,
        description="Most records that may share a token for it to block: a token "
        "shared by more says nothing about which records belong together.",
    )
    max_seconds: float = Field(
        60.0, gt=0, description="Seconds of scoring after which the check stops."
    )
    max_pairs_listed: int = Field(
        20, ge=1, le=200, description="Most similar pairs listed in the output."
    )
    warn_share: float = Field(
        0.01,
        ge=0,
        le=1,
        description="Share of rows in a flagged pair above which the finding is a "
        "warning.",
    )
    exclude_columns: tuple[str, ...] = Field(
        (),
        description="Columns left out of the comparison, such as a key that "
        "differs between copies of a record.",
    )
    seed: int = Field(42, description="Seed of the sample taken above `max_rows`.")


class PairSide(Contract):
    row: int = Field(ge=0, description="First row of this record.")
    rows: int = Field(ge=1, description="Rows holding this record.")


class NearDuplicatePair(Contract):
    first: PairSide
    second: PairSide
    similarity: float = Field(ge=0, le=1)
    differing_columns: tuple[str, ...] = Field(
        description="Columns whose values differ, case and padding included."
    )
    values: dict[str, tuple[str | None, str | None]] = Field(
        description="The differing columns' values on each side, cut short."
    )


class FuzzyDuplicatesOutput(CheckOutputModel):
    columns: tuple[str, ...] = Field(description="The columns compared.")
    column_weights: dict[str, float] = Field(
        description="Each compared column's weight in a pair's similarity: its "
        "entropy in bits."
    )
    skipped_columns: dict[str, str] = Field(
        description="Columns left out of it, with the reason."
    )
    rows: int = Field(ge=0, description="Rows with a value in at least one column.")
    records: int = Field(ge=0, description="Distinct rows of those columns.")
    records_compared: int = Field(
        ge=0, description="Records after the sample taken above `max_rows`."
    )
    exact_repeats: int = Field(
        ge=0,
        description="Rows beyond the first of a record; `duplicate_rows` owns them.",
    )
    blocking: Literal["exhaustive", "rare_tokens"]
    pairs_possible: int = Field(ge=0, description="Pairs among all the records.")
    pairs_compared: int = Field(ge=0)
    coverage: float = Field(
        ge=0, le=1, description="`pairs_compared` over `pairs_possible`."
    )
    stopped_by: Literal["rows", "pairs", "seconds"] | None = Field(
        description="The budget that left candidate pairs unscored, if any."
    )
    threshold: float
    pairs_flagged: int = Field(ge=0)
    rows_involved: int = Field(
        ge=0, description="Rows holding a record that is in a flagged pair."
    )
    top_pairs: tuple[NearDuplicatePair, ...] = Field(
        description="The most similar flagged pairs; the bulk is not stored."
    )


@check(
    id="fuzzy_duplicates",
    version="1.2.0",
    kind="analysis",
    applies_to=SemanticTypes("categorical", "text", "identifier"),
    dimensions=[Dimension.VALIDITY],
    params=FuzzyDuplicatesParams,
    output=FuzzyDuplicatesOutput,
    requires=Requires(profile=["counts"]),
    # C++ scoring of up to `max_pairs` pairs beside Python loops, so it gets its own
    # process and cannot slow the checks that share the interpreter lock.
    cost=Cost.HEAVY,
    runs_in="process",
    extras=["text"],
    packs=["core"],
    tags=["duplicates", "rows", "string", "fuzzy"],
)
def fuzzy_duplicates(ctx: CheckContext, params: FuzzyDuplicatesParams) -> CheckOutput:
    """Find pairs of rows whose text is almost, but not exactly, the same."""
    started = time.monotonic()
    weights, skipped = _surface(ctx, params)
    columns = list(weights)
    if not columns:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "no_text_columns",
            "No column has more than "
            f"{params.max_label_levels} distinct string values, so there is no text "
            "to compare rows by.",
        )
    table = _records(ctx, columns, tuple(weights.values()), params)
    rows = int(table.frame["rows"].sum()) if table.frame.height else 0
    records = table.frame.height
    if records < 2:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "too_few_records",
            "Fewer than two distinct non-empty rows are left to compare.",
        )
    stopped_by: Literal["rows", "pairs", "seconds"] | None = None
    if records > params.max_rows:
        table = table.take(
            np.sort(
                np.random.default_rng(params.seed).choice(
                    records, size=params.max_rows, replace=False
                )
            )
        )
        stopped_by = "rows"
    compared = table.frame.height
    possible = records * (records - 1) // 2
    ctx.checkpoint()

    if compared * (compared - 1) // 2 <= params.max_pairs:
        blocking = "exhaustive"
        first, second = np.triu_indices(compared, k=1)
    else:
        blocking = "rare_tokens"
        first, second, truncated = _blocked_pairs(ctx, table.tokens, params)
        if truncated and stopped_by is None:
            stopped_by = "pairs"

    flagged_first, flagged_second, similarity, scored, timed_out = _score(
        ctx, table, first, second, params, started + params.max_seconds, possible
    )
    if timed_out:
        stopped_by = "seconds"

    order = np.lexsort((flagged_second, flagged_first, -similarity))
    flagged_first, flagged_second = flagged_first[order], flagged_second[order]
    similarity = similarity[order]
    involved = np.unique(np.concatenate([flagged_first, flagged_second]))
    counts = table.frame["rows"].to_numpy()
    rows_involved = int(counts[involved].sum()) if involved.size else 0
    top = _top_pairs(
        table.frame, columns, flagged_first, flagged_second, similarity, params
    )

    output = FuzzyDuplicatesOutput(
        columns=tuple(columns),
        column_weights=weights,
        skipped_columns=skipped,
        rows=rows,
        records=records,
        records_compared=compared,
        exact_repeats=rows - records,
        blocking=blocking,
        pairs_possible=possible,
        pairs_compared=scored,
        coverage=scored / possible if possible else 1.0,
        stopped_by=stopped_by,
        threshold=params.threshold,
        pairs_flagged=int(flagged_first.size),
        rows_involved=rows_involved,
        top_pairs=top,
    )
    _report(ctx, output, params)
    return ctx.output(output)


def _surface(
    ctx: CheckContext, params: FuzzyDuplicatesParams
) -> tuple[dict[str, float], dict[str, str]]:
    """The columns compared with their weights, and why the others are left out."""
    weights: dict[str, float] = {}
    skipped: dict[str, str] = {}
    for name in ctx.columns:
        profile = ctx.profile.column(name)
        if name in params.exclude_columns:
            skipped[name] = "left out by exclude_columns"
        elif not holds_strings(profile):
            skipped[name] = f"stored as {profile.physical_dtype}, not as strings"
        elif not profile.counts.non_null:
            skipped[name] = "no values"
        elif profile.counts.distinct <= params.max_label_levels:
            skipped[name] = (
                f"{profile.counts.distinct} distinct values, a label and not a name"
            )
        else:
            bits = getattr(profile, "entropy_bits", None)
            weights[name] = float(bits) if bits else math.log2(profile.counts.distinct)
    return weights, skipped


@dataclass(frozen=True)
class _Records:
    """The distinct rows of the compared columns, ready to score."""

    # The raw values as c0, c1, ..., `first` (first row of the record) and `rows`.
    frame: pl.DataFrame
    values: list[list[str]]  # per column, normalised; "" where there is no value
    digits: list[list[str]]  # per column, only the digits of the value
    present: list[np.ndarray]  # per column, whether the record has a value
    weights: tuple[float, ...]  # per column
    bits: tuple[float, ...]  # per column, what a match is worth: -log2 of the chance
    tokens: list[str]  # per record, every column's value joined by a space

    def take(self, index: np.ndarray) -> "_Records":
        picked = index.tolist()
        return _Records(
            frame=self.frame[picked],
            values=[[column[i] for i in picked] for column in self.values],
            digits=[[column[i] for i in picked] for column in self.digits],
            present=[column[index] for column in self.present],
            weights=self.weights,
            bits=self.bits,
            tokens=[self.tokens[i] for i in picked],
        )


def _records(
    ctx: CheckContext,
    columns: list[str],
    weights: tuple[float, ...],
    params: FuzzyDuplicatesParams,
) -> _Records:
    """The distinct raw rows of ``columns``, in order of first appearance.

    Values are lower-cased, stripped, collapsed to single spaces and cut to
    ``max_chars``; a blank value counts as missing. A record with no value in any
    column is dropped.
    """
    # The working frame names its columns c0, c1, ...: a column of the data may be
    # called "first", "rows" or nothing at all.
    inner = [f"c{i}" for i in range(len(columns))]
    frame = pl.DataFrame(
        [
            pl.Series(key, ctx.values(name, dropna=False).tolist(), dtype=pl.String)
            for key, name in zip(inner, columns, strict=True)
        ]
    )
    records = (
        frame.with_row_index("first")
        .group_by(inner, maintain_order=True)
        .agg(pl.col("first").first(), rows=pl.len())
    )
    ctx.checkpoint()
    normal = records.select(
        pl.col(key)
        .str.to_lowercase()
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .str.slice(0, params.max_chars)
        for key in inner
    )
    normal = normal.with_columns(
        pl.when(pl.col(c) == "").then(None).otherwise(pl.col(c)).alias(c)
        for c in normal.columns
    )
    keep = pl.any_horizontal(pl.all().is_not_null())
    mask = normal.select(keep.alias("keep"))["keep"]
    records, normal = records.filter(mask), normal.filter(mask)
    return _Records(
        frame=records,
        values=[normal[c].fill_null("").to_list() for c in normal.columns],
        digits=[
            normal[c].fill_null("").str.replace_all(r"[^0-9]", "").to_list()
            for c in normal.columns
        ],
        present=[normal[c].is_not_null().to_numpy() for c in normal.columns],
        weights=weights,
        bits=tuple(_match_bits(normal[c]) for c in normal.columns),
        tokens=normal.select(
            pl.concat_str(pl.all().fill_null(""), separator=" ")
            .str.replace_all(r"\s+", " ")
            .str.strip_chars()
        )
        .to_series()
        .to_list(),
    )


def _match_bits(column: pl.Series) -> float:
    """Bits of evidence in two records holding the same value of ``column``.

    ``-log2`` of the chance that two different records picked at random agree on it,
    counted over pairs of records, with one coincidence added so that a column of
    values all different is worth the bits of its pairs and no more: a thousand
    names are worth 20 bits, a column of two cities about one.
    """
    counts = column.drop_nulls().value_counts()["count"].to_numpy().astype(float)
    n = counts.sum()
    if n < 2:
        return 0.0
    agreeing = float(np.sum(counts * (counts - 1))) + 1
    return float(-np.log2(agreeing / (n * (n - 1))))


def _with_deletions(held: pl.DataFrame, threshold: float) -> pl.DataFrame:
    """``held`` (record, token) with a row for each token and for each of its
    one-character deletions, so that tokens one character apart share a key.

    Two tokens one character apart (one dropped, added, changed or swapped with its
    neighbour) both reduce to a common deletion, so records whose every token is
    mistyped still meet. Only tokens long enough to lose a character and stay
    above the threshold are expanded, since a shorter one cannot be a typo of
    another that scores as a duplicate.
    """
    if threshold >= 1:
        return held
    shortest = max(3, math.floor(1 / (2 * (1 - threshold))))
    distinct = held["token"].unique().to_list()
    pairs = [
        (token, token[:i] + token[i + 1 :])
        for token in distinct
        if len(token) >= shortest
        for i in range(len(token))
    ]
    if not pairs:
        return held
    expanded = held.join(
        pl.DataFrame(
            {"token": [p[0] for p in pairs], "key": [p[1] for p in pairs]},
            schema={"token": pl.String, "key": pl.String},
        ),
        on="token",
    ).select("record", pl.col("key").alias("token"))
    return pl.concat([held, expanded]).unique()


def _blocked_pairs(
    ctx: CheckContext, texts: list[str], params: FuzzyDuplicatesParams
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Pairs of records that share a rare token, rarest tokens first.

    A token is a block when 2 to ``max_block_size`` records hold it, or hold a token
    that is one character away (see ``_with_deletions``). Blocks are taken
    in order of size, then of the token, until the pairs would pass ``max_pairs``, so
    neither the hash order of the tokens nor the number of workers changes the pairs.
    Returns the first and second record of each pair, sorted, and whether the budget
    cut the last blocks.
    """
    held = (
        pl.DataFrame({"record": np.arange(len(texts)), "token": texts})
        .with_columns(pl.col("token").str.split(" "))
        .explode("token")
        .unique()
    )
    tokens = (
        _with_deletions(held, params.threshold)
        .group_by("token")
        .agg(pl.col("record").sort(), size=pl.len())
        .filter(pl.col("size").is_between(2, params.max_block_size))
        .sort(["size", "token"])
        .with_columns(
            pairs=pl.col("size") * (pl.col("size") - 1) // 2,
        )
        .with_columns(spent=pl.col("pairs").cum_sum())
    )
    ctx.checkpoint()
    within = tokens.filter(pl.col("spent") <= params.max_pairs)
    truncated = within.height < tokens.height
    parts: list[np.ndarray] = []
    n = len(texts)
    for size in within["size"].unique().sort().to_list():
        members = np.array(
            within.filter(pl.col("size") == size)["record"].to_list(), dtype=np.int64
        )
        low, high = np.triu_indices(size, k=1)
        parts.append((members[:, low] * n + members[:, high]).ravel())
        ctx.checkpoint()
    if not parts:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, truncated
    keys = np.unique(np.concatenate(parts))
    return keys // n, keys % n, truncated


def _score(
    ctx: CheckContext,
    table: _Records,
    first: np.ndarray,
    second: np.ndarray,
    params: FuzzyDuplicatesParams,
    deadline: float,
    pairs_possible: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, bool]:
    """Score the pairs and keep those at or above the threshold.

    A pair scores the weighted mean, over the columns either record has a value in,
    of the Indel ratio of the two values (0 where only one has a value). A pair also
    scores with any one column left out, so a key that differs between two systems
    does not hide the same person, but only when the columns that match carry more
    evidence than chance would give some pair among all those possible. Returns the
    flagged pairs (first, second, similarity in 0 to 1), the number of pairs scored
    and whether the clock ran out before all were.
    """
    from rapidfuzz import fuzz, process  # the `text` extra

    cutoff = params.threshold - 1e-9
    kept_first: list[np.ndarray] = []
    kept_second: list[np.ndarray] = []
    kept_score: list[np.ndarray] = []
    scored = 0
    timed_out = False
    for start in range(0, first.size, _CHUNK):
        if time.monotonic() > deadline:
            timed_out = True
            break
        ctx.checkpoint()
        low = first[start : start + _CHUNK]
        high = second[start : start + _CHUNK]
        total = np.zeros(low.size)
        used = np.zeros(low.size)
        # Per column: its weighted ratio, its weight where it counts, and the bits of
        # evidence it gives where the two values agree.
        shares: list[np.ndarray] = []
        weighed: list[np.ndarray] = []
        evidence: list[np.ndarray] = []
        for values, digits, present, weight, bits in zip(
            table.values,
            table.digits,
            table.present,
            table.weights,
            table.bits,
            strict=True,
        ):
            either = present[low] | present[high]
            if not either.any():
                shares.append(np.zeros(low.size))
                weighed.append(np.zeros(low.size))
                evidence.append(np.zeros(low.size))
                continue
            ratio = process.cpdist(  # type: ignore[no-matching-overload]
                [values[i] for i in low],
                [values[j] for j in high],
                scorer=fuzz.ratio,
                dtype=np.float64,
                workers=1,
            )
            if params.number_cap < 1:
                # Another number is another record, whatever else differs: a letter and
                # a digit changed at once is two edits, not a typo. A digit typed as
                # a letter, against none, is a typo.
                differ = np.array(
                    [
                        bool(digits[i]) and bool(digits[j]) and digits[i] != digits[j]
                        for i, j in zip(low, high, strict=True)
                    ]
                )
                ratio = np.where(
                    differ, np.minimum(ratio, 100 * params.number_cap), ratio
                )
            share = np.where(either, ratio, 0.0) * weight
            total += share
            used += either * weight
            shares.append(share)
            weighed.append(either * weight)
            evidence.append(np.where(either & (ratio >= 100 * cutoff), bits, 0.0))
        similarity = total / np.maximum(used, 1e-12) / 100
        if len(shares) > 1:
            similarity = np.maximum(
                similarity,
                _best_without_one(
                    total, used, shares, weighed, evidence, pairs_possible
                ),
            )
        scored += low.size
        hit = similarity >= cutoff
        kept_first.append(low[hit])
        kept_second.append(high[hit])
        kept_score.append(similarity[hit])
    empty = np.empty(0, dtype=np.int64)
    return (
        np.concatenate(kept_first) if kept_first else empty,
        np.concatenate(kept_second) if kept_second else empty,
        np.concatenate(kept_score) if kept_score else np.empty(0),
        scored,
        timed_out,
    )


def _best_without_one(
    total: np.ndarray,
    used: np.ndarray,
    shares: list[np.ndarray],
    weighed: list[np.ndarray],
    evidence: list[np.ndarray],
    pairs_possible: int,
) -> np.ndarray:
    """The best similarity of each pair with one column left out, 0 where not enough
    evidence is left: a pair must match on more bits than ``log2(pairs possible)``,
    or some pair would match by chance."""
    needed = math.log2(max(pairs_possible, 2)) + 1
    matched = np.sum(evidence, axis=0)
    best = np.zeros(total.size)
    for share, weight, bits in zip(shares, weighed, evidence, strict=True):
        left = used - weight
        similarity = (total - share) / np.maximum(left, 1e-12) / 100
        enough = (left > 0) & (matched - bits >= needed)
        best = np.maximum(best, np.where(enough, similarity, 0.0))
    return best


def _top_pairs(
    frame: pl.DataFrame,
    columns: list[str],
    first: np.ndarray,
    second: np.ndarray,
    similarity: np.ndarray,
    params: FuzzyDuplicatesParams,
) -> tuple[NearDuplicatePair, ...]:
    pairs: list[NearDuplicatePair] = []
    for a, b, score in zip(
        first[: params.max_pairs_listed].tolist(),
        second[: params.max_pairs_listed].tolist(),
        similarity[: params.max_pairs_listed].tolist(),
        strict=True,
    ):
        left = frame.row(a, named=True)
        right = frame.row(b, named=True)
        differing = [i for i in range(len(columns)) if left[f"c{i}"] != right[f"c{i}"]]
        pairs.append(
            NearDuplicatePair(
                first=PairSide(row=left["first"], rows=left["rows"]),
                second=PairSide(row=right["first"], rows=right["rows"]),
                similarity=min(score, 1.0),
                differing_columns=tuple(columns[i] for i in differing),
                values={
                    columns[i]: (_cut(left[f"c{i}"]), _cut(right[f"c{i}"]))
                    for i in differing
                },
            )
        )
    return tuple(pairs)


def _cut(value: str | None) -> str | None:
    if value is None or len(value) <= _VALUE_WIDTH:
        return value
    return value[: _VALUE_WIDTH - 3] + "..."


def _report(
    ctx: CheckContext, output: FuzzyDuplicatesOutput, params: FuzzyDuplicatesParams
) -> None:
    if output.stopped_by:
        reasons = {
            "rows": f"the {output.records:,} records were sampled to "
            f"{output.records_compared:,} (max_rows)",
            "pairs": "the candidate pairs passed max_pairs, so the pairs of the "
            "commonest shared tokens were not scored",
            "seconds": "max_seconds ran out before every candidate pair was scored",
        }
        ctx.note(
            "partial_comparison",
            f"{output.pairs_compared:,} of {output.pairs_possible:,} possible pairs "
            f"were scored ({output.coverage:.2%}): {reasons[output.stopped_by]}.",
            severity=Severity.NOTICE,
            hint="Raise the budget (max_rows, max_pairs, max_seconds) for a fuller "
            "comparison, or filter to the rows in question.",
        )
    if not output.pairs_flagged:
        return
    share = output.rows_involved / output.rows
    top = output.top_pairs[0]
    columns = ", ".join(repr(c) for c in output.columns[:4])
    where = ", ".join(repr(c) for c in top.differing_columns[:3])
    differs = (
        f"differ only in case or spacing in {where}"
        if top.similarity >= 1.0 and where
        else f"differ in {where}"
        if where
        else "differ in nothing the comparison reads"
    )
    scope = (
        "every pair"
        if output.blocking == "exhaustive"
        else "the pairs that share a rare token"
    )
    ctx.finding(
        code="near_duplicate_rows",
        subject=Subject.row_set(*output.columns),
        severity=Severity.WARN if share > params.warn_share else Severity.INFO,
        confidence=Confidence.MODERATE,
        title=f"{output.pairs_flagged:,} pairs of near-duplicate rows",
        message=(
            f"{output.pairs_flagged:,} pairs of rows are at least "
            f"{output.threshold:.0%} alike across {columns}"
            f"{'...' if len(output.columns) > 4 else ''}, involving "
            f"{output.rows_involved:,} of {output.rows:,} rows ({share:.1%}). The "
            f"closest, rows {top.first.row} and {top.second.row} "
            f"({top.similarity:.1%}), {differs}. Scored: {scope} "
            f"({output.coverage:.2%} of all pairs). Such rows survive an exact "
            "duplicate check but count one record twice."
        ),
        evidence=Evidence(
            metrics={
                "pairs_flagged": output.pairs_flagged,
                "rows_involved": output.rows_involved,
                "rows_involved_share": share,
                "threshold": output.threshold,
                "pairs_compared": output.pairs_compared,
                "pairs_possible": output.pairs_possible,
                "coverage": output.coverage,
                "columns_compared": len(output.columns),
            },
            n_used=output.rows,
            n_total=ctx.profile.n_rows,
            method="indel_ratio_" + output.blocking,
        ),
    )
