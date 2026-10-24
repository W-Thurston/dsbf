# dsbf/checks/encoded_columns.py
"""Columns of identifiers, hashes and other tokens that carry no readable meaning."""

import math
from collections import Counter
from typing import Literal

import numpy as np
import polars as pl
from pydantic import Field

from dsbf.checks.strings import (
    STRING_TYPES,
    distinct_values,
    examples,
    holds_strings,
)
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

Kind = Literal["uuid", "hex", "base64", "opaque_token"]

_UUID = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
# A digest has a fixed length, and a hexadecimal number of that length with no letter
# is a number.
_HEX = r"^[0-9a-fA-F]+$"
_HEX_LETTER = r"[a-fA-F]"
_HEX_DIGIT = r"[0-9]"
_HEX_LENGTHS = (32, 40, 56, 64, 96, 128)
# Base64 needs enough characters to mean something and a length that is a multiple of
# four. Its text must either decode to printable ASCII, or look random: both cases and
# a character that a plain word does not hold.
_BASE64 = r"^[A-Za-z0-9+/]+={0,2}$"
_PRINTABLE_HEX = r"^(?:[2-6][0-9a-f]|7[0-9a-e]){4,}$"
_BASE64_MARK = r"[0-9+/=]"
_BASE64_LOWER, _BASE64_UPPER = r"[a-z]", r"[A-Z]"
_WHITESPACE = r"\s"
# Values whose entropy is measured: at most this many distinct ones, drawn with a seed.
_TOKEN_SAMPLE = 2_000
_SEED = 0


class EncodedColumnsParams(CheckParams):
    min_values: int = Field(
        10, ge=1, description="Non-null values a column needs to be read."
    )
    min_share: float = Field(
        0.9,
        gt=0,
        le=1,
        description="Share of a column's values that must look encoded.",
    )
    detect_uuid: bool = Field(True, description="Look for UUIDs.")
    detect_hex: bool = Field(
        True,
        description="Look for hexadecimal digests of 32 to 128 digits, and shorter "
        "hexadecimal keys (see `min_hex_length`).",
    )
    min_hex_length: int = Field(
        8,
        ge=4,
        description="Digits a hexadecimal key of another length than a digest has at "
        "least, when it holds both a digit and a letter: a request id or a short "
        "commit hash. Random hexadecimal has no digit or no letter in about one "
        "value in fifty at 8 digits and almost never at more.",
    )
    detect_base64: bool = Field(True, description="Look for base64 strings.")
    detect_tokens: bool = Field(
        True,
        description="Look for opaque tokens: long strings of one length with a high "
        "entropy per character.",
    )
    min_base64_length: int = Field(
        12,
        ge=4,
        description="Characters a base64 string has at least; shorter words that "
        "mix cases and digits are too often plain text.",
    )
    min_token_length: int = Field(
        16, ge=4, description="Characters an opaque token has at least."
    )
    min_entropy: float = Field(
        5.0,
        gt=0,
        description="Shannon entropy, in bits per character, of the characters of "
        "all the candidate tokens together from which they look random. Random "
        "letters and digits have 5.2 to 6, email addresses and other text 4.1 to "
        "4.7.",
    )
    length_std_threshold: float = Field(
        2.0,
        ge=0,
        description="Largest standard deviation of the length of the opaque tokens: "
        "hashes and keys are of one length.",
    )
    max_examples: int = Field(3, ge=1, le=10, description="Examples listed per column.")


class EncodedColumn(Contract):
    kind: Kind = Field(description="The kind most of the encoded values are.")
    share: float = Field(
        ge=0, le=1, description="Share of the rows whose value looks encoded."
    )
    shares: dict[str, float] = Field(description="That share, by kind.")
    n_used: int = Field(ge=1, description="Non-null values read.")
    distinct: int = Field(ge=1)
    mean_length: float = Field(ge=0)
    length_std: float = Field(ge=0)
    examples: tuple[str, ...]


class EncodedColumnsOutput(CheckOutputModel):
    columns: dict[str, EncodedColumn] = Field(
        description="Each column whose values look encoded, by name."
    )
    columns_read: int = Field(ge=0)
    skipped_columns: dict[str, str] = Field(
        description="Columns left out, with the reason."
    )


@check(
    id="encoded_columns",
    version="1.1.0",
    kind="analysis",
    applies_to=SemanticTypes(*STRING_TYPES),
    dimensions=[Dimension.USABILITY],
    params=EncodedColumnsParams,
    output=EncodedColumnsOutput,
    requires=Requires(profile=["counts"]),
    # Regular expressions over each column's distinct values; entropy for a sample.
    cost=Cost.CHEAP,
    packs=["core"],
    tags=["encoding", "hash", "uuid", "base64"],
)
def encoded_columns(ctx: CheckContext, params: EncodedColumnsParams) -> CheckOutput:
    """Report columns whose values are UUIDs, digests, base64 or random tokens."""
    columns: dict[str, EncodedColumn] = {}
    skipped: dict[str, str] = {}
    read = 0
    for name in ctx.columns:
        ctx.checkpoint()
        profile = ctx.profile.column(name)
        if not holds_strings(profile):
            skipped[name] = f"stored as {profile.physical_dtype}, not as strings"
            continue
        if profile.counts.non_null < params.min_values:
            skipped[name] = f"fewer than {params.min_values} values"
            continue
        read += 1
        found = _read(distinct_values(ctx.values(name)), params)
        if found is not None and found.share >= params.min_share:
            columns[name] = found
            _report(ctx, name, found)
    if not read:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "no_strings",
            f"No column in scope holds at least {params.min_values} non-null strings.",
        )
    return ctx.output(
        EncodedColumnsOutput(
            columns=columns, columns_read=read, skipped_columns=skipped
        )
    )


def _read(distinct: pl.DataFrame, params: EncodedColumnsParams) -> EncodedColumn | None:
    """How much of the column looks encoded, and as what; None when none of it does."""
    value = pl.col("value")
    length = value.str.len_chars()
    kinds = pl.when(pl.lit(False)).then(pl.lit(None, dtype=pl.String))
    if params.detect_uuid:
        kinds = kinds.when(value.str.contains(_UUID)).then(pl.lit("uuid"))
    if params.detect_hex:
        kinds = kinds.when(
            value.str.contains(_HEX)
            & value.str.contains(_HEX_LETTER)
            & (
                length.is_in(_HEX_LENGTHS)
                | (value.str.contains(_HEX_DIGIT) & (length >= params.min_hex_length))
            )
        ).then(pl.lit("hex"))
    if params.detect_base64:
        printable = (
            value.str.decode("base64", strict=False)
            .bin.encode("hex")
            .str.contains(_PRINTABLE_HEX)
            .fill_null(False)
        )
        random_like = (
            value.str.contains(_BASE64_MARK)
            & value.str.contains(_BASE64_LOWER)
            & value.str.contains(_BASE64_UPPER)
        )
        kinds = kinds.when(
            value.str.contains(_BASE64)
            & (length % 4 == 0)
            & (length >= params.min_base64_length)
            & (printable | random_like)
        ).then(pl.lit("base64"))
    classified = distinct.with_columns(kind=kinds.otherwise(None))
    total = int(distinct["count"].sum())
    counts: dict[str, int] = {
        row["kind"]: row["rows"]
        for row in classified.filter(pl.col("kind").is_not_null())
        .group_by("kind")
        .agg(rows=pl.col("count").sum())
        .iter_rows(named=True)
    }
    if params.detect_tokens:
        tokens = _tokens(classified.filter(pl.col("kind").is_null()), params)
        if tokens:
            counts["opaque_token"] = tokens
    if not counts:
        return None
    matched = sum(counts.values())
    kind: Kind = max(counts, key=lambda k: (counts[k], k))  # type: ignore[assignment]
    lengths = distinct.with_columns(length=length)
    weights = lengths["count"].to_numpy().astype(np.float64)
    sizes = lengths["length"].to_numpy().astype(np.float64)
    mean = float(np.average(sizes, weights=weights))
    spread = float(np.sqrt(np.average((sizes - mean) ** 2, weights=weights)))
    chosen = pl.col("kind") == kind
    return EncodedColumn(
        kind=kind,
        share=matched / total,
        shares={k: v / total for k, v in sorted(counts.items())},
        n_used=total,
        distinct=distinct.height,
        mean_length=mean,
        length_std=spread,
        examples=tuple(
            examples(classified, chosen, params.max_examples)
            if kind != "opaque_token"
            else examples(
                classified.filter(pl.col("kind").is_null()),
                value.str.contains(_WHITESPACE).not_()
                & (length >= params.min_token_length),
                params.max_examples,
            )
        ),
    )


def _tokens(unmatched: pl.DataFrame, params: EncodedColumnsParams) -> int:
    """Rows holding an opaque token among the values no format matched.

    Candidates have no whitespace and at least ``min_token_length`` characters. They
    are tokens when, pooled, their characters have an entropy of at least
    ``min_entropy`` bits (a random string of letters and digits has about 5.9, text
    about 4) and their lengths have a standard deviation of at most
    ``length_std_threshold``. The pool is a seeded sample of at most 2 000 distinct
    candidates.
    """
    candidates = unmatched.filter(
        ~pl.col("value").str.contains(_WHITESPACE)
        & (pl.col("value").str.len_chars() >= params.min_token_length)
    )
    if candidates.is_empty():
        return 0
    sample = (
        candidates
        if candidates.height <= _TOKEN_SAMPLE
        else candidates.sample(_TOKEN_SAMPLE, seed=_SEED)
    )
    values = sample["value"].to_list()
    if float(np.std([len(v) for v in values])) > params.length_std_threshold:
        return 0
    if _entropy("".join(values)) < params.min_entropy:
        return 0
    return int(candidates["count"].sum())


def _entropy(value: str) -> float:
    """Shannon entropy, in bits, of the characters of ``value``."""
    n = len(value)
    return -sum(c / n * math.log2(c / n) for c in Counter(value).values())


_KIND_NAMES: dict[str, str] = {
    "uuid": "UUIDs",
    "hex": "hexadecimal digests or keys",
    "base64": "base64 strings",
    "opaque_token": "random-looking tokens of one length",
}


def _report(ctx: CheckContext, name: str, found: EncodedColumn) -> None:
    what = _KIND_NAMES[found.kind]
    strict = found.kind in ("uuid", "hex")
    shown = ", ".join(repr(v) for v in found.examples)
    ctx.finding(
        code="encoded_values",
        subject=Subject.column(name),
        severity=Severity.INFO,
        confidence=Confidence.HIGH if strict else Confidence.MODERATE,
        title=f"Values look like {what}",
        message=(
            f"{found.share:.0%} of the {found.n_used:,} values of {name!r} look like "
            f"{what} (e.g. {shown}; mean length {found.mean_length:.1f}). Such "
            "values name or fingerprint a record and carry no meaning of their own, "
            "so as a model feature they let it memorise rows; check whether the "
            "column is a key to drop, a code to decode or a field to leave out."
        ),
        evidence=Evidence(
            metrics={
                "kind": found.kind,
                "encoded_share": found.share,
                "distinct": found.distinct,
                "mean_length": found.mean_length,
                "length_std": found.length_std,
                **{f"share_{k}": v for k, v in found.shares.items()},
            },
            n_used=found.n_used,
            n_total=ctx.profile.n_rows,
            method="value_formats",
        ),
    )
