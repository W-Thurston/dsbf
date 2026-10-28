# dsbf/checks/text_encoding.py
"""Text that was damaged when it was decoded."""

from pydantic import Field

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
    Confidence,
    Contract,
    Cost,
    Dimension,
    Evidence,
    Severity,
    Subject,
)

_REPLACEMENT = "U+FFFD"


class TextEncodingParams(CheckParams):
    warn_share: float = Field(
        0.01,
        ge=0,
        le=1,
        description="Share of a column's non-null values above which damaged "
        "values are a warning.",
    )


class ColumnEncoding(Contract):
    name_damaged: bool
    values_with_replacement_char: int = Field(ge=0)
    values_with_nul: int = Field(ge=0)


class TextEncodingOutput(CheckOutputModel):
    columns: dict[str, ColumnEncoding] = Field(
        description="Each column with a damaged name or damaged values, by name."
    )


@check(
    id="text_encoding",
    version="1.0.0",
    kind="rule",
    applies_to=SemanticTypes(),
    dimensions=[Dimension.VALIDITY],
    params=TextEncodingParams,
    output=TextEncodingOutput,
    requires=Requires(profile=["counts"]),
    cost=Cost.CHEAP,
    codes=["damaged_column_name", "nul_characters", "replacement_characters"],
    packs=["core"],
    tags=["encoding", "text"],
)
def text_encoding(ctx: CheckContext, params: TextEncodingParams) -> CheckOutput:
    """Report names and values that hold U+FFFD or NUL characters.

    U+FFFD is what a lossy decode leaves for bytes it could not read; NUL is what a
    UTF-16 file read as UTF-8 leaves between its characters.
    """
    columns: dict[str, ColumnEncoding] = {}
    for column in ctx.profile.columns:
        replacement = column.values_with_replacement_char or 0
        nul = column.values_with_nul or 0
        if not (column.name_encoding_damage or replacement or nul):
            continue
        columns[column.name] = ColumnEncoding(
            name_damaged=column.name_encoding_damage,
            values_with_replacement_char=replacement,
            values_with_nul=nul,
        )
        subject = Subject.column(column.name)
        counts = column.counts
        if column.name_encoding_damage:
            ctx.finding(
                code="damaged_column_name",
                subject=subject,
                severity=Severity.WARN,
                confidence=Confidence.HIGH,
                title="Column name holds undecodable characters",
                message=(
                    f"The name {column.name!r} holds U+FFFD or NUL characters, so "
                    "the file was probably read in the wrong encoding."
                ),
                evidence=Evidence(
                    metrics={"name_damaged": True},
                    n_used=1,
                    n_total=1,
                    method="name_characters",
                ),
            )
        for code, count, what, hint in (
            (
                "replacement_characters",
                replacement,
                f"the replacement character {_REPLACEMENT}",
                "bytes the decoder could not read were replaced, and the original "
                "text is gone",
            ),
            (
                "nul_characters",
                nul,
                "NUL characters",
                "a UTF-16 file read as UTF-8 puts one between every two characters",
            ),
        ):
            if not count:
                continue
            share = count / counts.non_null
            ctx.finding(
                code=code,
                subject=subject,
                severity=Severity.WARN if share > params.warn_share else Severity.INFO,
                confidence=Confidence.HIGH,
                title=f"{count:,} values hold {what} ({share:.1%})",
                message=(
                    f"{count:,} of {counts.non_null:,} non-null values of "
                    f"{column.name!r} hold {what} ({share:.1%}): {hint}. Re-read "
                    "the source with its real encoding."
                ),
                evidence=Evidence(
                    metrics={"damaged_values": count, "damaged_share": share},
                    n_used=counts.non_null,
                    n_total=counts.rows,
                    method="value_characters",
                ),
            )
    return ctx.output(TextEncodingOutput(columns=columns))
