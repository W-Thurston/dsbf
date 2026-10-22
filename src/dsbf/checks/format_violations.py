# dsbf/checks/format_violations.py
"""Values that break a format the caller declares, and dates written several ways."""

import re
from typing import Literal

import polars as pl
from pydantic import Field, field_validator, model_validator

from dsbf.checks.strings import (
    STRING_TYPES,
    distinct_values,
    examples,
    holds_strings,
    value_kind,
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
from dsbf.profiling.types import DATE_FORMATS

Mode = Literal["fullmatch", "search"]

_SIGNATURE_RUNS = ((r"[A-Z]+", "A"), (r"[a-z]+", "a"), (r"[0-9]+", "9"))
# A long loop calls ``checkpoint`` once in this many values.
_CHECKPOINT_EVERY = 10_000


class FormatRule(Contract):
    """A pattern the values of one column must match."""

    column: str = Field(min_length=1, description="The column the rule is about.")
    pattern: str = Field(
        min_length=1, description="A Python regular expression (the `re` module)."
    )
    mode: Mode = Field(
        "fullmatch",
        description="`fullmatch` needs the whole value to match; `search` needs the "
        "pattern to occur in it.",
    )

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, pattern: str) -> str:
        try:
            re.compile(pattern)
        except re.error as error:
            raise ValueError(f"not a regular expression: {error}") from error
        return pattern


class FormatViolationsParams(CheckParams):
    rules: tuple[FormatRule, ...] = Field(
        (),
        description="The rules to check, one per column. With none, only the "
        "mixed date formats are reported.",
    )
    check_date_formats: bool = Field(
        True, description="Report string columns that write dates in several formats."
    )
    min_date_share: float = Field(
        0.9,
        gt=0,
        le=1,
        description="Share of a column's values that must be dates for it to be "
        "read as a column of dates.",
    )
    min_format_share: float = Field(
        0.05,
        gt=0,
        le=0.5,
        description="Share of the values a date format needs to count as in use.",
    )
    min_values: int = Field(
        10, ge=1, description="Values a column needs for the date formats."
    )
    warn_share: float = Field(
        0.01,
        ge=0,
        le=1,
        description="Share of a column's values that break its rule above which "
        "the finding is a warning.",
    )
    max_examples: int = Field(5, ge=1, le=20, description="Examples listed per rule.")

    @model_validator(mode="after")
    def _one_rule_per_column(self) -> "FormatViolationsParams":
        seen: set[str] = set()
        for rule in self.rules:
            if rule.column in seen:
                raise ValueError(
                    f"column {rule.column!r} has two rules; join the patterns "
                    "with `|` into one"
                )
            seen.add(rule.column)
        return self


class RuleResult(Contract):
    column: str
    pattern: str
    mode: Mode
    n_used: int = Field(ge=0, description="Non-null values checked.")
    violations: int = Field(ge=0)
    share: float = Field(ge=0, le=1)
    distinct_violations: int = Field(ge=0)
    examples: tuple[str, ...]


class DateFormat(Contract):
    format: str = Field(description="The `strptime` format.")
    signature: str = Field(
        description="The shape of an example: runs of capitals, lowercase letters "
        "and digits become `A`, `a` and `9`, as in the profile."
    )
    rows: int = Field(ge=1)
    share: float = Field(gt=0, le=1, description="Of the column's non-null values.")
    examples: tuple[str, ...]


class MixedDates(Contract):
    n_used: int = Field(ge=1, description="Values counted, null tokens excluded.")
    date_share: float = Field(gt=0, le=1)
    formats: tuple[DateFormat, ...] = Field(description="Most used first.")


class FormatViolationsOutput(CheckOutputModel):
    rules: tuple[RuleResult, ...] = Field(description="The result of every rule run.")
    unchecked_rules: dict[str, str] = Field(
        description="Rules not run, by column, with the reason."
    )
    mixed_dates: dict[str, MixedDates] = Field(
        description="Each string column that writes dates in several formats."
    )
    columns_read: int = Field(ge=0, description="String columns read for dates.")


@check(
    id="format_violations",
    version="1.0.0",
    kind="analysis",
    # Every column, so a rule about a column that is missing is known to be so.
    applies_to=SemanticTypes(),
    dimensions=[Dimension.VALIDITY],
    params=FormatViolationsParams,
    output=FormatViolationsOutput,
    requires=Requires(profile=["counts"]),
    # One regular expression per distinct value of a ruled column, and the date
    # formats for the values of other string columns that could be dates.
    cost=Cost.CHEAP,
    packs=["core"],
    tags=["format", "regex", "date"],
)
def format_violations(ctx: CheckContext, params: FormatViolationsParams) -> CheckOutput:
    """Count the values that break each rule, and find dates written two ways."""
    rules: list[RuleResult] = []
    unchecked: dict[str, str] = {}
    ruled: set[str] = set()
    for rule in params.rules:
        ctx.checkpoint()
        if rule.column not in ctx.columns:
            unchecked[rule.column] = "no such column"
            continue
        profile = ctx.profile.column(rule.column)
        if not holds_strings(profile):
            unchecked[rule.column] = f"stored as {profile.physical_dtype}, not strings"
            continue
        ruled.add(rule.column)
        result = _apply(ctx, rule, params)
        rules.append(result)
        _report_rule(ctx, result, params, profile.counts.rows)
    for column, why in unchecked.items():
        ctx.note(
            "rule_not_run",
            f"The rule for {column!r} was not run: {why}.",
            hint="Check the column name and that the column holds strings.",
        )

    mixed: dict[str, MixedDates] = {}
    read = 0
    if params.check_date_formats:
        for name in ctx.columns:
            ctx.checkpoint()
            profile = ctx.profile.column(name)
            if not holds_strings(profile) or profile.semantic_type not in (
                *STRING_TYPES,
                "datetime",
            ):
                continue
            if profile.counts.non_null < params.min_values:
                continue
            read += 1
            found = _mixed_dates(ctx, name, params)
            if found is not None:
                mixed[name] = found
                _report_dates(ctx, name, found)
    if not params.rules and not read:
        return ctx.stop(
            CheckStatus.NOT_APPLICABLE,
            "nothing_to_check",
            "There are no rules, and no string column with enough values to look "
            "for mixed date formats.",
            severity=Severity.INFO,
            hint="Set the `rules` parameter to a list of `{column, pattern}` entries.",
        )
    return ctx.output(
        FormatViolationsOutput(
            rules=tuple(rules),
            unchecked_rules=unchecked,
            mixed_dates=mixed,
            columns_read=read,
        )
    )


def _apply(
    ctx: CheckContext, rule: FormatRule, params: FormatViolationsParams
) -> RuleResult:
    distinct = distinct_values(ctx.values(rule.column))
    regex = re.compile(rule.pattern)
    match = regex.fullmatch if rule.mode == "fullmatch" else regex.search
    n_used = int(distinct["count"].sum())
    broken: list[tuple[str, int]] = []
    for position, (value, count) in enumerate(
        zip(distinct["value"], distinct["count"], strict=True)
    ):
        if position % _CHECKPOINT_EVERY == 0:
            ctx.checkpoint()
        if match(value) is None:
            broken.append((value, count))
    broken.sort(key=lambda item: (-item[1], item[0]))
    violations = sum(count for _, count in broken)
    return RuleResult(
        column=rule.column,
        pattern=rule.pattern,
        mode=rule.mode,
        n_used=n_used,
        violations=violations,
        share=violations / n_used if n_used else 0.0,
        distinct_violations=len(broken),
        examples=tuple(_cut(value) for value, _ in broken[: params.max_examples]),
    )


def _cut(value: str, width: int = 80) -> str:
    return value if len(value) <= width else value[: width - 3] + "..."


def _signature(value: str) -> str:
    for pattern, symbol in _SIGNATURE_RUNS:
        value = re.sub(pattern, symbol, value)
    return value


def _mixed_dates(
    ctx: CheckContext, name: str, params: FormatViolationsParams
) -> MixedDates | None:
    kinded = value_kind(distinct_values(ctx.values(name)), DATE_FORMATS)
    kinded = kinded.filter(pl.col("kind") != "null_token")
    total = int(kinded["count"].sum())
    if not total:
        return None
    dated = kinded.filter(pl.col("kind") == "date")
    date_rows = int(dated["count"].sum())
    if date_rows / total < params.min_date_share:
        return None
    by_format = (
        dated.group_by("format")
        .agg(rows=pl.col("count").sum())
        .filter(pl.col("rows") / total >= params.min_format_share)
        .sort(["rows", "format"], descending=[True, False])
    )
    if by_format.height < 2:
        return None
    formats = []
    for row in by_format.iter_rows(named=True):
        shown = examples(dated, pl.col("format") == row["format"], 3)
        formats.append(
            DateFormat(
                format=row["format"],
                signature=_signature(shown[0]),
                rows=row["rows"],
                share=row["rows"] / total,
                examples=tuple(shown),
            )
        )
    return MixedDates(
        n_used=total, date_share=date_rows / total, formats=tuple(formats)
    )


def _report_rule(
    ctx: CheckContext, result: RuleResult, params: FormatViolationsParams, rows: int
) -> None:
    if not result.violations:
        return
    shown = ", ".join(repr(v) for v in result.examples[:3])
    ctx.finding(
        code="format_violations",
        subject=Subject.column(result.column),
        severity=Severity.WARN if result.share > params.warn_share else Severity.INFO,
        confidence=Confidence.HIGH,
        title=f"{result.violations:,} values break the format rule",
        message=(
            f"{result.violations:,} of {result.n_used:,} values of "
            f"{result.column!r} ({result.share:.1%}) do not {result.mode} the pattern "
            f"{result.pattern!r}, e.g. {shown}. Violations come from entry mistakes, "
            "a change upstream, or free text in a structured field; see whether "
            "they can be normalised (trimmed, case-folded) or need review."
        ),
        evidence=Evidence(
            metrics={
                "violations": result.violations,
                "violation_share": result.share,
                "distinct_violations": result.distinct_violations,
                "mode": result.mode,
            },
            n_used=result.n_used,
            n_total=rows,
            method=f"regex_{result.mode}",
        ),
    )


def _report_dates(ctx: CheckContext, name: str, found: MixedDates) -> None:
    summary = "; ".join(
        f"{f.signature} ({f.share:.0%}, e.g. {f.examples[0]!r})" for f in found.formats
    )
    ctx.finding(
        code="mixed_date_formats",
        subject=Subject.column(name),
        severity=Severity.WARN,
        confidence=Confidence.HIGH,
        title=f"Dates are written in {len(found.formats)} formats",
        message=(
            f"{found.date_share:.0%} of the values of {name!r} are dates, written "
            f"in {len(found.formats)} formats: {summary}. A parser given one format "
            "turns the others into nulls, and a day-first and a month-first "
            "format can read the same text as different dates. Convert them all "
            "to one format (ISO 8601) at the source."
        ),
        evidence=Evidence(
            metrics={
                "formats": len(found.formats),
                "date_share": found.date_share,
                "smallest_share": min(f.share for f in found.formats),
            },
            n_used=found.n_used,
            n_total=ctx.profile.n_rows,
            method="date_formats",
        ),
    )
