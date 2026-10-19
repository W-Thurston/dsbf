# dsbf/legacy/findings.py
"""Turn a legacy task's guidance blurbs into ``Finding``s."""

import hashlib
import re
from collections import Counter
from typing import Any

from dsbf.engine import CheckContext
from dsbf.model import (
    Action,
    Confidence,
    Evidence,
    Guidance,
    Severity,
    Subject,
)

DATASET_KEY = "__dataset__"
_ACTION_FIELDS = ("action", "method", "condition", "detail")
_LEVELS = {"info": Severity.INFO, "warn": Severity.WARN, "error": Severity.ERROR}
_STEM_LENGTH = 60

_PARENTHESES = re.compile(r"\([^)]*\)")
_NUMBER = re.compile(r"[-+]?\d[\d.,]*%?")
_NOT_WORD = re.compile(r"[^a-z]+")


def findings_from(ctx: CheckContext, guidance: dict[str, Any] | None) -> None:
    """Record a finding for each blurb in ``guidance`` that is within the check's scope.

    A blurb about a column the check does not apply to is left out, and one note says
    how many and which. The legacy blurbs carry no code, confidence or row count, so
    this supplies them; see ``_code`` and ``_evidence``.
    """
    scope = set(ctx.columns)
    seen: Counter[tuple[str, str]] = Counter()
    left_out: dict[str, int] = {}
    for column, phases in (guidance or {}).items():
        for blurbs in phases.values():
            for blurb in blurbs:
                if column != DATASET_KEY and column not in scope:
                    left_out[column] = left_out.get(column, 0) + 1
                    continue
                code = _code(blurb)
                seen[column, code] += 1
                if seen[column, code] > 1:
                    code = f"{code}_{seen[column, code]}"
                ctx.finding(
                    code=code,
                    subject=Subject.dataset()
                    if column == DATASET_KEY
                    else Subject.column(column),
                    severity=_LEVELS.get(blurb["level"], Severity.INFO),
                    confidence=Confidence.MODERATE,
                    title=blurb["title"],
                    message=blurb["body"],
                    evidence=_evidence(ctx, blurb),
                    guidance=Guidance(
                        phase=blurb["phase"],
                        actions=tuple(_action(a) for a in blurb["actions"]),
                    ),
                    tags=_tags(blurb["level"]),
                )
    if left_out:
        count = sum(left_out.values())
        ctx.note(
            "findings_out_of_scope",
            f"Left out {count} finding(s) about columns {ctx.spec.id} does not apply "
            "to under the profile's types.",
            detail=", ".join(sorted(left_out)),
        )


def _code(blurb: dict[str, Any]) -> str:
    """The phase and what the title says, with the numbers taken out.

    A title such as ``Heavy Right Skew (+4.78)`` keeps its words and loses its value,
    so the code does not follow the value or the order the task emits blurbs in.
    """
    title = _NUMBER.sub(" ", _PARENTHESES.sub(" ", blurb["title"].lower()))
    stem = _NOT_WORD.sub("_", title).strip("_") or "finding"
    if len(stem) > _STEM_LENGTH:
        digest = hashlib.sha256(stem.encode()).hexdigest()[:8]
        stem = f"{stem[:_STEM_LENGTH].rstrip('_')}_{digest}"
    return f"{blurb['phase']}_{stem}"


def _tags(level: str) -> tuple[str, ...]:
    if level == "good":
        return ("good",)
    return () if level in _LEVELS else (f"legacy_level_{level}",)


def _evidence(ctx: CheckContext, blurb: dict[str, Any]) -> Evidence:
    n_total = ctx.profile.n_rows
    n = blurb["metric"].get("n")
    used = n if isinstance(n, int) and not isinstance(n, bool) else n_total
    return Evidence(
        metrics=blurb["metric"],
        n_used=used if 0 <= used <= n_total else n_total,
        n_total=n_total,
        method=ctx.spec.id,
    )


def _action(legacy: dict[str, Any]) -> Action:
    columns = (
        [legacy["column"]] if "column" in legacy else list(legacy.get("columns", ()))
    )
    known = {*_ACTION_FIELDS, "column", "columns"}
    return Action.model_validate(
        {
            **{key: legacy.get(key) for key in _ACTION_FIELDS},
            "columns": columns,
            "params": {k: v for k, v in legacy.items() if k not in known},
        }
    )
