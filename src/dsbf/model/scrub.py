# dsbf/model/scrub.py
"""Turn values from the analysis libraries into plain JSON values."""

import math
from datetime import date, datetime
from typing import Any


def json_safe(value: Any, *, null_non_finite: bool = True) -> tuple[Any, bool]:
    """Return the JSON-safe form of ``value`` and whether it held an undefined number.

    NumPy scalars become Python scalars and dates become ISO 8601 strings. Tuples
    and lists become lists, and dict keys must be strings. NaN, infinity and
    ``NaT`` become ``None`` (the second result is then ``True``), or raise
    ``ValueError`` when ``null_non_finite`` is false. A value of any other type
    raises ``ValueError``: it is a producer bug, not a property of the data.
    """
    item = getattr(value, "item", None)
    if callable(item) and not isinstance(value, (bool, int, float, str, datetime)):
        value = item()  # NumPy scalar -> Python scalar
    if value is None or isinstance(value, (bool, int, str)):
        return value, False
    if isinstance(value, float):
        if math.isfinite(value):
            return value, False
        return _undefined(value, null_non_finite)
    if isinstance(value, datetime):
        if value != value:  # NaT
            return _undefined(value, null_non_finite)
        return value.isoformat(), False
    if isinstance(value, date):
        return value.isoformat(), False
    if isinstance(value, (list, tuple)):
        items = [json_safe(v, null_non_finite=null_non_finite) for v in value]
        return [v for v, _ in items], any(undefined for _, undefined in items)
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        undefined = False
        for key, member in value.items():
            if not isinstance(key, str):
                raise ValueError(f"keys must be strings, not {type(key).__name__}")
            out[key], member_undefined = json_safe(
                member, null_non_finite=null_non_finite
            )
            undefined = undefined or member_undefined
        return out, undefined
    raise ValueError(f"{type(value).__name__} is not a JSON value")


def _undefined(value: Any, null_non_finite: bool) -> tuple[None, bool]:
    if not null_non_finite:
        raise ValueError(f"{value!r} is not a finite number")
    return None, True
