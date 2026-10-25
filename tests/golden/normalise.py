"""Turn core findings into a stable snapshot that is practical to review.

A raw ``Run`` is megabytes of floats, timings and ids. The snapshot keeps what a
behaviour change should be judged on, drops anything that differs between two identical
runs, rounds floats, and renders with fixed formatting so identical content is
byte-identical.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any

SIG_FIGS = 6

# Keys that hold timings, memory or timestamps wherever they appear.
_VOLATILE_KEY = re.compile(
    r"memory|duration|elapsed|timestamp|^(start|end)_time$", re.IGNORECASE
)

_NON_FINITE = {"nan": "NaN", "inf": "Infinity", "-inf": "-Infinity"}


def round_sig(value: float, digits: int = SIG_FIGS) -> float | str:
    """Round to ``digits`` significant figures; spell out NaN and infinities."""
    if math.isnan(value):
        return _NON_FINITE["nan"]
    if math.isinf(value):
        return _NON_FINITE["inf"] if value > 0 else _NON_FINITE["-inf"]
    if value == 0:
        return 0.0
    return float(f"{value:.{digits - 1}e}")


def canonical(value: Any) -> Any:
    """Return ``value`` with floats rounded, volatile keys dropped and keys sorted."""
    if isinstance(value, bool) or value is None or isinstance(value, int):
        return value
    if isinstance(value, float):
        return round_sig(value)
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return {
            str(k): canonical(v)
            for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))
            if not _VOLATILE_KEY.search(str(k))
        }
    if isinstance(value, list | tuple):
        return [canonical(v) for v in value]
    return str(value)


def render_snapshot(snapshot: dict[str, Any]) -> str:
    """Serialise with fixed formatting so identical content is byte-identical."""
    return json.dumps(snapshot, indent=1, sort_keys=True, ensure_ascii=False) + "\n"
