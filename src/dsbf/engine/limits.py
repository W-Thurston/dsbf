# dsbf/engine/limits.py
"""How long the executor lets a check, and the whole run, take."""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    """Time limits for a run; ``None`` leaves a limit off.

    A check that passes ``check_timeout_seconds`` ends as ``timeout``, and so does the
    check running when ``run_budget_seconds`` is used up; checks not yet started are
    ``skipped``. ``cancel_grace_seconds`` is how long the executor waits after a
    timeout for the check to notice (see ``CheckContext.checkpoint``) before it
    abandons the check's thread.
    """

    check_timeout_seconds: float | None = None
    run_budget_seconds: float | None = None
    cancel_grace_seconds: float = 0.25

    def __post_init__(self) -> None:
        for name in ("check_timeout_seconds", "run_budget_seconds"):
            value = getattr(self, name)
            if value is not None and not (math.isfinite(value) and value > 0):
                raise ValueError(f"{name} must be a positive number, not {value!r}")
        grace = self.cancel_grace_seconds
        if not (math.isfinite(grace) and grace >= 0):
            raise ValueError(f"cancel_grace_seconds must be 0 or more, not {grace!r}")
