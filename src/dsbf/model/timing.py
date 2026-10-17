# dsbf/model/timing.py
"""Resource accounting for one check execution."""

from pydantic import AwareDatetime, Field

from dsbf.model.base import Contract, FiniteFloat


class Timing(Contract):
    """Wall and CPU time and memory for one check, as measured by the executor."""

    wall_seconds: FiniteFloat = Field(ge=0)
    cpu_seconds: FiniteFloat | None = Field(default=None, ge=0)
    peak_memory_bytes: int | None = Field(
        default=None,
        ge=0,
        description="Peak memory by sampling; approximate.",
    )
    started_at: AwareDatetime | None = None
