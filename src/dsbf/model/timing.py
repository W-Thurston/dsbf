# dsbf/model/timing.py
"""Resource accounting for one check execution."""

from pydantic import AwareDatetime, Field

from dsbf.model.base import Contract, FiniteFloat


class Timing(Contract):
    """Wall and CPU time and memory for one check, as measured by the executor."""

    wall_seconds: FiniteFloat = Field(
        ge=0, description="Time the executor spent on the check, from its start."
    )
    cpu_seconds: FiniteFloat | None = Field(
        default=None,
        ge=0,
        description=(
            "CPU time of the whole process while the check ran, so work on other "
            "threads counts too; approximate."
        ),
    )
    peak_memory_bytes: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Peak resident memory of the whole process while the check ran, "
            "including the loaded data, found by sampling; approximate, and a "
            "short spike between samples is missed."
        ),
    )
    started_at: AwareDatetime | None = None
