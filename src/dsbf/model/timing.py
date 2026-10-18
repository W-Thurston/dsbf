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
            "CPU time while the check ran; approximate. For a check on a thread "
            "this is the whole process's, so work on other threads counts too, "
            "and `shared_with` says whether checks overlapped. For a check in a "
            "worker process it is the worker's own, and missing if the worker "
            "was killed or died."
        ),
    )
    peak_memory_bytes: int | None = Field(
        default=None,
        ge=0,
        description=(
            "Peak resident memory while the check ran, found by sampling; "
            "approximate, and a short spike between samples is missed. For a "
            "check on a thread it is the whole process's, including the loaded "
            "data, and shared with any check that overlapped (`shared_with`). "
            "For a check in a worker process it is the worker's own, including "
            "what it mapped of the data."
        ),
    )
    shared_with: int = Field(
        default=0,
        ge=0,
        description=(
            "How many other checks on threads ran at the same time as this one. "
            "While any did, `cpu_seconds` and `peak_memory_bytes` are the whole "
            "process's and cannot be told apart per check. Always 0 for a check "
            "in a worker process."
        ),
    )
    cached: bool = Field(
        default=False,
        description=(
            "True when the result came from the result cache: the check did not "
            "run in this run, and the timing is that of reading the entry."
        ),
    )
    started_at: AwareDatetime | None = None
