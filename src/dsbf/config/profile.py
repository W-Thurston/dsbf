# dsbf/config/profile.py
"""The settings of a profiling run: what ``dsbf.profile()`` and the CLI both build."""

import math
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dsbf.engine.check import RunsIn
from dsbf.engine.selection import Depth

DEFAULT_PACKS = ("core",)


class SampleStrategy(StrEnum):
    """How the rows of a sample are chosen."""

    HEAD = "head"
    RANDOM = "random"
    STRATIFIED = "stratified"


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SampleConfig(_Section):
    """Profile a sample of the rows rather than all of them."""

    rows: int = Field(ge=1, description="How many rows to keep.")
    strategy: SampleStrategy = Field(
        default=SampleStrategy.HEAD, description="How the rows are chosen."
    )
    seed: int | None = Field(
        default=None, description="Seeds a random or stratified sample."
    )
    stratify_by: tuple[str, ...] = Field(
        default=(), description="The columns whose groups a stratified sample keeps."
    )


class LimitsConfig(_Section):
    """How long a check, and the whole run, may take; unset leaves a limit off."""

    check_timeout_seconds: float | None = Field(
        default=None, gt=0, description="Seconds one check may take."
    )
    run_budget_seconds: float | None = Field(
        default=None, gt=0, description="Seconds the whole run may take."
    )

    @field_validator("check_timeout_seconds", "run_budget_seconds")
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("must be a finite number")
        return value


class CheckSettings(_Section):
    """What one run changes about one check; unset keeps the check's own."""

    timeout_seconds: float | None = Field(
        default=None, gt=0, description="Seconds this check may take."
    )
    runs_in: RunsIn | None = Field(
        default=None,
        description="`process` to run the check in a worker process, which a timeout "
        "can kill; `thread` for a thread.",
    )

    @field_validator("timeout_seconds")
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("must be a finite number")
        return value


class CacheConfig(_Section):
    """Keep the results of checks between runs under the DSBF home."""

    enabled: bool = Field(
        default=False, description="Reuse results of checks whose inputs are unchanged."
    )
    max_megabytes: int | None = Field(
        default=500,
        ge=0,
        description="Evict the least recently used entries past this size; "
        "unset keeps all.",
    )
    max_age_days: float | None = Field(
        default=90,
        gt=0,
        description="Evict entries not used for this long; unset keeps all.",
    )

    @field_validator("max_age_days")
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("must be a finite number")
        return value


class StoreConfig(_Section):
    """What the run store keeps beside a run."""

    snapshot_rows: int = Field(
        default=10_000,
        ge=0,
        description="Rows of the analysed data to keep as a snapshot, sampled "
        "(seed 0) above this; 0 keeps none.",
    )


class ProfileConfig(_Section):
    """Every setting of one profiling run, with its default."""

    dataset: str | None = Field(
        default=None,
        description="The key that groups this run with earlier runs of the same "
        "dataset. Unset, a file's absolute path is the key, and an in-memory table "
        "has none.",
    )
    packs: tuple[str, ...] = Field(
        default=DEFAULT_PACKS, description="The packs whose checks are candidates."
    )
    include: tuple[str, ...] = Field(
        default=(), description="Checks to add by id, whatever the filters say."
    )
    exclude: tuple[str, ...] = Field(default=(), description="Checks to remove by id.")
    depth: Depth = Field(
        default=Depth.STANDARD, description="The most expensive checks to allow."
    )
    target: str | None = Field(
        default=None, description="The column to predict, for the checks that need one."
    )
    time_index: str | None = Field(
        default=None, description="The datetime column the time series checks use."
    )
    sample: SampleConfig | None = None
    limits: LimitsConfig = LimitsConfig()
    workers: int | None = Field(
        default=None, ge=1, description="Checks to run at once; unset is one per core."
    )
    check_settings: dict[str, CheckSettings] = Field(
        default_factory=dict,
        description="A timeout or `runs_in` for single checks, by check id.",
    )
    cache: CacheConfig = CacheConfig()
    store: StoreConfig = StoreConfig()
    plugins: tuple[str, ...] = Field(
        default=(), description="Directories of check plugins to load."
    )
    params: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Parameters of checks, by check id; they win over `target` "
        "and `time_index`.",
    )

    @model_validator(mode="after")
    def _no_check_both_ways(self) -> "ProfileConfig":
        both = sorted(set(self.include) & set(self.exclude))
        if both:
            raise ValueError(f"checks both included and excluded: {', '.join(both)}")
        return self
