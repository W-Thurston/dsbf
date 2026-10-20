# dsbf/config/profile.py
"""The settings of a profiling run: what ``dsbf.profile()`` and the CLI both build."""

import math
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dsbf.engine.selection import Depth

DEFAULT_PACKS = ("core", "legacy")


class Backend(StrEnum):
    """The frame library the legacy tasks are handed."""

    POLARS = "polars"
    PANDAS = "pandas"


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
    strategy: SampleStrategy = SampleStrategy.HEAD
    seed: int | None = Field(
        default=None, description="Seeds a random or stratified sample."
    )
    stratify_by: tuple[str, ...] = Field(
        default=(), description="The columns whose groups a stratified sample keeps."
    )


class LimitsConfig(_Section):
    """How long a check, and the whole run, may take; unset leaves a limit off."""

    check_timeout_seconds: float | None = Field(default=None, gt=0)
    run_budget_seconds: float | None = Field(default=None, gt=0)

    @field_validator("check_timeout_seconds", "run_budget_seconds")
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("must be a finite number")
        return value


class ProfileConfig(_Section):
    """Every setting of one profiling run, with its default."""

    packs: tuple[str, ...] = Field(
        default=DEFAULT_PACKS, description="The packs whose checks are candidates."
    )
    include: tuple[str, ...] = Field(
        default=(), description="Checks to add by id, whatever the filters say."
    )
    exclude: tuple[str, ...] = Field(default=(), description="Checks to remove by id.")
    depth: Depth = Depth.STANDARD
    target: str | None = Field(
        default=None, description="The column to predict, for the checks that need one."
    )
    time_index: str | None = Field(
        default=None, description="The datetime column the time series checks use."
    )
    sample: SampleConfig | None = None
    backend: Backend = Backend.PANDAS
    limits: LimitsConfig = LimitsConfig()
    workers: int | None = Field(
        default=None, ge=1, description="Checks to run at once; unset is one per core."
    )
    plugins: tuple[str, ...] = Field(
        default=(), description="Directories of check plugins to load."
    )
    tasks: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Per-task settings of the legacy tasks, by task name.",
    )

    @model_validator(mode="after")
    def _no_check_both_ways(self) -> "ProfileConfig":
        both = sorted(set(self.include) & set(self.exclude))
        if both:
            raise ValueError(f"checks both included and excluded: {', '.join(both)}")
        return self
