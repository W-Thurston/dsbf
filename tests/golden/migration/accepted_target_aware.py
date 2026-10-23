"""What differs between `mutual_information` and the task it replaces.

See ``accepted.py`` for what an entry is. The task needed a target the golden runs do
not give it; the core check is given one on the dataset that plants a target.
"""

from __future__ import annotations

from tests.golden.migration.harness import Accepted

NO_TARGET = (
    "The task returns a successful, empty result when no target is configured, and no "
    "golden dataset gives the legacy tasks one. With no target set, "
    "`mutual_information` has nothing to measure against and ends `not_applicable`, "
    "which says so in a note rather than reporting success with nothing in it."
)
CORE_ONLY_TARGET = (
    "kitchen5k plants a target and the columns that leak it. The engine golden run "
    "gives the core checks `target_for_leakage` through `core_params`, and the legacy "
    "tasks run without one, so the task scored nothing and the check measured every "
    "numeric, categorical and boolean column of 2 to 50 values against it."
)
PLANTED_COLUMN = (
    "The column is a copy, a recode or a noisy function of the target, so it removes "
    "all of the target's uncertainty. The task, given no target, had nothing to say; "
    "the check reports it as informative and points to `target_leakage`."
)

_WITHOUT_TARGET = (
    "all_categorical",
    "clean",
    "high_missingness",
    "near_clean",
    "severe_multicollinearity",
    "single_column",
    "sms_spam",
    "tiny",
    "titanic",
    "wide",
)

ACCEPTED_TARGET_AWARE: tuple[Accepted, ...] = (
    *(
        Accepted(
            "compute_mutual_information",
            dataset,
            "",
            "status",
            "legacy ok, core not_applicable",
            NO_TARGET,
        )
        for dataset in _WITHOUT_TARGET
    ),
    Accepted(
        "compute_mutual_information",
        "kitchen5k",
        "",
        "target",
        "legacy had no target, core measured 37 features against 'target_for_leakage'",
        CORE_ONLY_TARGET,
    ),
    *(
        Accepted(
            "compute_mutual_information",
            "kitchen5k",
            column,
            "finding",
            "core only (info)",
            PLANTED_COLUMN,
        )
        for column in ("future_leakage", "perfect_leakage", "subtle_leakage")
    ),
)
