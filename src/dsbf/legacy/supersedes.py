# dsbf/legacy/supersedes.py
"""Which legacy tasks a core check replaces, and how a plan keeps their inputs."""

from collections.abc import Collection, Mapping
from dataclasses import replace
from typing import Any

from dsbf.checks import CORE_PACK
from dsbf.engine import Plan, Registry, Selection, make_plan
from dsbf.legacy.adapter import UNDECLARED_READS
from dsbf.model import DatasetProfile

# Legacy task to the core checks that replace it. A unit that migrates a task adds it
# here, and the mapping goes when the last task does. A task whose findings are
# several checks' (`summarize_numeric` raised a near-zero-variance and a mean–median
# finding) is replaced only when all of them are selected, or a finding would vanish.
SUPERSEDED_BY: Mapping[str, tuple[str, ...]] = {
    "detect_zeros": ("zeros",),
    "summarize_nulls": ("null_rate",),
    "detect_constant_columns": ("constant_columns",),
    "detect_high_cardinality": ("high_cardinality",),
    "detect_id_columns": ("id_columns",),
    "detect_single_dominant_value": ("dominant_value",),
    "summarize_boolean_fields": ("dominant_value",),
    "detect_near_zero_variance": ("near_zero_variance",),
    "detect_duplicates": ("duplicate_rows",),
    "detect_duplicate_columns": ("duplicate_columns",),
    "detect_skewness": ("skewness",),
    "summarize_numeric": ("near_zero_variance", "skewness"),
    "compute_kurtosis": ("kurtosis",),
    "compute_entropy": ("entropy",),
    "suggest_dtype_optimizations": ("dtype_optimizations",),
    "sample_size_adequacy": ("sample_size",),
    "detect_class_imbalance": ("class_imbalance",),
    "schema_validation": ("schema",),
    "detect_outliers": ("outliers", "multivariate_outliers"),
    "normality_tests": ("normality",),
    "normality_qq_plots": ("normality",),
    "detect_bimodal_distribution": ("bimodality",),
    "detect_out_of_bounds": ("out_of_bounds",),
    "compute_pairwise_associations": ("associations", "contingency"),
    "contingency_tables": ("contingency",),
    "detect_data_leakage": ("associations",),
}

# Tasks that restate the profile and raise no finding of their own. No check replaces
# them: the profile holds what they computed, so they are replaced whenever the core
# pack is selected.
RETIRED_TASKS: frozenset[str] = frozenset(
    {
        "summarize_unique",
        "summarize_value_counts",
        "summarize_modes",
        "summarize_text_fields",
        "summarize_dataset_shape",
        "categorical_length_stats",
        "missingness_heatmap",
        "infer_types",
        "sample_head",
        "sample_tail",
    }
)

PROFILE = "the profile"


def superseded_tasks(
    packs: Collection[str],
    core_checks: Mapping[str, Collection[str]],
    include: Collection[str] = (),
    mapping: Mapping[str, tuple[str, ...]] = SUPERSEDED_BY,
    retired: Collection[str] = RETIRED_TASKS,
) -> dict[str, str]:
    """The tasks replaced in a run that selects ``packs``, with what replaces each.

    ``core_checks`` maps a pack name to the ids of its checks (``Registry.packs``). A
    task is replaced when every check that supersedes it is in a selected pack, unless
    the caller named the task in ``include``: asking for a task by id gets the task as
    it was, findings and all. A retired task is replaced by the profile whenever the
    core pack is selected.
    """
    selected = {check for pack in packs for check in core_checks.get(pack, ())}
    replaced = {
        task: " and ".join(checks)
        for task, checks in mapping.items()
        if set(checks) <= selected and task not in include
    }
    if CORE_PACK in packs:
        replaced.update({task: PROFILE for task in retired if task not in include})
    return replaced


def make_legacy_plan(
    registry: Registry,
    selection: Selection,
    superseded: Mapping[str, str],
    *,
    params: Mapping[str, Mapping[str, Any]] | None = None,
    profile: DatasetProfile | None = None,
) -> Plan:
    """Plan a run, keeping the inputs of the legacy tasks that remain.

    A task in no pack is planned only when another check requires it. Some tasks also
    read results without requiring them (``UNDECLARED_READS``), which orders the run
    but pulls nothing in, so a superseded task such as ``detect_out_of_bounds`` would
    be missing from ``data_quality_scorer``'s inputs. After planning, every
    superseded task a planned task reads is added by id and the run is planned again,
    until none is missing (one the caller excluded stays out).
    """
    added: set[str] = set()
    while True:
        chosen = replace(selection, include=(*selection.include, *sorted(added)))
        plan = make_plan(registry, chosen, params=params, profile=profile)
        missing = {
            read
            for entry in plan
            for read in UNDECLARED_READS.get(entry.id, ())
            if read in superseded and read not in plan and read not in selection.exclude
        }
        if not missing:
            return plan
        added |= missing
