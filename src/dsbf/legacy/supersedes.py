# dsbf/legacy/supersedes.py
"""Which legacy tasks a core check replaces, and how a plan keeps their inputs."""

from collections.abc import Collection, Mapping
from dataclasses import replace
from typing import Any

from dsbf.engine import Plan, Registry, Selection, make_plan
from dsbf.legacy.adapter import UNDECLARED_READS
from dsbf.model import DatasetProfile

# Legacy task to the core check that replaces it. A unit that migrates a task adds it
# here, and the mapping goes when the last task does.
SUPERSEDED_BY: Mapping[str, str] = {
    "detect_zeros": "zeros",
}


def superseded_tasks(
    packs: Collection[str],
    core_checks: Mapping[str, Collection[str]],
    include: Collection[str] = (),
    mapping: Mapping[str, str] = SUPERSEDED_BY,
) -> dict[str, str]:
    """The tasks replaced in a run that selects ``packs``, with the check for each.

    ``core_checks`` maps a pack name to the ids of its checks (``Registry.packs``). A
    task is replaced when the check that supersedes it is in a selected pack, unless
    the caller named the task in ``include``: asking for a task by id gets the task as
    it was, findings and all.
    """
    selected = {check for pack in packs for check in core_checks.get(pack, ())}
    return {
        task: check
        for task, check in mapping.items()
        if check in selected and task not in include
    }


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
