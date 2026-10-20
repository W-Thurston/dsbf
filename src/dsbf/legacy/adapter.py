# dsbf/legacy/adapter.py
"""Wrap a legacy ``BaseTask`` as a ``Check`` the new engine can plan and run."""

import hashlib
import importlib
import inspect
import re
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

from pydantic import create_model

import dsbf
from dsbf.core.base_task import BaseTask
from dsbf.eda.task_loader import load_all_tasks
from dsbf.eda.task_registry import TASK_REGISTRY, TaskSpec, load_task_group
from dsbf.engine import (
    Check,
    CheckContext,
    CheckOutput,
    CheckSpec,
    Requires,
    SemanticTypes,
)
from dsbf.legacy.context import LegacyContext
from dsbf.legacy.result import LegacyTaskOutput, from_output, to_output
from dsbf.legacy.session import LegacySession
from dsbf.model import CheckParams, CheckStatus, Cost, Dimension, Severity, Stage

LEGACY_PACK = "legacy"
# Where a task that a core check replaces sits: no default selection includes it, and a
# check that requires it still pulls it in.
SUPERSEDED_PACK = "legacy_superseded"
LEGACY_VERSION = "0.1.0"

# The profile types a legacy semantic type covers.
_PROFILE_TYPES: dict[str, tuple[str, ...]] = {
    "continuous": ("numeric",),
    "categorical": ("categorical", "boolean"),
    "datetime": ("datetime",),
    "text": ("text",),
    "id": ("identifier",),
    "unknown": ("unknown",),
}
_COST = {"fast": Cost.CHEAP, "moderate": Cost.MODERATE, "slow": Cost.HEAVY}

# The first tag of a task that names a dimension decides it; a task whose tags name
# none is about the distribution of its values.
_DIMENSION_OF_TAG: dict[str, Dimension] = {
    "missing": Dimension.COMPLETENESS,
    "missingness": Dimension.COMPLETENESS,
    "validation": Dimension.VALIDITY,
    "schema": Dimension.VALIDITY,
    "format": Dimension.VALIDITY,
    "redundancy": Dimension.REDUNDANCY,
    "duplicates": Dimension.REDUNDANCY,
    "multicollinearity": Dimension.REDUNDANCY,
    "leakage": Dimension.LEAKAGE,
    "relationships": Dimension.RELATIONSHIP,
    "association": Dimension.RELATIONSHIP,
    "correlation": Dimension.RELATIONSHIP,
    "drift": Dimension.RELATIONSHIP,
    "time_series": Dimension.TEMPORAL,
    "datetime": Dimension.TEMPORAL,
    "ml_readiness": Dimension.USABILITY,
    "quality": Dimension.USABILITY,
    "scoring": Dimension.USABILITY,
}

# The two tasks that report how long the others took. Each result's ``Timing`` says
# that now, and neither can run in an adapted run: they read a duration table only the
# legacy graph engine fills in (the first ends ``error`` without it, the second
# reports on empty statistics).
EXCLUDED_TASKS = frozenset({"identify_bottleneck_tasks", "log_resource_usage"})

# Results a task reads without naming the task in ``depends_on``. The legacy engine
# showed them only if the task happened to run earlier; here they order the run
# without ever blocking it. ``tests/test_legacy/test_undeclared_reads.py`` fails when a
# task reads one that is missing.
UNDECLARED_READS: dict[str, tuple[str, ...]] = {
    "data_quality_scorer": (
        "detect_collinear_features",
        "detect_out_of_bounds",
        "summarize_numeric",
    ),
    "generate_dataset_summary_plots": ("compute_pairwise_associations",),
    "imputation_strategy_suggestions": ("detect_skewness",),
}


class LegacyParams(CheckParams):
    """The configuration an adapted task runs with.

    The values are the task's own config and the shared config, so changing either
    changes the cache key; ``source`` is a digest of the code that runs, so editing a
    task does too. ``output_dir`` and ``reference`` are set only for a task that
    writes files or reads the reference frame, so a cached result never points at a
    directory it was not written to nor compares the wrong frame. ``superseded_by``
    names the core check that reports this task's findings, so the task reports none
    of its own; it is part of the key because the result differs.
    """

    task: dict[str, Any] = {}
    shared: dict[str, Any] = {}
    source: str = ""
    output_dir: str | None = None
    reference: str | None = None
    superseded_by: str | None = None


def adapt_task(
    spec: TaskSpec, session: LegacySession, superseded_by: str | None = None
) -> Check:
    """The check that runs the task ``spec`` registered, on ``session``'s data.

    The check applies to the profile types the task's expected semantic types map
    to, so a dataset with none of them plans it ``not_applicable`` where the legacy
    engine ran it and got an empty result. It runs on a thread: an adapted check is
    built at run time, which a worker process cannot import by name.

    A task ``superseded_by`` a core check is in the ``legacy_superseded`` pack instead
    of ``legacy``, so the default selection leaves it out, but a check that requires it
    still pulls it in. It reports its output and no
    findings, so each finding is reported once.
    """
    source = _module_source(spec.cls)
    params = _params_model(spec, session, source, superseded_by)
    declared = CheckSpec(
        id=spec.name,
        version=LEGACY_VERSION,
        kind="analysis",
        applies_to=_applies_to(spec.expected_semantic_types),
        dimensions=(_dimension(spec.tags or ()),),
        params=params,
        output=LegacyTaskOutput,
        requires=Requires(checks=spec.depends_on or ()),
        after=_soft_upstream(spec),
        cost=_COST.get(spec.runtime_estimate or "", Cost.MODERATE),
        stage=Stage(spec.stage or "any"),
        runs_in="thread",
        packs=(SUPERSEDED_PACK if superseded_by else LEGACY_PACK,),
        tags=tuple(spec.tags or ()),
        summary=_summary(spec),
    )

    def run(ctx: CheckContext, params: Any) -> CheckOutput:
        return _run_task(spec, session, ctx, params)

    return Check(spec=declared, function=run)


def legacy_checks(
    session: LegacySession,
    plugin_groups: tuple[str, ...] = (),
    superseded: Mapping[str, str] | None = None,
) -> tuple[Check, ...]:
    """An adapted check for every registered task but ``EXCLUDED_TASKS``, by name.

    The built-in tasks are loaded first, then each plugin group (a directory path
    or a module name, as the legacy ``task_groups`` config key names them). A task
    that writes files is left out when the session has no output directory, since
    it could only end ``error``; ``adapt_task`` still adapts one by name.
    ``superseded`` maps a task to the core check that replaces it in this run
    (``dsbf.legacy.superseded_tasks``).
    """
    _scan_native_libraries()
    load_all_tasks()
    for group in plugin_groups:
        load_task_group(group)
    return tuple(
        adapt_task(TASK_REGISTRY[name], session, (superseded or {}).get(name))
        for name in sorted(TASK_REGISTRY)
        if name not in EXCLUDED_TASKS
        and (
            session.output_dir is not None or not _writes_files(TASK_REGISTRY[name].cls)
        )
    )


def _scan_native_libraries() -> None:
    """Have scikit-learn scan the process's native libraries now, on this thread.

    The scan (threadpoolctl) walks every loaded shared library while holding the
    dynamic loader's lock, and needs the interpreter's lock to do it. A check that
    imports a compiled module at that moment, such as the first PNG save of a plot,
    holds the interpreter's lock and waits for the loader's, so both wait for ever.
    scikit-learn keeps the result of its first scan, so making it before any check
    starts leaves nothing to scan while they run in parallel.
    """
    try:
        parallel = importlib.import_module("sklearn.utils.parallel")
    except ImportError:  # no scikit-learn, nothing to scan
        return
    scan = getattr(parallel, "_get_threadpool_controller", None)
    if scan is not None:
        scan()


def _run_task(
    spec: TaskSpec, session: LegacySession, ctx: CheckContext, params: LegacyParams
) -> CheckOutput:
    if ctx.profile.n_rows != session.profile.n_rows:
        raise ValueError("the session was built for different data than this run")
    upstream = {}
    for name in ctx.spec.upstream:
        result = ctx.result(name)
        if result is not None and result.status is CheckStatus.OK and result.output:
            upstream[name] = from_output(result.output)
    config = {**params.shared, "tasks": {spec.name: params.task}}
    legacy = LegacyContext(session, config, upstream)
    task: BaseTask = spec.cls(name=spec.name, config=params.task)
    if _writes_files(spec.cls) and session.output_dir is None:
        return ctx.stop(
            CheckStatus.ERROR,
            "output_dir_missing",
            f"{spec.name} writes files and the run has no output directory.",
            severity=Severity.ERROR,
            hint="Give the LegacySession an output_dir.",
        )
    ctx.checkpoint()
    return to_output(ctx, legacy.run_task(task), superseded_by=params.superseded_by)


def _applies_to(expected: list[str] | None) -> SemanticTypes:
    if not expected or "any" in expected:
        return SemanticTypes()
    return SemanticTypes(*(t for kind in expected for t in _PROFILE_TYPES[kind]))


def _dimension(tags: list[str] | tuple[str, ...]) -> Dimension:
    return next(
        (_DIMENSION_OF_TAG[t] for t in tags if t in _DIMENSION_OF_TAG),
        Dimension.DISTRIBUTION,
    )


def _soft_upstream(spec: TaskSpec) -> tuple[str, ...]:
    declared = {spec.name, *(spec.depends_on or ())}
    return tuple(n for n in UNDECLARED_READS.get(spec.name, ()) if n not in declared)


def _summary(spec: TaskSpec) -> str:
    lines = (spec.description or "").strip().splitlines()
    return re.sub(r"\s+", " ", lines[0]).strip() if lines else ""


def _module_text(cls: type[BaseTask]) -> str:
    path = inspect.getsourcefile(cls)
    return Path(path).read_text(encoding="utf-8") if path else ""


def _writes_files(cls: type[BaseTask]) -> bool:
    return "get_output_path" in _module_text(cls)


def _reads_reference(cls: type[BaseTask]) -> bool:
    return "reference_data" in _module_text(cls)


@cache
def _support_digest() -> str:
    """A digest of the code every legacy task shares: base task, utils, findings."""
    root = Path(dsbf.__file__).parent
    files = [
        *sorted((root / "core").glob("*.py")),
        *sorted((root / "utils").glob("*.py")),
        *sorted((root / "eda").glob("*.py")),
        root / "eda" / "tasks" / "_time_series_utils.py",
        root / "legacy" / "findings.py",
    ]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _module_source(cls: type[BaseTask]) -> str:
    """A digest of the task's module and the shared code, to key the cache on."""
    text = _module_text(cls).encode()
    return hashlib.sha256(text + _support_digest().encode()).hexdigest()[:16]


def _params_model(
    spec: TaskSpec, session: LegacySession, source: str, superseded_by: str | None
) -> type[LegacyParams]:
    config = session.config
    return create_model(  # type: ignore[call-overload,no-any-return]
        f"{spec.cls.__name__}Params",
        __base__=LegacyParams,
        task=(dict[str, Any], dict(config.get("tasks", {}).get(spec.name) or {})),
        shared=(
            dict[str, Any],
            {key: value for key, value in config.items() if key != "tasks"},
        ),
        source=(str, source),
        output_dir=(
            str | None,
            session.output_dir if _writes_files(spec.cls) else None,
        ),
        reference=(
            str | None,
            session.reference_digest() if _reads_reference(spec.cls) else None,
        ),
        superseded_by=(str | None, superseded_by),
    )
