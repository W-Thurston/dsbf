# dsbf/eda/task_registry.py

import importlib
import importlib.util
import os
import re
import sys
import traceback
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dsbf.core.base_task import BaseTask
from dsbf.utils.logging_utils import setup_logger

logger = setup_logger("dsbf.task_registry", "info")

PLUGIN_LOG_FN = None


def set_plugin_logger(log_fn):
    global PLUGIN_LOG_FN
    PLUGIN_LOG_FN = log_fn


PLUGIN_WARNINGS: list[dict] = []


def get_plugin_warnings():
    return PLUGIN_WARNINGS


# -- Metadata container --
@dataclass
class TaskSpec:
    """
    Metadata container for registered DSBF tasks.

    This structure powers dynamic task discovery, filtering,
    documentation, and plugin system integration.
    """

    name: str  # Unique snake_case name used for registration and execution
    cls: type[BaseTask]  # Reference to the task class itself
    profiling_depth: str = "full"  # One of: "basic", "standard", "full"

    display_name: str | None = None  # Human-friendly name for UIs/docs
    description: str | None = None  # Full docstring or user-supplied override

    depends_on: list[str] | None = None  # list of task names this task depends on
    tags: list[str] | None = (
        None  # Descriptive tags for filtering (e.g., ["leakage", "text"])
    )
    stage: str | None = None  # Pipeline stage (e.g., "early", "cleaning", "modeling")
    domain: str | None = None  # Domain this task is relevant to (e.g., "healthcare")

    runtime_estimate: str | None = None  # Estimated cost ("fast", "medium", "slow")
    inputs: list[str] | None = None  # Expected inputs (e.g., ["dataframe"])
    outputs: list[str] | None = None  # Expected outputs (e.g., ["TaskResult"])

    experimental: bool = False  # Marks the task as unstable or in testing
    phase: str | None = None  # "eda", "ml_readiness", or "diagnostic"
    expected_semantic_types: list[str] | None = (
        None  # list of expected semantic types (e.g., ["continuous"])
    )


# -- Global registry --
TASK_REGISTRY: dict[str, TaskSpec] = {}


# -- Decorator --
def register_task(
    name: str | None = None,
    *,
    profiling_depth: str = "full",
    display_name: str | None = None,
    description: str | None = None,
    depends_on: list[str] | None = None,
    tags: list[str] | None = None,
    stage: str | None = None,
    domain: str | None = None,
    runtime_estimate: str | None = None,
    inputs: list[str] | None = None,
    outputs: list[str] | None = None,
    experimental: bool = False,
    phase: str | None = None,
    expected_semantic_types: list[str] | None = None,
) -> Callable[[type[BaseTask]], type[BaseTask]]:
    """
    Decorator to register a BaseTask subclass in the global TASK_REGISTRY.

    Args:
        name (str | None): Unique task name (defaults to snake_case of class name).
        profiling_depth (str): One of "basic", "standard", or "full".
        display_name (str | None): Friendly name for UIs or reports.
        description (str | None): Full description or docstring override.
        depends_on (list[str] | None]): list of prerequisite task names.
        tags (list[str] | None]): Tags for filtering or grouping tasks.
        stage (str | None): Logical stage in the pipeline
            (e.g., "early", "cleaning").
        domain (str | None): Domain this task is intended for (e.g., "finance").
        runtime_estimate (str | None): Expected runtime cost (e.g., "fast").
        inputs (list[str] | None]): Required input types or names.
        outputs (list[str] | None]): Outputs produced by this task.
        experimental (bool): Flag to mark unstable or test-only tasks.
        expected_semantic_types (list[str] | None]): Expected semantic types for
            column selection.

    Returns:
        Callable: Class decorator that registers the task into TASK_REGISTRY.
    """
    VALID_STAGES = ("raw", "cleaned", "modeling", "report", "any")

    def decorator(cls: type[BaseTask]) -> type[BaseTask]:
        task_name: str = name or _to_snake_case(cls.__name__)
        if stage and stage not in VALID_STAGES:
            raise ValueError(
                f"Invalid stage '{stage}' for task '{task_name}'. "
                f"Allowed stages are: {VALID_STAGES}"
            )

        VALID_PHASES = ("eda", "ml_readiness", "diagnostic")
        if phase and phase not in VALID_PHASES:
            raise ValueError(
                f"Invalid phase '{phase}' for task '{task_name}'. "
                f"Allowed phases are: {VALID_PHASES}"
            )

        if task_name in TASK_REGISTRY:
            existing_cls = TASK_REGISTRY[task_name].cls
            raise ValueError(
                f"Task '{task_name}' already registered by class "
                f"'{existing_cls.__name__}'."
            )

        spec = TaskSpec(
            name=task_name,
            cls=cls,
            display_name=display_name or cls.__name__,
            description=description or cls.__doc__,
            depends_on=depends_on,
            tags=tags,
            stage=stage,
            domain=domain,
            runtime_estimate=runtime_estimate,
            profiling_depth=profiling_depth,
            inputs=inputs,
            outputs=outputs,
            experimental=experimental,
            phase=phase,
            expected_semantic_types=expected_semantic_types,
        )

        TASK_REGISTRY[task_name] = spec
        return cls

    return decorator


def _to_snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def get_all_task_specs() -> list[TaskSpec]:
    return list(TASK_REGISTRY.values())


def describe_registered_tasks() -> None:
    print("Registered DSBF Tasks:\n")
    for name, spec in TASK_REGISTRY.items():
        tags = ", ".join(spec.tags or [])
        stage = spec.stage or "unspecified"
        print(
            f"- {name:30} - {spec.description or 'No description.'} [{stage}] ({tags})"
        )


def list_tasks(
    by: Literal["domain", "stage", "phase", "tags", "profiling_depth"] | None = None,
) -> None:
    """
    Print all registered tasks, optionally grouped by a metadata field.

    Args:
        by (str | None): Field to group by (e.g., "domain", "stage", "tags").
    """
    if not by:
        print("Registered DSBF Tasks:\n")
        for name, spec in TASK_REGISTRY.items():
            print(f"- {name:30} - {spec.description or 'No description'}")
        return

    grouped = defaultdict(list)
    for name, spec in TASK_REGISTRY.items():
        key = getattr(spec, by, None)
        if key is None:
            grouped["unspecified"].append((name, spec))
        elif isinstance(key, list):  # e.g. tags
            for item in key:
                grouped[item].append((name, spec))
        else:
            grouped[key].append((name, spec))

    print(f"Registered tasks grouped by '{by}':\n")
    for group, entries in sorted(grouped.items()):
        print(f"[{group}]")
        for name, spec in entries:
            print(f"  - {name:25} - {spec.description or 'No description'}")
        print()


def describe_task(name: str) -> None:
    """
    Print detailed metadata for a single registered task.

    Args:
        name (str): Task name (snake_case).
    """
    spec = TASK_REGISTRY.get(name)
    if not spec:
        print(f"[describe_task] Task '{name}' not found.")
        return

    print(f"Task:               {spec.name}")
    print(f"  Class:            {spec.cls.__name__}")
    print(f"  Display Name:     {spec.display_name}")
    print(f"  Description:      {spec.description}")
    print(f"  Profiling Depth:  {spec.profiling_depth}")
    print(f"  Domain:           {spec.domain}")
    print(f"  Stage:            {spec.stage}")
    print(f"  Tags:             {', '.join(spec.tags or [])}")
    print(f"  Depends On:       {', '.join(spec.depends_on or [])}")
    print(f"  Runtime Estimate: {spec.runtime_estimate}")
    print(f"  Inputs:           {', '.join(spec.inputs or [])}")
    print(f"  Outputs:          {', '.join(spec.outputs or [])}")
    print(f"  Experimental:     {spec.experimental}")
    print(f"  Phase:            {spec.phase or 'untagged'}")
    print(f"  Expected types:   {', '.join(spec.expected_semantic_types or [])}")


def load_task_group(group: str) -> None:
    """
    Load a group of tasks either from a built-in module (e.g., 'core')
    or from a local directory path (e.g., './custom_plugins/healthcare/').

    All tasks decorated with @register_task will be added to TASK_REGISTRY on import.

    Args:
        group (str): Module name or directory path.
    """
    group_path = Path(group)

    try:
        if group_path.is_dir():
            if not group_path.exists():
                logger.warning(f"[Plugin Load] Path does not exist: {group_path}")
                return

            # Load all Python files in the directory
            for file in group_path.glob("*.py"):
                if not file.name.startswith("_"):
                    _import_local_python_file(file.resolve())
        else:
            # Treat as Python module (e.g., 'core')
            module_path = f"dsbf.custom_plugins.{group}"
            importlib.import_module(module_path)

    except Exception as e:
        logger.warning(f"[Plugin Load] Failed to load group '{group}': {e}")
        traceback.print_exc()

    # Automatically export full metadata after loading tasks
    if os.environ.get("DSBF_AUTO_EXPORT_METADATA", "1") == "1":
        from dsbf.utils.task_utils import write_task_metadata

        write_task_metadata("dsbf/static_metadata/task_metadata.json")


def _import_local_python_file(path: Path) -> None:
    """
    Dynamically import a local plugin file and warn if no tasks were registered.

    Args:
        path (Path): Full path to a .py plugin file.
    """
    task_names_before = set(TASK_REGISTRY.keys())

    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, str(path))

    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        task_names_after = set(TASK_REGISTRY.keys())
        new_tasks = task_names_after - task_names_before

        if not new_tasks:
            warning_msg = f"Plugin file '{path.name}' did not register any tasks."

            PLUGIN_WARNINGS.append(
                {
                    "file": str(path),
                    "message": warning_msg,
                }
            )

            if PLUGIN_LOG_FN:
                PLUGIN_LOG_FN(f"[PLUGIN WARNING] {warning_msg}", level="warn")
            else:
                logger.warning(f"[WARN] {warning_msg}")


DIAGNOSTIC_TASK_PREFIXES = ["identifybottleneck", "logresourceusage"]


def is_diagnostic_name(name: str) -> bool:
    """
    Determines if a task name matches a known diagnostic task pattern.

    Args:
        name (str): Task name to check (e.g., 'identify_bottleneck_tasks').

    Returns:
        bool: True if diagnostic, False otherwise.
    """
    normalized = name.lower().replace("_", "")
    return any(normalized.startswith(prefix) for prefix in DIAGNOSTIC_TASK_PREFIXES)
