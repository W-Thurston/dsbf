# tests/helpers/context_utils.py

from typing import Optional, Type

from dsbf.config import load_default_config
from dsbf.core.context import AnalysisContext
from dsbf.eda.task_registry import TASK_REGISTRY
from dsbf.eda.task_result import TaskResult
from dsbf.utils.task_utils import instantiate_task


def make_ctx_and_task(
    task_cls: Type,
    current_df,
    reference_df=None,
    task_overrides: Optional[dict] = None,
    global_overrides: Optional[dict] = None,
):
    """
    Create an AnalysisContext and task instance using values from default_config.yaml,
    allowing overrides at both the task level and global level.

    Args:
        task_cls (Type): Task class to instantiate.
        current_df: Current input dataset (Polars or Pandas DataFrame).
        reference_df: Optional reference dataset.
        task_overrides (dict): Overrides for the task-specific config block.
        global_overrides (dict): Overrides for the top-level config
            (e.g., metadata, engine).

    Returns:
        ctx (AnalysisContext): Fully constructed analysis context.
        task (BaseTask): Initialized task instance.
    """
    default_config = load_default_config()
    task_name = task_cls.__name__

    # Get and update task-specific config
    task_config = default_config.get("tasks", {}).get(task_name, {}).copy()
    if task_overrides:
        task_config.update(task_overrides)

    # Apply global overrides
    full_config = default_config.copy()
    if global_overrides:
        full_config.update(global_overrides)

    # Inject updated task config
    full_config.setdefault("tasks", {})[task_name] = task_config

    if global_overrides is None:
        global_overrides = {}

    output_dir = global_overrides.pop("output_dir", None)
    ctx = AnalysisContext(
        data=current_df,
        config=full_config,
        output_dir=output_dir,
    )

    if reference_df is not None:
        ctx.reference_data = reference_df

    task = task_cls(name=task_name, config=task_config)
    return ctx, task


def run_task_with_dependencies(ctx: AnalysisContext, task_cls: Type) -> TaskResult:
    """
    Recursively run all declared dependencies (via TASK_REGISTRY) for the given task,
    then run the task itself. Returns the final TaskResult.

    Args:
        ctx (AnalysisContext): The shared context for task execution.
        task_cls (Type): The main task class to run after dependencies.

    Returns:
        TaskResult: Output of the final task.
    """

    registry_entry = next(
        (spec for name, spec in TASK_REGISTRY.items() if spec.cls == task_cls),
        None,
    )
    if not registry_entry:
        raise ValueError(f"Task {task_cls.__name__} not found in registry.")

    task_name = registry_entry.name
    if not registry_entry:
        raise ValueError(f"Task '{task_name}' not registered.")

    visited = set()

    def _run_recursive(name: str):
        if name in visited:
            return
        visited.add(name)
        deps = TASK_REGISTRY[name].depends_on or []
        for dep_name in deps:
            _run_recursive(dep_name)
        dep_task = instantiate_task(name)
        ctx.run_task(dep_task)  # uses full validation

    _run_recursive(task_name)
    result = ctx.get_result(task_name)
    if result is None:
        raise RuntimeError(f"Task '{task_name}' did not produce a TaskResult.")
    return result
