# tests/helpers/context_utils.py


from typing import TYPE_CHECKING, Any

from dsbf.config import load_default_config
from dsbf.core.context import AnalysisContext
from dsbf.eda.task_registry import TASK_REGISTRY
from dsbf.eda.task_result import TaskResult
from dsbf.utils.task_utils import instantiate_task

if TYPE_CHECKING:
    from dsbf.core.base_task import BaseTask
    from dsbf.eda.task_registry import TaskSpec


def make_ctx_and_task(
    task_cls: type,
    current_df,
    reference_df=None,
    task_overrides: dict | None = None,
    global_overrides: dict | None = None,
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
    default_config: dict[str, Any] = load_default_config()
    # Use the snake_case registry name as the config key so that
    # make_ctx_and_task and run_task_with_dependencies agree on which
    # key to read from ctx.config["tasks"].
    # _to_snake_case("DetectZeros") → "detect_zeros" matches registry_entry.name.
    from dsbf.eda.task_registry import _to_snake_case

    task_name: str = _to_snake_case(task_cls.__name__)

    # Get and update task-specific config
    task_config = default_config.get("tasks", {}).get(task_name, {}).copy()
    if task_overrides:
        task_config.update(task_overrides)

    # Apply global overrides
    full_config: dict[str, Any] = default_config.copy()
    if global_overrides:
        full_config.update(global_overrides)

    # Inject updated task config under the snake_case registry key
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


def run_task_with_dependencies(ctx: AnalysisContext, task_cls: type) -> TaskResult:
    """
    Recursively run all declared dependencies (via TASK_REGISTRY) for the given task,
    then run the task itself. Returns the final TaskResult.

    Args:
        ctx (AnalysisContext): The shared context for task execution.
        task_cls (Type): The main task class to run after dependencies.

    Returns:
        TaskResult: Output of the final task.
    """

    registry_entry: TaskSpec | None = next(
        (spec for name, spec in TASK_REGISTRY.items() if spec.cls == task_cls),
        None,
    )
    if not registry_entry:
        raise ValueError(f"Task {task_cls.__name__} not found in registry.")

    task_name: str = registry_entry.name
    if not registry_entry:
        raise ValueError(f"Task '{task_name}' not registered.")

    visited: set = set()

    def _run_recursive(name: str, is_target: bool = False):
        if name in visited:
            return
        visited.add(name)
        deps: list[str] = TASK_REGISTRY[name].depends_on or []
        for dep_name in deps:
            _run_recursive(dep_name)
        # For the target task, pass any task-level config overrides that
        # were set via make_ctx_and_task(task_overrides=...) so that
        # parameters like flag_threshold, custom_bounds, etc. reach the
        # task even when run_task_with_dependencies re-instantiates it.
        task_cfg = ctx.config.get("tasks", {}).get(name, {}) if is_target else None
        dep_task: BaseTask = instantiate_task(name, task_specific_cfg=task_cfg or None)
        ctx.run_task(dep_task)  # uses full validation

    _run_recursive(task_name, is_target=True)
    result: TaskResult | None = ctx.get_result(task_name)
    if result is None:
        raise RuntimeError(f"Task '{task_name}' did not produce a TaskResult.")
    return result
