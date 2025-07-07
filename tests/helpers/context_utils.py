# tests/helpers/context_utils.py

from typing import Optional, Type

from dsbf.config import load_default_config
from dsbf.core.context import AnalysisContext


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
