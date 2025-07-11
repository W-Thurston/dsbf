import json
import os
from typing import Any, Dict, List, Optional, Union

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_loader import load_all_tasks
from dsbf.eda.task_registry import TASK_REGISTRY, TaskSpec
from dsbf.eda.task_registry import is_diagnostic_name as is_diagnostic_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.logging_utils import setup_logger

logger = setup_logger("dsbf.task_validator", "warn")

load_all_tasks()

__all__ = [
    "instantiate_task",
    "is_diagnostic_task",
    "validate_task_result",
    "filter_tasks",
    "write_task_metadata",
]


def instantiate_task(
    task_name: str,
    task_specific_cfg: Optional[Dict[str, Any]] = None,
) -> BaseTask:

    # Construct task instance using registry spec
    spec = TASK_REGISTRY[task_name]
    try:
        return spec.cls(name=task_name, config=task_specific_cfg)
    except KeyError:
        logger.warning(f"[instantiate_task] Task '{task_name}' not found in registry.")
        raise


def validate_task_result(result: TaskResult, raise_on_error: bool = False) -> bool:
    """
    Validate the structure and content of a TaskResult object.

    Args:
        result (TaskResult): The result to validate.
        raise_on_error (bool): If True, raises ValueError on failure. Else logs warning.

    Returns:
        bool: True if valid, False if invalid and not raised.
    """
    valid = True

    def fail(msg: str):
        nonlocal valid
        valid = False
        if raise_on_error:
            raise ValueError(msg)
        else:
            logger.warning(f"[TaskResultValidator] {msg}")

    if not result.name:
        fail("Missing 'name' field")

    if result.status not in {"success", "failed", "skipped"}:
        fail(f"Invalid 'status': {result.status}")

    if result.recommendations is not None:
        if not isinstance(result.recommendations, list):
            fail("'recommendations' must be a list of strings")
        elif not all(isinstance(r, str) for r in result.recommendations):
            fail("All items in 'recommendations' must be strings")

    if hasattr(result, "ml_impact_score"):
        score = getattr(result, "ml_impact_score")
        if score is not None and not (0 <= score <= 1):
            fail(f"'ml_impact_score' must be in [0, 1], got: {score}")

    return valid


def filter_tasks(criteria: Dict[str, Union[str, List[str]]]) -> List[str]:
    """
    Filters registered tasks based on AND-combined metadata criteria.

    Args:
        criteria (dict): Keys should match TaskSpec fields
            (e.g. 'domain', 'tags', 'runtime_estimate').

    Returns:
        list: List of task names matching the filters.
    """

    def matches(spec: TaskSpec) -> bool:
        for key, value in criteria.items():
            task_val = getattr(spec, key, None)
            if task_val is None:
                return False
            if isinstance(value, list):
                if isinstance(task_val, list):
                    if not any(v in task_val for v in value):
                        return False
                else:
                    if task_val not in value:
                        return False
            else:
                if isinstance(task_val, list):
                    if value not in task_val:
                        return False
                else:
                    if task_val != value:
                        return False
        return True

    return [name for name, spec in TASK_REGISTRY.items() if matches(spec)]


def write_task_metadata(output_path: str = "dsbf/docs/task_metadata.json") -> None:
    """
    Export all static task metadata to a JSON file (for documentation or UI use).

    Args:
        output_path (str): Path to save the metadata file.
    """
    metadata_dict = {}
    for name, spec in TASK_REGISTRY.items():
        metadata_dict[name] = {
            "domain": spec.domain,
            "tags": spec.tags,
            "stage": spec.stage,
            "runtime_estimate": spec.runtime_estimate,
            "profiling_depth": spec.profiling_depth,
            "summary": (
                spec.description.strip().splitlines()[0] if spec.description else ""
            ),
            "ml_impact_score": getattr(spec.cls, "ml_impact_score", None),  # Optional
            "experimental": spec.experimental,
        }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(metadata_dict, f, indent=2)

    logger.info(f"[Metadata Export] Task metadata written to: {output_path}")
