# dsbf/eda/task_registry.py

import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Type

from dsbf.core.base_task import BaseTask

# -- Metadata container --


@dataclass
class TaskSpec:
    name: str
    cls: Type[BaseTask]
    display_name: Optional[str] = None
    description: Optional[str] = None
    depends_on: Optional[List[str]] = None
    tags: Optional[List[str]] = None
    stage: Optional[str] = None
    inputs: Optional[List[str]] = None
    outputs: Optional[List[str]] = None
    experimental: bool = False


# -- Global registry --

TASK_REGISTRY: Dict[str, TaskSpec] = {}

# -- Decorator --


def register_task(
    name: Optional[str] = None,
    *,
    display_name: Optional[str] = None,
    description: Optional[str] = None,
    depends_on: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    stage: Optional[str] = None,
    inputs: Optional[List[str]] = None,
    outputs: Optional[List[str]] = None,
    experimental: bool = False,
) -> Callable[[Type[BaseTask]], Type[BaseTask]]:
    """
    Decorator factory that registers a BaseTask subclass into the TASK_REGISTRY.
    """

    def decorator(cls: Type[BaseTask]) -> Type[BaseTask]:
        task_name: str = name or _to_snake_case(cls.__name__)
        if stage and stage not in {"raw", "cleaned", "modeling", "report"}:
            raise ValueError(
                f"Invalid stage '{stage}' for task '{task_name}'. "
                "Allowed stages are: raw, cleaned, modeling, report."
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
            inputs=inputs,
            outputs=outputs,
            experimental=experimental,
        )

        TASK_REGISTRY[task_name] = spec
        return cls

    return decorator


def _to_snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def get_all_task_specs() -> List[TaskSpec]:
    return list(TASK_REGISTRY.values())


def describe_registered_tasks() -> None:
    print("Registered DSBF Tasks:\n")
    for name, spec in TASK_REGISTRY.items():
        tags = ", ".join(spec.tags or [])
        stage = spec.stage or "unspecified"
        print(
            f"- {name:30} — {spec.description or 'No description.'} [{stage}] ({tags})"
        )
