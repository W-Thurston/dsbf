# dsbf/eda/task_registry.py

import re
from typing import Dict, Optional, Type

from dsbf.core.base_task import BaseTask

# Global registry
TASK_REGISTRY: Dict[str, Type[BaseTask]] = {}


def to_snake_case(name: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def register_task(name: Optional[str] = None):
    """
    Decorator to register a task class in the global registry.
    If no name is provided, uses the class name.
    """

    def decorator(cls):
        task_name = name or to_snake_case(cls.__name__)
        if task_name in TASK_REGISTRY:
            raise ValueError(f"Task name '{task_name}' already registered.")
        TASK_REGISTRY[task_name] = cls
        return cls

    return decorator


def get_task_by_name(name: str) -> BaseTask:
    if name not in TASK_REGISTRY:
        raise KeyError(f"Task '{name}' not found in registry.")
    return TASK_REGISTRY[name]()
