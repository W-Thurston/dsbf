# dsbf/utils/reco_engine.py
"""The legacy tasks' way to a recommendation tip; the library lives in the engine."""

from typing import Any

from dsbf.engine.tips import legacy_tip


def get_recommendation_tip(task_name: str, context_vars: dict[str, Any]) -> str | None:
    """
    Given a task name and context vars,
    return the best matching advanced recommendation.

    Returns None when no tip applies or the library is unusable.
    """
    return legacy_tip(task_name, context_vars)
