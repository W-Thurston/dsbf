# dsbf/utils/reco_engine.py

from importlib.resources import files
from importlib.resources.abc import Traversable
from typing import Any

import yaml

from dsbf.utils.logging_utils import setup_logger

logger = setup_logger("dsbf.reco_engine", "warn")

RECO_CACHE: dict[str, Any] | None = None  # cache after load


def _library_resource() -> Traversable:
    """Locate the packaged recommendation library, independent of the CWD."""
    return files("dsbf").joinpath("static_metadata", "recommendation_library.yaml")


def load_recommendation_library() -> dict[str, Any]:
    """
    Load the recommendation library once per process.

    A library that cannot be read or parsed is logged once and treated as empty,
    because tips are optional enrichment and must never fail an analysis.
    """
    global RECO_CACHE
    if RECO_CACHE is None:
        try:
            library = yaml.safe_load(_library_resource().read_text(encoding="utf-8"))
            if not isinstance(library, dict):
                raise ValueError("expected a mapping of tip ids to tips")
            RECO_CACHE = library
        except Exception as e:
            logger.warning(f"[reco_engine] Recommendation library unavailable: {e}")
            RECO_CACHE = {}
    return RECO_CACHE


def get_recommendation_tip(task_name: str, context_vars: dict) -> str | None:
    """
    Given a task name and context vars,
    return the best matching advanced recommendation.

    Returns None when no tip applies or the library is unusable.
    """
    try:
        tips = load_recommendation_library()
        for tip_id, tip in tips.items():
            if task_name not in tip.get("applicable_to", []):
                continue
            condition = tip.get("condition")
            try:
                if eval(condition, {}, context_vars):
                    return tip.get("message")
            except Exception:
                continue
    except Exception as e:
        logger.warning(f"[reco_engine] Tip lookup failed for '{task_name}': {e}")
    return None
