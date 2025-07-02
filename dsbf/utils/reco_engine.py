# dsbf/utils/reco_engine.py

from pathlib import Path
from typing import Optional

import yaml

RECO_PATH = Path("dsbf/static_metadata/recommendation_library.yaml")
RECO_CACHE = None  # cache after load


def load_recommendation_library():
    global RECO_CACHE
    if RECO_CACHE is None:
        with open(RECO_PATH) as f:
            RECO_CACHE = yaml.safe_load(f)
    return RECO_CACHE


def get_recommendation_tip(task_name: str, context_vars: dict) -> Optional[str]:
    """
    Given a task name and context vars,
    return the best matching advanced recommendation.
    """
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
    return None
