# dsbf/utils/constants.py

from typing import Dict

# Default ML severity thresholds (used if not overridden in config)
DEFAULT_SEVERITY_THRESHOLDS: Dict[str, float] = {
    "low": 0.0,
    "moderate": 0.6,
    "high": 0.85,
}

# Standard recommendation tags
IMPACT_TAGS = ["drop", "transform", "monitor", "check_leakage", "resample"]
