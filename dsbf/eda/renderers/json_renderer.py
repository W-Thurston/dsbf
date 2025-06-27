# dsbf/eda/renderers/json_renderer.py
import json

import numpy as np

from dsbf.eda.task_result import TaskResult  # optional, for type safety


def render(results: dict, metadata: dict, output_path: str):
    results_serialized = {
        k: v.to_dict() if isinstance(v, TaskResult) else v for k, v in results.items()
    }

    wrapped = {"metadata": metadata, "results": results_serialized}

    def convert(o):
        if isinstance(o, (np.integer, np.floating)):
            return o.item()
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return str(o)  # fallback for unknowns

    with open(output_path, "w") as f:
        json.dump(wrapped, f, indent=2, default=convert)
