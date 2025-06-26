# dsbf/eda/rednerers/json_renderer.py

import json


def render(results: dict, metadata: dict, output_path: str):
    wrapped = {"metadata": metadata, "results": results}
    with open(output_path, "w") as f:
        json.dump(wrapped, f, indent=2, default=str)
