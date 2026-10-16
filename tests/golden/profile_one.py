"""Child process for the golden runs: profile one CSV and write its run information.

This is a separate script, run in a fresh interpreter, because ``PYTHONHASHSEED`` can
only be set before Python starts. It imports nothing from ``tests``.

Usage: python profile_one.py <name> <csv> <out_dir> [<time_series_column>]
"""

import json
import os
import sys

from dsbf.config import load_default_config
from dsbf.eda.profile_engine import ProfileEngine


def main() -> None:
    name, csv_path, out_dir = sys.argv[1:4]
    time_series_column = sys.argv[4] if len(sys.argv) > 4 else None

    config = load_default_config()
    config["output_dir"] = out_dir
    config["metadata"].update(dataset_name=name, dataset_path=os.path.abspath(csv_path))
    config["time_series"].update(
        enabled=time_series_column is not None,
        datetime_index_column=time_series_column,
    )

    engine = ProfileEngine(config)
    engine.run()

    context = engine.context
    assert context is not None
    metadata = context.metadata
    info = {
        "shape": list(context.data.shape),
        "inferred_stage": engine.inferred_stage,
        "sampling": engine.run_metadata.get("sampling"),
        "task_count": len(engine.results),
        "task_outcomes": metadata.get("task_outcomes"),
        "semantic_types": metadata.get("semantic_types"),
        "inferred_dtypes": metadata.get("inferred_dtypes"),
    }
    with open(os.path.join(out_dir, "golden_run.json"), "w") as f:
        json.dump(info, f, default=str)


if __name__ == "__main__":
    main()
