"""Child process for the golden runs: run one CSV through the core pack.

A separate interpreter, so a run shares nothing with the test session. It writes the
``Run`` as ``run.json`` in the output directory. It imports nothing from ``tests``.

Parameters for the core checks that need them come from the environment variable
``DSBF_GOLDEN_CORE_PARAMS``, a JSON mapping of check id to parameters; the time
series column, if there is one, is each check's ``time_index``.

Usage: python profile_one.py <name> <csv> <out_dir> [<time_series_column>]
"""

import json
import os
import sys
from pathlib import Path

from dsbf.checks import core_pack
from dsbf.data import load
from dsbf.engine import Registry, Selection, execute, make_plan
from dsbf.profiling import profile_dataset


def main() -> None:
    _name, csv_path, out_dir = sys.argv[1:4]
    time_series_column = sys.argv[4] if len(sys.argv) > 4 else None

    dataset = load(Path(csv_path))
    profile = profile_dataset(dataset)
    registry = Registry()
    registry.register_pack(core_pack())
    params = json.loads(os.environ.get("DSBF_GOLDEN_CORE_PARAMS", "{}"))
    if time_series_column is not None:
        # As `dsbf.profile(time_index=...)` does: to each check with that parameter.
        for registered in registry:
            if "time_index" in registered.spec.params.model_fields:
                params.setdefault(registered.spec.id, {})["time_index"] = (
                    time_series_column
                )
    plan = make_plan(
        registry,
        Selection(depth="deep", entry_points=False),
        params=params or None,
        profile=profile,
    )
    run = execute(plan, dataset, profile, dsbf_version="golden")
    Path(out_dir, "run.json").write_text(run.model_dump_json())


if __name__ == "__main__":
    main()
