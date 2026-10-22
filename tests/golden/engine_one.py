"""Child process for the engine golden runs: run one CSV through the new engine.

The counterpart of ``profile_one.py``. It loads the file with ``dsbf.data``, profiles
it, runs every adapted legacy task through ``execute`` and writes what each check
ended with (``engine_run.json``, ``run.json``), then runs the core pack alone and writes
its ``Run`` as ``core_run.json``. The legacy tasks are planned on their own, so
``engine_run.json`` stays the legacy tasks as they were. A separate interpreter,
because ``PYTHONHASHSEED`` can only be set before Python starts. It imports nothing
from ``tests``.

Parameters for the core checks that need them come from the environment variable
``DSBF_GOLDEN_CORE_PARAMS``, a JSON mapping of check id to parameters.

Usage: python engine_one.py <name> <csv> <out_dir> [<time_series_column>]
"""

import json
import os
import sys
from pathlib import Path

from dsbf.checks import core_pack
from dsbf.config import load_default_config
from dsbf.data import load
from dsbf.engine import Registry, Selection, execute, make_plan
from dsbf.legacy import LegacySession, legacy_checks
from dsbf.profiling import profile_dataset


def main() -> None:
    name, csv_path, out_dir = sys.argv[1:4]
    time_series_column = sys.argv[4] if len(sys.argv) > 4 else None

    config = load_default_config()
    config["output_dir"] = out_dir
    # The legacy engine reads a file with pandas whatever ``engine.backend`` says, so
    # that is the frame the snapshots were made from.
    config["engine"]["backend"] = "pandas"
    config["metadata"].update(dataset_name=name, dataset_path=os.path.abspath(csv_path))
    config["time_series"].update(
        enabled=time_series_column is not None,
        datetime_index_column=time_series_column,
    )

    dataset = load(Path(csv_path))
    profile = profile_dataset(dataset)
    session = LegacySession(dataset, profile, config=config, output_dir=out_dir)
    plan = make_plan(
        Registry(legacy_checks(session)), Selection(depth="deep"), profile=profile
    )
    run = execute(plan, dataset, profile, dsbf_version="golden")

    checks = {
        check_id: {
            "status": result.status.value,
            "notes": [note.code for note in result.notes],
            "findings": len(result.findings),
            "result": (result.output or {}).get("result"),
        }
        for check_id, result in run.results.items()
    }
    info = {"shape": [dataset.n_rows, len(dataset.columns)], "checks": checks}
    Path(out_dir, "engine_run.json").write_text(json.dumps(info, default=str))
    Path(out_dir, "run.json").write_text(run.model_dump_json())

    core_registry = Registry()
    core_registry.register_pack(core_pack())
    core_plan = make_plan(
        core_registry,
        Selection(depth="deep", entry_points=False),
        params=json.loads(os.environ.get("DSBF_GOLDEN_CORE_PARAMS", "{}")) or None,
        profile=profile,
    )
    core_run = execute(core_plan, dataset, profile, dsbf_version="golden")
    Path(out_dir, "core_run.json").write_text(core_run.model_dump_json())


if __name__ == "__main__":
    main()
