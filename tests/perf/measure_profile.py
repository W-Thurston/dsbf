"""Child process for the profile-pass runs: load a file, profile it, write the cost.

A separate script in a fresh interpreter, like ``measure_one.py``, so the import time
and the peak memory belong to this run alone. It imports nothing from ``tests``.

The pass is type inference, the dataset profile (which profiles every column) and stage
inference. Loading is timed on its own and is not part of ``total_seconds``, so the
budget is not moved by CSV parsing.

Usage: python measure_profile.py <name> <data file> <out_dir>
"""

import argparse
import json
import resource
import sys
import time
from pathlib import Path

START = time.perf_counter()
from dsbf.data import load  # noqa: E402
from dsbf.profile import (  # noqa: E402
    infer_column_types,
    infer_stage,
    profile_dataset,
)

IMPORT_SECONDS = time.perf_counter() - START

STATUS = Path("/proc/self/status")
CLEAR_REFS = Path("/proc/self/clear_refs")
MIB = 2**20


def _status_mb(field: str) -> float:
    """A ``kB`` field of ``/proc/self/status`` in MiB."""
    for line in STATUS.read_text().splitlines():
        if line.startswith(f"{field}:"):
            return int(line.split()[1]) / 1024
    raise KeyError(field)


def rss_mb() -> float:
    """Resident set size now."""
    return _status_mb("VmRSS")


def reset_peak() -> bool:
    """Reset the peak resident size to the current one; False where Linux can't."""
    try:
        CLEAR_REFS.write_text("5")
        _status_mb("VmHWM")
    except (OSError, KeyError):
        return False
    return True


def peak_mb() -> float:
    """Peak resident size since the last reset (``ru_maxrss`` is KiB on Linux)."""
    try:
        return _status_mb("VmHWM")
    except (OSError, KeyError):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / MIB if sys.platform == "darwin" else peak / 2**10


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("data")
    parser.add_argument("out_dir")
    args = parser.parse_args()

    stages: dict[str, float] = {}

    started = time.perf_counter()
    dataset = load(args.data, name=args.name)
    stages["load"] = time.perf_counter() - started
    data_mb = dataset.frame.estimated_size() / MIB

    # From here the peak is the pass's: the loaded frame is already resident.
    isolated = reset_peak()
    resident_mb = rss_mb()

    started = time.perf_counter()
    types = infer_column_types(dataset)
    stages["infer_column_types"] = time.perf_counter() - started

    started = time.perf_counter()
    profile = profile_dataset(dataset, types)
    stages["profile_dataset"] = time.perf_counter() - started

    started = time.perf_counter()
    stage = infer_stage(profile)
    stages["infer_stage"] = time.perf_counter() - started

    pass_overhead_mb = max(peak_mb() - resident_mb, 0.0)
    total_seconds = sum(v for k, v in stages.items() if k != "load")
    result = {
        "shape": list(dataset.frame.shape),
        "inferred_stage": stage,
        "import_seconds": round(IMPORT_SECONDS, 3),
        "total_seconds": round(total_seconds, 3),
        "stages": {k: round(v, 3) for k, v in stages.items()},
        "peak_rss_mb": round(peak_mb(), 1),
        "data_mb": round(data_mb, 1),
        "pass_overhead_mb": round(pass_overhead_mb, 1),
        "memory_ratio": round(1 + pass_overhead_mb / data_mb, 3),
        "peak_isolated": isolated,
        "n_columns_profiled": len(profile.columns),
    }
    (Path(args.out_dir) / "perf_run.json").write_text(json.dumps(result))


if __name__ == "__main__":
    main()
