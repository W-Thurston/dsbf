"""Child process for the reference golden runs: drift against a perturbed copy.

Loads the CSV, derives a reference from it by a fixed perturbation, then runs the three
legacy drift tasks (with the reference as the session's frame) and the core
``reference_drift`` check (with the reference's profile), and writes both to
``reference_run.json``. A separate interpreter, as in ``engine_one.py``, because
``PYTHONHASHSEED`` can only be set before Python starts. It imports nothing from
``tests``.

The reference is the data with every third numeric column moved by half its standard
deviation, 30% of the first categorical column's values set to its most frequent value,
20% of the first numeric column set to null, and the last column removed. The target is
the first categorical column with two to ten values, if there is one.

Usage: python reference_one.py <name> <csv> <out_dir>
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from dsbf.checks import core_pack
from dsbf.config import load_default_config
from dsbf.data import load
from dsbf.engine import Reference, Registry, Selection, execute, make_plan
from dsbf.legacy import LegacySession, legacy_checks
from dsbf.profiling import profile_dataset

TASKS = (
    "compare_with_reference_dataset",
    "detect_feature_drift",
    "detect_target_drift",
)
SEED = 7


def perturbed(frame: pd.DataFrame) -> pd.DataFrame:
    """The reference: ``frame`` moved, reweighted, hollowed and shortened."""
    rng = np.random.default_rng(SEED)
    out = frame.copy()
    numeric = [
        column
        for column in out.columns
        if pd.api.types.is_numeric_dtype(out[column])
        and not pd.api.types.is_bool_dtype(out[column])
        and out[column].nunique() > 10
    ]
    for column in numeric[::3]:
        spread = out[column].std()
        out[column] = out[column].astype(float) + 0.5 * (spread if spread else 1.0)
    categorical = [
        column
        for column in out.columns
        if out[column].dtype == object and 1 < out[column].nunique() <= 20
    ]
    for column in categorical[:1]:
        top = out[column].mode()[0]
        out.loc[rng.random(len(out)) < 0.3, column] = top
    if numeric:
        out[numeric[0]] = out[numeric[0]].astype(float)
        out.loc[rng.random(len(out)) < 0.2, numeric[0]] = np.nan
    if len(out.columns) > 1:
        out = out.drop(columns=[out.columns[-1]])
    return out


def target_of(frame: pd.DataFrame) -> str | None:
    for column in frame.columns:
        if frame[column].dtype == object and 2 <= frame[column].nunique() <= 10:
            return column
    return None


def main() -> None:
    name, csv_path, out_dir = sys.argv[1:4]
    dataset = load(Path(csv_path))
    profile = profile_dataset(dataset)
    current = dataset.frame.to_pandas()
    target = target_of(current)
    reference_frame = pl.from_pandas(perturbed(current))
    reference = load(reference_frame, name="reference")

    config = load_default_config()
    config["output_dir"] = out_dir
    # Two of the tasks need Polars frames (`get_column`, `drop_nulls`); pandas is the
    # library's default backend, so the legacy `detect_feature_drift` and
    # `detect_target_drift` fail there.
    config["engine"]["backend"] = "polars"
    config["metadata"].update(dataset_name=name, dataset_path=os.path.abspath(csv_path))
    config["tasks"].setdefault("detect_target_drift", {})["target"] = target
    session = LegacySession(
        dataset,
        profile,
        config=config,
        output_dir=out_dir,
        reference_data=reference.frame,
    )
    plan = make_plan(
        Registry(legacy_checks(session)),
        Selection(include=list(TASKS), depth="deep"),
        profile=profile,
    )
    legacy = execute(plan, dataset, profile, dsbf_version="golden")
    tasks = {
        task: {
            "status": legacy.results[task].status.value,
            "notes": [note.code for note in legacy.results[task].notes],
            "findings": len(legacy.results[task].findings),
            "result": (legacy.results[task].output or {}).get("result"),
        }
        for task in TASKS
    }

    registry = Registry()
    registry.register_pack(core_pack())
    core_plan = make_plan(
        registry,
        Selection(include="reference_drift", entry_points=False),
        params={"reference_drift": {"target": target}} if target else None,
        profile=profile,
        has_reference=True,
    )
    core = execute(
        core_plan,
        dataset,
        profile,
        dsbf_version="golden",
        reference=Reference.of_data(reference, profile_dataset(reference)),
    )
    Path(out_dir, "reference_run.json").write_text(
        json.dumps(
            {
                "target": target,
                "legacy": tasks,
                "core": json.loads(core.model_dump_json()),
            },
            default=str,
        )
    )


if __name__ == "__main__":
    main()
