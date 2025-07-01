# dsbf/utils/report_utils.py

import json
import os
from typing import Any, Dict

from dsbf.core.context import AnalysisContext
from dsbf.eda.task_result import TaskResult
from dsbf.utils.task_utils import is_diagnostic_task


def render_user_report(results: Dict[str, TaskResult], output_path: str) -> None:
    """
    Write user-facing EDA task results (excluding diagnostics) to a JSON report.

    Args:
        results (Dict[str, TaskResult]): All task results from the run.
        output_path (str): Full path to the output report.json file.
    """
    # Filter only EDA tasks (exclude runtime diagnostics)
    eda_only_raw = {
        k: v
        for k, v in results.items()
        if isinstance(v, TaskResult) and not is_diagnostic_task(k)
    }
    eda_only = {k: v.to_dict() for k, v in eda_only_raw.items()}

    # Top 10 risks
    top_ml_risks: list[Dict[str, Any]] = sorted(
        [
            {
                "task": k,
                "score": v.ml_impact_score,
                "column": v.summary.get("column", "unknown"),
                "recommendation": (v.recommendations or [""])[0],
                "tags": v.recommendation_tags,
            }
            for k, v in eda_only_raw.items()
            if v.ml_impact_score is not None
        ],
        key=lambda x: x.get("score", 0.0),
        reverse=True,
    )

    with open(output_path, "w") as f:
        json.dump(
            {
                "results": eda_only,
                "top_ml_risks": top_ml_risks,
            },
            f,
            indent=2,
        )


def write_metadata_report(
    context: AnalysisContext, filename: str = "metadata_report.json"
) -> None:
    """
    Write system-level metadata and diagnostic results to a separate JSON report.

    This includes:
    - Engine and environment metadata (e.g., Python version, Git SHA)
    - Task execution outcomes and durations
    - Run timing information
    - Results from diagnostic tasks like IdentifyBottleneckTasks and LogResourceUsage

    Args:
        context (AnalysisContext): The profiling context containing
            metadata and results.
        filename (str): Name of the metadata output file
            (default: 'metadata_report.json').

    Raises:
        RuntimeError: If context is missing a required output directory.
    """
    # Merge run-time metadata and system stats
    metadata = context.metadata.copy()
    metadata.update(context.run_metadata)

    # Extract diagnostic TaskResults (e.g., runtime logs, bottlenecks)
    diagnostic_keys = [k for k in context.results if is_diagnostic_task(k)]

    metadata["diagnostic_results"] = {
        k: context.results[k].to_dict() for k in diagnostic_keys
    }

    # Determine output path
    output_dir = context.output_dir or "dsbf/outputs/latest"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    # Write metadata as JSON
    with open(output_path, "w") as f:
        json.dump(metadata, f, indent=2)

    # Log result location for debugging
    context._log(f"[write_metadata_report] Metadata written to: {output_path}", "info")
