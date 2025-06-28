# dsbf/eda/tasks/detect_encoded_columns.py

import math
import re
import statistics
from collections import Counter

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult
from dsbf.utils.backend import is_polars, is_text_polars


@register_task(
    name="detect_encoded_columns",
    display_name="Detect Encoded Columns",
    description=(
        "Detects columns containing base64, hex, UUID,"
        " or other suspiciously encoded data."
    ),
    depends_on=["infer_types"],
    tags=["format", "encoded", "anomaly"],
    stage="cleaned",
    inputs=["dataframe"],
    outputs=["TaskResult"],
)
class DetectEncodedColumns(BaseTask):
    """
    Detects columns that contain hash-like or encoded data such as
        base64, hex, or UUIDs.
    Flags based on entropy, uniform length, and restricted character sets.
    """

    def run(self) -> None:
        try:
            df = self.input_data
            cfg = self.config.get("tasks", {}).get("detect_encoded_columns", {})

            min_entropy = cfg.get("min_entropy", 4.5)
            length_std_threshold = cfg.get("length_std_threshold", 2.0)
            detect_base64 = cfg.get("detect_base64", True)
            detect_hex = cfg.get("detect_hex", True)
            detect_uuid = cfg.get("detect_uuid", True)

            flagged_columns = []
            details = {}
            recommendations = []

            for col in df.columns:
                column = df.get_column(col)
                if not is_polars(df) or not is_text_polars(column):
                    continue

                charsets = {
                    "base64": re.compile(r"^[A-Za-z0-9+/=]+$"),
                    "hex": re.compile(r"^[0-9a-fA-F]+$"),
                    "uuid": re.compile(
                        (
                            r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]"
                            "{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
                        ),
                        re.I,
                    ),
                }

                try:
                    values = column.drop_nulls().to_list()
                except Exception:
                    continue

                if len(values) < 10:
                    continue  # Skip small columns

                match_type = None
                if detect_uuid and all(
                    charsets["uuid"].fullmatch(v) for v in values[:50]
                ):
                    match_type = "uuid"
                elif detect_hex and all(
                    charsets["hex"].fullmatch(v) for v in values[:50]
                ):
                    match_type = "hex"
                elif detect_base64 and all(
                    charsets["base64"].fullmatch(v) for v in values[:50]
                ):
                    match_type = "base64"

                # Entropy + length uniformity check
                lengths = [len(v) for v in values]
                avg_len = statistics.mean(lengths)
                std_len = statistics.stdev(lengths) if len(lengths) > 1 else 0

                all_chars = "".join(values)
                freqs = Counter(all_chars)
                probs = [v / len(all_chars) for v in freqs.values() if v > 0]
                entropy = -sum(p * math.log2(p) for p in probs)

                if match_type or (
                    entropy > min_entropy and std_len < length_std_threshold
                ):
                    flagged_columns.append(col)
                    details[col] = {
                        "match_type": match_type or "high_entropy",
                        "avg_length": avg_len,
                        "length_std": std_len,
                        "entropy": entropy,
                        "sample_values": values[:5],
                    }
                    recommendations.append(
                        f"Column '{col}' appears to contain {match_type or 'encoded'}"
                        f" strings. Consider decoding or excluding from modeling."
                    )

            summary = {
                "num_encoded_columns": len(flagged_columns),
                "columns": flagged_columns,
            }

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary=summary,
                data=details,
                recommendations=recommendations,
            )

        except Exception as e:
            self.output = TaskResult(
                name=self.name,
                status="failed",
                summary={"message": f"Error during Encoded Column Detection: {e}"},
                data=None,
                metadata={"exception": type(e).__name__},
            )
