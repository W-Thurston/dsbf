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
    profiling_depth="standard",
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
            # ctx = self.context
            df = self.input_data

            min_entropy = float(self.get_task_param("min_entropy") or 4.5)
            length_std_threshold = float(
                self.get_task_param("length_std_threshold") or 2.0
            )
            detect_base64 = self.get_task_param("detect_base64", True)
            detect_hex = self.get_task_param("detect_hex", True)
            detect_uuid = self.get_task_param("detect_uuid", True)

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

                # Entropy + length uniformity check
                lengths = [len(v) for v in values]
                avg_len = statistics.mean(lengths)
                std_len = statistics.stdev(lengths) if len(lengths) > 1 else 0

                all_chars = "".join(values)
                freqs = Counter(all_chars)
                probs = [v / len(all_chars) for v in freqs.values() if v > 0]
                entropy = -sum(p * math.log2(p) for p in probs)

                match_type = None

                # Entropy + length check first
                if entropy > min_entropy and std_len < length_std_threshold:
                    match_type = "high_entropy"

                # Only apply pattern match if no entropy-based match
                if match_type is None:
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

                if match_type:
                    self._log(f"Flagged column '{col}' as {match_type}", "debug")
                    flagged_columns.append(col)
                    self._log(f"Flagged column '{col}' as {match_type}", "debug")
                    details[col] = {
                        "match_type": match_type,
                        "avg_length": avg_len,
                        "length_std": std_len,
                        "entropy": entropy,
                        "sample_values": values[:5],
                    }
                    recommendations.append(
                        f"Column '{col}' appears to contain {match_type} strings."
                        f"Consider decoding or excluding from modeling."
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
