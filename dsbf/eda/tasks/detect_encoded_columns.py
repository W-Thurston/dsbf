# dsbf/eda/tasks/detect_encoded_columns.py

import math
import re
import statistics
from collections import Counter
from re import Pattern
from typing import Any, Literal

import polars as pl

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, make_failure_result
from dsbf.utils.backend import is_polars
from dsbf.utils.reco_engine import get_recommendation_tip


@register_task(
    name="detect_encoded_columns",
    display_name="Detect Encoded Columns",
    description=(
        "Detects columns containing base64, hex, UUID, or other "
        "suspiciously encoded data."
    ),
    depends_on=["infer_types"],
    profiling_depth="standard",
    stage="cleaned",
    domain="core",
    runtime_estimate="fast",
    phase="eda",
    tags=["format", "encoded", "anomaly"],
    expected_semantic_types=["text", "categorical"],
)
class DetectEncodedColumns(BaseTask):
    """
    Detects columns containing hash-like or encoded data.

    Flags columns containing values that match known encoded formats:

    - **UUID**: 8-4-4-4-12 hex character pattern with version/variant bits.
    - **Hex**: Values composed entirely of hexadecimal characters.
    - **Base64**: Values composed of ``[A-Za-z0-9+/=]`` characters.
    - **High entropy**: Shannon entropy above threshold with low length variance
      - catches hashes and fingerprints not matching the above patterns.

    Detection runs on both Polars and Pandas DataFrames. For Polars, only
    columns with String/Utf8 dtype are processed. For Pandas, only object dtype
    columns are processed.

    Encoded columns are rarely useful as model features and often indicate IDs,
    fingerprints, or opaque system-generated keys that should be excluded from
    analysis.

    Configurable parameters (via config["tasks"]["detect_encoded_columns"]):
        min_entropy (float): Shannon entropy threshold. Default: 4.5
        length_std_threshold (float): Maximum string length std for high-entropy
            detection. Default: 2.0
        detect_base64 (bool): Enable base64 pattern matching. Default: True
        detect_hex (bool): Enable hex pattern matching. Default: True
        detect_uuid (bool): Enable UUID pattern matching. Default: True
    """

    def run(self) -> None:  # noqa: C901, PLR0912, PLR0915
        """
        Execute encoded column detection and populate self.output.

        Raises:
            Exception: Re-raised if a context is present (handled by ExecutionGraph).

        """
        try:
            df = self.input_data

            matched_cols, excluded = self.get_columns_by_intent()
            self._log(
                f"    Processing {len(matched_cols)} ['text', 'categorical'] column(s)",
                "debug",
            )

            min_entropy = float(self.get_task_param("min_entropy") or 4.5)
            length_std_threshold = float(
                self.get_task_param("length_std_threshold") or 2.0,
            )
            detect_base64: Literal[True] | Any = (
                self.get_task_param("detect_base64") or True
            )
            detect_hex: Literal[True] | Any = self.get_task_param("detect_hex") or True
            detect_uuid: Literal[True] | Any = (
                self.get_task_param("detect_uuid") or True
            )

            charsets: dict[str, Pattern[str]] = {
                "base64": re.compile(r"^[A-Za-z0-9+/=]+$"),
                "hex": re.compile(r"^[0-9a-fA-F]+$"),
                "uuid": re.compile(
                    r"^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}"
                    r"-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
                    re.IGNORECASE,
                ),
            }

            flagged_columns: list[str] = []
            details: dict = {}
            recommendations: list[str] = []

            for col in matched_cols:
                try:
                    # Extract string values from either backend
                    if is_polars(df):
                        # Only process string dtype columns
                        if df[col].dtype not in (pl.String, pl.Utf8):
                            continue
                        values = df[col].drop_nulls().to_list()
                    else:
                        # Only process object dtype columns
                        if df[col].dtype != object:
                            continue
                        values = df[col].dropna().astype(str).tolist()
                except Exception as e:  # noqa: BLE001
                    self._log(
                        f"    [{self.name}] Failed extracting values from '{col}': {e}",
                        "debug",
                    )
                    continue

                if len(values) < 10:  # noqa: PLR2004
                    # Too few values for reliable detection
                    continue

                lengths: list[int] = [len(v) for v in values]
                avg_len: int = statistics.mean(lengths)
                std_len: float | int = (
                    statistics.stdev(lengths) if len(lengths) > 1 else 0.0
                )

                all_chars = "".join(values)
                freqs: Counter[str] = Counter(all_chars)
                probs: list[float] = [
                    v / len(all_chars) for v in freqs.values() if v > 0
                ]
                entropy: Literal[0] | float = -sum(p * math.log2(p) for p in probs)

                match_type = None

                # Entropy + length uniformity - catches hashes not matching
                # the specific format patterns below.
                if entropy > min_entropy and std_len < length_std_threshold:
                    match_type = "high_entropy"

                # Format-specific pattern matching on a sample of up to 50 values.
                if match_type is None:
                    sample = values[:50]
                    if detect_uuid and all(
                        charsets["uuid"].fullmatch(v) for v in sample
                    ):
                        match_type = "uuid"
                    elif detect_hex and all(
                        charsets["hex"].fullmatch(v) for v in sample
                    ):
                        match_type = "hex"
                    elif detect_base64 and all(
                        charsets["base64"].fullmatch(v) for v in sample
                    ):
                        match_type = "base64"

                if match_type:
                    flagged_columns.append(col)
                    details[col] = {
                        "match_type": match_type,
                        "avg_length": avg_len,
                        "length_std": std_len,
                        "entropy": entropy,
                        "sample_values": values[:5],
                    }
                    recommendations.append(
                        f"Column '{col}' appears to contain {match_type} strings. "
                        "Consider decoding or excluding from modeling.",
                    )

            self.output = TaskResult(
                name=self.name,
                status="success",
                summary={
                    "num_encoded_columns": len(flagged_columns),
                    "columns": flagged_columns,
                },
                data=details,
                recommendations=recommendations,
                metadata={
                    "suggested_viz_type": "bar",
                    "recommended_section": "Format",
                    "display_priority": "medium",
                    "excluded_columns": excluded,
                    "column_types": self.get_column_type_info(
                        matched_cols + list(excluded.keys()),
                    ),
                },
            )

            for col in flagged_columns:
                self._attach_guidance(col, details[col])

            # ML impact scoring
            if self.get_engine_param("enable_impact_scoring", True) and flagged_columns:
                top_col: str = flagged_columns[0]
                match_type = details[top_col]["match_type"]
                tip: str | None = get_recommendation_tip(
                    self.name,
                    {"match_type": match_type},
                )
                self.set_ml_signals(
                    result=self.output,
                    score=0.9,
                    tags=["drop", "check_leakage"],
                    recommendation=tip
                    or (
                        f"Column '{top_col}' appears to be encoded "
                        f"({match_type}). Consider dropping it to avoid "
                        "overfitting or leakage."
                    ),
                )
                self.output.summary["column"] = top_col

        except Exception as e:
            if self.context:
                raise
            self._log(
                f"    [{self.name}] Task failed outside execution context: "
                f"{type(e).__name__} - {e}",
                level="warn",
            )
            self.output = make_failure_result(self.name, e)

    def _attach_guidance(self, col: str, col_details: dict) -> None:
        """
        Generate EDA guidance for a column flagged as likely encoded.

        Args:
            col: Column name.
            col_details: Detection result dict containing ``match_type``,
                ``entropy``, ``avg_length``, and ``sample_values``.

        """
        match_type = col_details["match_type"]
        entropy = col_details["entropy"]
        avg_len = col_details["avg_length"]

        type_descriptions: dict[str, str] = {
            "uuid": "UUID (Universally Unique Identifier)",
            "hex": "hexadecimal-encoded string",
            "base64": "base64-encoded string",
            "high_entropy": "high-entropy string (likely a hash or fingerprint)",
        }
        description = type_descriptions.get(match_type, match_type)

        eda_body: str = (
            f"'{col}' appears to contain {description} values "
            f"(entropy: {entropy:.2f}, avg length: {avg_len:.1f}). "
            f"These values are opaque identifiers or encoded data that carry no "
            f"interpretable analytical signal in their raw form. They most commonly "
            f"represent system-generated keys, cryptographic hashes, or serialised "
            f"objects. Verify whether this column is a meaningful feature or an "
            f"artefact of the data export process."
        )

        self.add_guidance(
            result=self.output,
            column=col,
            phase="eda",
            level="info",
            title=f"Likely Encoded Column ({match_type})",
            body=eda_body.strip(),
            actions=[],
            metric={
                "match_type": match_type,
                "entropy": round(entropy, 4),
                "avg_length": round(avg_len, 2),
            },
        )
