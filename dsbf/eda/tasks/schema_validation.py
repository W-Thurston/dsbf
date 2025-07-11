# dsbf/eda/tasks/schema_validation.py

from dsbf.core.base_task import BaseTask
from dsbf.eda.task_registry import register_task
from dsbf.eda.task_result import TaskResult, add_reliability_warning
from dsbf.utils.backend import is_polars


@register_task(
    name="schema_validation",
    display_name="Schema Validation",
    description=(
        "Validates dataset against a declared schema: "
        "required columns, types, ranges, and categories."
    ),
    tags=["validation", "schema", "contract", "power_user"],
    stage="raw",
    domain="core",
    profiling_depth="full",
    runtime_estimate="fast",
    inputs=["dataframe"],
    outputs=["TaskResult"],
    experimental=False,
    expected_semantic_types=["any"],
)
class SchemaValidation(BaseTask):
    def run(self):

        ctx = self.context
        if ctx is None:
            raise RuntimeError("SchemaValidationTask requires AnalysisContext.")

        schema_cfg = ctx.get_config("schema_validation") or {}
        if not schema_cfg.get("enable_schema_validation", False):
            self._log(
                "    [schema_validation] Skipping — validation disabled in config.",
                level="debug",
            )
            self.output = TaskResult(
                name=self.name,
                status="skipped",
                summary={"message": "Validation disabled."},
            )
            return

        df = self.input_data
        is_pl = is_polars(df)
        fail_or_warn = schema_cfg.get("fail_or_warn", "warn").lower()
        schema = schema_cfg.get("schema", {})

        VALID_SCHEMA_KEYS = {"required_columns", "dtypes", "value_ranges", "categories"}
        unknown_keys = set(schema.keys()) - VALID_SCHEMA_KEYS
        if unknown_keys:
            if self.output is None:
                self.output = TaskResult(name=self.name)
            for key in unknown_keys:
                add_reliability_warning(
                    self.output,
                    level="warning",
                    code=f"unknown_schema_key_{key}",
                    description=f"Unknown schema key: '{key}'. This will be ignored.",
                    recommendation=(
                        "Use only: required_columns, dtypes,"
                        " value_ranges, categories"
                    ),
                )

        required_cols = schema.get("required_columns", [])
        expected_dtypes = schema.get("dtypes", {})
        value_ranges = schema.get("value_ranges", {})
        allowed_categories = schema.get("categories", {})

        result_data = {
            "missing_columns": [],
            "dtype_mismatches": {},
            "mixed_type_columns": [],
            "value_range_violations": {},
            "unexpected_categories": {},
        }

        # --- Required columns ---
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            result_data["missing_columns"] = missing
            if self.output is None:
                self.output = TaskResult(name=self.name)
            for col in missing:
                add_reliability_warning(
                    self.output,
                    "error",
                    f"missing_col_{col}",
                    f"Missing required column: {col}",
                )

        # --- Dtype mismatches ---
        for col, expected_type in expected_dtypes.items():
            if col not in df.columns:
                continue
            actual_dtype = (
                str(df[col].dtype).lower() if is_pl else str(df[col].dtype.name).lower()
            )
            expected_type_normalized = expected_type.lower()

            # Optionally normalize synonyms
            dtype_aliases = {
                "float": ["float64", "float32"],
                "str": ["string", "str", "object"],
                "int": ["int64", "int32"],
            }
            valid_expected = dtype_aliases.get(
                expected_type_normalized, [expected_type_normalized]
            )

            if not any(t in actual_dtype for t in valid_expected):
                result_data["dtype_mismatches"][col] = actual_dtype
                if self.output is None:
                    self.output = TaskResult(name=self.name)
                add_reliability_warning(
                    self.output,
                    "warning",
                    f"dtype_{col}",
                    (
                        f"Column '{col}' expected type '{expected_type}',"
                        f" found '{actual_dtype}'"
                    ),
                )

        # --- Mixed type detection --- (fallback to pandas)
        if is_pl:
            df_pd = df.to_pandas()
        else:
            df_pd = df
        for col in df_pd.columns:
            unique_types = df_pd[col].map(type).nunique()
            if unique_types > 1:
                result_data["mixed_type_columns"].append(col)
                if self.output is None:
                    self.output = TaskResult(name=self.name)
                add_reliability_warning(
                    self.output,
                    "warning",
                    f"mixed_type_{col}",
                    f"Column '{col}' contains mixed data types",
                )

        # --- Value range checks ---
        for col, bounds in value_ranges.items():
            if col not in df.columns:
                continue
            try:
                series = df[col].to_numpy() if is_pl else df[col].values
                violations = []
                if "min" in bounds:
                    below = series < bounds["min"]
                    if below.sum() > 0:
                        violations.append(f"{below.sum()} < min")
                if "max" in bounds:
                    above = series > bounds["max"]
                    if above.sum() > 0:
                        violations.append(f"{above.sum()} > max")
                if violations:
                    result_data["value_range_violations"][col] = violations
                    if self.output is None:
                        self.output = TaskResult(name=self.name)
                    add_reliability_warning(
                        self.output,
                        "warning",
                        f"range_{col}",
                        (
                            f"Column '{col}' has values out "
                            f"of declared bounds: {violations}"
                        ),
                    )
            except Exception:
                continue

        # --- Categorical value checks ---
        for col, allowed in allowed_categories.items():
            if col not in df.columns:
                continue
            observed = (
                df[col].unique().to_list() if is_pl else df[col].unique().tolist()
            )
            extras = [val for val in observed if val not in allowed]
            if extras:
                result_data["unexpected_categories"][col] = extras
                if self.output is None:
                    self.output = TaskResult(name=self.name)
                add_reliability_warning(
                    self.output,
                    "warning",
                    f"categories_{col}",
                    f"Column '{col}' has values outside allowed set: {extras}",
                )

        # --- Finalize result ---
        has_errors = bool(result_data["missing_columns"])
        status = "failed" if fail_or_warn == "fail" and has_errors else "success"
        msg = (
            "Schema validated successfully."
            if status == "success"
            else "Schema validation failed."
        )

        if self.output is None:
            self.output = TaskResult(name=self.name)

        self.output.status = status
        self.output.summary = {"message": msg}
        self.output.data = result_data
        self.output.recommendations = self.output.recommendations or []
