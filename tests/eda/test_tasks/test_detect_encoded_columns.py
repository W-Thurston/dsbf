# tests/eda/test_tasks/test_detect_encoded_columns.py

import pandas as pd
import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_encoded_columns import DetectEncodedColumns
from tests.helpers.context_utils import make_ctx_and_task, run_task_with_dependencies


def test_detects_base64_strings(tmp_path):
    """Base64-encoded strings must be detected and classified correctly."""
    df = pl.DataFrame(
        {
            "token": [
                "aGVsbG8=",
                "d29ybGQ=",
                "Zm9vYmFy",
                "YmxhaA==",
                "Y2hhcg==",
                "dGVzdDE=",
                "dGVzdDI=",
                "dGVzdDM=",
                "dGVzdDQ=",
                "dGVzdDU=",
            ],
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectEncodedColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    # Inject semantic type directly - infer_types classifies high-uniqueness
    # string columns as 'id'. We inject 'categorical' to test detection logic
    # independently of type inference decisions.
    ctx.set_metadata("semantic_types", {"token": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result.status == "success"
    assert result.summary["num_encoded_columns"] == 1
    assert "token" in result.summary["columns"]
    assert result.data["token"]["match_type"] == "base64"


def test_detects_hex_strings(tmp_path):
    """Hex-encoded strings must be detected and classified correctly."""
    df = pl.DataFrame(
        {
            "hex_id": [
                "deadbeef",
                "cafebabe",
                "123abc",
                "456def",
                "0a0b0c",
                "abcdef",
                "987654",
                "00ffcc",
                "badc0de",
                "feedface",
            ],
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectEncodedColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    # Inject semantic type directly - infer_types classifies high-uniqueness
    # string columns as 'id'. We inject 'categorical' to test detection logic
    # independently of type inference decisions.
    ctx.set_metadata("semantic_types", {"hex_id": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["num_encoded_columns"] == 1
    assert "hex_id" in result.summary["columns"]
    assert result.data["hex_id"]["match_type"] == "hex"


def test_detects_uuid_strings(tmp_path):
    """UUID strings must be detected and classified correctly."""
    df = pl.DataFrame(
        {
            "uuid": [
                "550e8400-e29b-41d4-a716-446655440000",
                "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                "123e4567-e89b-12d3-a456-426614174000",
                "c56a4180-65aa-42ec-a945-5fd21dec0538",
                "f9c28bfb-3d0a-4d58-a3f6-859c46c9d2f6",
                "c9bf9e57-1685-4c89-bafb-ff5af830be8a",
                "7c9e6679-7425-40de-944b-e07fc1f90ae7",
                "16fd2706-8baf-433b-82eb-8c7fada847da",
                "e902893a-9d22-3c7e-a7b8-d6e313b71d9f",
                "2c1b8d1e-bc1a-4b3e-a9ef-3b1d6c57cf23",
            ],
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectEncodedColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    # Inject semantic type directly - infer_types classifies high-uniqueness
    # string columns as 'id'. We inject 'categorical' to test detection logic
    # independently of type inference decisions.
    ctx.set_metadata("semantic_types", {"uuid": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["num_encoded_columns"] == 1
    assert "uuid" in result.summary["columns"]
    assert result.data["uuid"]["match_type"] == "uuid"


def test_ignores_regular_text_columns(tmp_path):
    """Plain natural language text must not be flagged as encoded."""
    # Two-word phrases contain spaces, which are outside [A-Za-z0-9+/=],
    # so base64 fullmatch fails entirely. No hex-only chars either.
    # Repeated 3x to keep unique_ratio below 0.9.
    df = pl.DataFrame(
        {
            "names": [
                "red car",
                "blue sky",
                "green tea",
                "hot dog",
                "cold air",
                "old map",
                "new bag",
                "big cat",
                "wet dog",
                "dry cup",
            ]
            * 3,
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectEncodedColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"names": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, DetectEncodedColumns)

    assert result.status == "success"
    assert result.summary["num_encoded_columns"] == 0


def test_ignores_low_entropy_text(tmp_path):
    """
    Low-entropy text (e.g. constant repeated strings) must not be flagged.

    Uses ≥ 10 rows to avoid the minimum-sample-size guard.
    """
    # Two-word phrases: spaces break base64/hex patterns. Low entropy
    # (repetitive structure) tests the entropy threshold guard.
    # Repeated 3x to keep unique_ratio below 0.9.
    df = pl.DataFrame(
        {
            "letters": [
                "word one",
                "word two",
                "word three",
                "word four",
                "word five",
                "word six",
                "word seven",
                "word eight",
                "word nine",
                "word ten",
            ]
            * 3,
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectEncodedColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    ctx.set_metadata("semantic_types", {"letters": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, DetectEncodedColumns)

    assert result.status == "success"
    assert result.summary["num_encoded_columns"] == 0


def test_detects_high_entropy_without_regex_match(tmp_path) -> None:
    """High-entropy uniform-length strings must be flagged even w/o a regex match."""
    df = pl.DataFrame(
        {
            "hashy": [
                "xR7f9zPq",
                "Wm2Kq9Bn",
                "aY3dLp0Z",
                "eX1jUv5N",
                "qT4hZj8M",
                "bD6nGs7L",
                "Hr9pVw2X",
                "tC3fKy1Q",
                "uZ0oRw6Y",
                "mL5sAx9E",
            ],
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectEncodedColumns,
        current_df=df,
        task_overrides={
            "min_entropy": 3.0,
            "length_std_threshold": 1.0,
            "detect_base64": False,
            "detect_hex": False,
            "detect_uuid": False,
        },
        global_overrides={"output_dir": str(tmp_path)},
    )
    # Inject semantic type directly - infer_types classifies high-uniqueness
    # string columns as 'id'. We inject 'categorical' to test detection logic
    # independently of type inference decisions.
    ctx.set_metadata("semantic_types", {"hashy": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.summary["num_encoded_columns"] == 1
    assert "hashy" in result.summary["columns"]
    assert result.data["hashy"]["match_type"] == "high_entropy"


def test_pandas_dataframe_handled(tmp_path):
    """Task must process pandas DataFrames correctly (validates the pandas bug fix)."""
    df = pd.DataFrame(
        {
            "token": [
                "deadbeef",
                "cafebabe",
                "123abc",
                "456def",
                "0a0b0c",
                "abcdef",
                "987654",
                "00ffcc",
                "badc0de",
                "feedface",
            ],
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectEncodedColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    # Inject semantic type directly - infer_types classifies high-uniqueness
    # string columns as 'id'. We inject 'categorical' to test detection logic
    # independently of type inference decisions.
    ctx.set_metadata("semantic_types", {"token": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    # If the pandas path works, hex detection should fire
    assert result.summary["num_encoded_columns"] == 1
    assert "token" in result.summary["columns"]


def test_guidance_attached_for_flagged_columns(tmp_path):
    """EDA guidance blurbs must be attached for each detected encoded column."""
    df = pl.DataFrame(
        {
            "uuid": [
                "550e8400-e29b-41d4-a716-446655440000",
                "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                "123e4567-e89b-12d3-a456-426614174000",
                "c56a4180-65aa-42ec-a945-5fd21dec0538",
                "f9c28bfb-3d0a-4d58-a3f6-859c46c9d2f6",
                "c9bf9e57-1685-4c89-bafb-ff5af830be8a",
                "7c9e6679-7425-40de-944b-e07fc1f90ae7",
                "16fd2706-8baf-433b-82eb-8c7fada847da",
                "e902893a-9d22-3c7e-a7b8-d6e313b71d9f",
                "2c1b8d1e-bc1a-4b3e-a9ef-3b1d6c57cf23",
            ],
        },
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectEncodedColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    # Inject semantic type directly - infer_types classifies high-uniqueness
    # string columns as 'id'. We inject 'categorical' to test detection logic
    # independently of type inference decisions.
    ctx.set_metadata("semantic_types", {"uuid": "categorical"})
    result: TaskResult = ctx.run_task(task)

    assert result.status == "success"
    assert result.guidance is not None
    assert "uuid" in result.guidance
    assert len(result.guidance["uuid"]["eda"]) > 0
    # Encoded columns are EDA-only (eda phase) - no ML blurb
    assert result.guidance["uuid"]["eda"][0]["level"] == "info"


def test_no_plots_generated(tmp_path):
    """Encoded column detection must not generate plots."""
    df = pl.DataFrame(
        {
            "hex_id": [
                "deadbeef",
                "cafebabe",
                "123abc",
                "456def",
                "0a0b0c",
                "abcdef",
                "987654",
                "00ffcc",
                "badc0de",
                "feedface",
            ],
        },
    )

    ctx, _ = make_ctx_and_task(
        task_cls=DetectEncodedColumns,
        current_df=df,
        global_overrides={"output_dir": str(tmp_path)},
    )
    # Inject semantic type directly - infer_types classifies high-uniqueness
    # string columns as 'id'. We inject 'categorical' to test detection logic
    # independently of type inference decisions.
    ctx.set_metadata("semantic_types", {"hex_id": "categorical"})
    result: TaskResult = run_task_with_dependencies(ctx, DetectEncodedColumns)

    assert result.status == "success"
    assert result.plots is None
