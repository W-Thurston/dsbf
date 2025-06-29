import polars as pl

from dsbf.eda.task_result import TaskResult
from dsbf.eda.tasks.detect_encoded_columns import DetectEncodedColumns
from tests.helpers.context_utils import make_ctx_and_task


def test_detects_base64_strings():
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
            ]
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectEncodedColumns,
        current_df=df,
    )
    result = ctx.run_task(task)

    assert isinstance(result, TaskResult)
    assert result is not None
    assert result.status == "success"
    assert result.summary["num_encoded_columns"] == 1
    assert "token" in result.summary["columns"]
    assert result.data is not None
    assert result.data["token"]["match_type"] == "base64"


def test_detects_hex_strings():
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
            ]
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectEncodedColumns,
        current_df=df,
    )
    result = ctx.run_task(task)

    assert result is not None
    assert result.status == "success"
    assert result.summary["num_encoded_columns"] == 1
    assert "hex_id" in result.summary["columns"]
    assert result.data is not None
    assert result.data["hex_id"]["match_type"] == "hex"


def test_detects_uuid_strings():
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
            ]
        }
    )

    ctx, task = make_ctx_and_task(
        task_cls=DetectEncodedColumns,
        current_df=df,
    )
    result = ctx.run_task(task)

    assert result is not None
    assert result.status == "success"
    assert result.summary["num_encoded_columns"] == 1
    assert "uuid" in result.summary["columns"]
    assert result.data is not None
    assert result.data["uuid"]["match_type"] == "uuid"


def test_ignores_regular_text_columns():
    df = pl.DataFrame({"names": ["alice", "bob", "charlie", "dave"]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectEncodedColumns,
        current_df=df,
    )
    result = ctx.run_task(task)

    assert result is not None
    assert result.status == "success"
    assert result.summary["num_encoded_columns"] == 0


def test_ignores_low_entropy_text():
    df = pl.DataFrame({"letters": ["aaaa", "bbbb", "cccc", "dddd", "eeee"]})

    ctx, task = make_ctx_and_task(
        task_cls=DetectEncodedColumns,
        current_df=df,
    )
    result = ctx.run_task(task)

    assert result is not None
    assert result.status == "success"
    assert result.summary["num_encoded_columns"] == 0


def test_detects_high_entropy_column_without_regex_match():
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
            ]
        }
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
    )
    result = ctx.run_task(task)

    assert result is not None
    assert result.status == "success"
    assert result.summary["num_encoded_columns"] == 1
    assert "hashy" in result.summary["columns"]
    assert result.data is not None
    assert result.data["hashy"]["match_type"] == "high_entropy"
