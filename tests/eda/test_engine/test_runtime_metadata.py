# tests/eda/test_engine/test_runtime_metadata.py

import json

import pytest


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_metadata_file_exists(clean_engine_run):
    report_path = clean_engine_run()
    metadata_path = report_path.parent / "metadata_report.json"
    assert (
        metadata_path.exists()
    ), f"metadata_report.json was not created at {metadata_path}"


@pytest.mark.filterwarnings("ignore::PendingDeprecationWarning")
def test_report_and_metadata_are_separate(clean_engine_run):
    report_path = clean_engine_run()
    metadata_path = report_path.parent / "metadata_report.json"

    assert report_path.exists(), f"report.json not found at {report_path}"
    assert metadata_path.exists(), f"metadata_report.json not found at {metadata_path}"

    with report_path.open("r") as f1, metadata_path.open("r") as f2:
        report = json.load(f1)
        metadata = json.load(f2)

    assert isinstance(report, dict)
    assert isinstance(metadata, dict)
    assert report != metadata, "Report and metadata files should differ"
