# tests/eda/test_top_ml_risks_block.py

import json


def test_top_ml_risks_structure(clean_engine_run):
    report_path = clean_engine_run()
    assert report_path.exists()

    with open(report_path) as f:
        report = json.load(f)

    assert "top_ml_risks" in report
