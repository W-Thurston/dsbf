"""``load`` reads files the way pandas does, except where listed in ``accepted``."""

import datetime as dt

import pandas as pd
import polars as pl
import pytest
import xlsxwriter

from dsbf.data import load
from tests.golden.datasets import GOLDEN_DATASETS, GoldenDataset


@pytest.mark.integration
@pytest.mark.parametrize("dataset", GOLDEN_DATASETS, ids=lambda d: d.name)
def test_a_golden_csv_loads_with_pandas_shape_names_and_null_counts(
    dataset: GoldenDataset, tmp_path
):
    path = dataset.write_csv(tmp_path)
    frame = pd.read_csv(path)

    loaded = load(path)

    assert (loaded.n_rows, len(loaded.columns)) == frame.shape
    assert loaded.columns == list(frame.columns)
    nulls = loaded.frame.null_count().row(0, named=True)
    assert nulls == {c: int(n) for c, n in frame.isna().sum().items()}


@pytest.mark.unit
def test_an_all_null_csv_column_is_a_string_not_a_float(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("a,b\n1,\n2,\n")

    loaded = load(path)

    assert pd.read_csv(path)["b"].dtype == "float64"
    assert loaded.schema["b"] == pl.String
    assert loaded.frame["b"].null_count() == 2


@pytest.mark.unit
def test_an_empty_csv_header_keeps_its_name_where_pandas_numbers_it(tmp_path):
    path = tmp_path / "t.csv"
    path.write_text("a,,c\n1,x,3\n")

    assert list(pd.read_csv(path).columns) == ["a", "Unnamed: 1", "c"]
    assert load(path).columns == ["a", "", "c"]


@pytest.mark.unit
def test_an_empty_excel_header_is_named_by_position(tmp_path):
    path = tmp_path / "t.xlsx"
    workbook = xlsxwriter.Workbook(path)
    sheet = workbook.add_worksheet()
    sheet.write_row(0, 0, ["a", "b"])
    sheet.write(0, 3, "d")
    for row in range(1, 3):
        sheet.write_row(row, 0, [row, "x", row, "y"])
    workbook.close()

    assert load(path).columns == ["a", "b", "__UNNAMED__2", "d"]


@pytest.mark.unit
def test_an_excel_column_of_midnight_datetimes_is_a_date(tmp_path):
    path = tmp_path / "t.xlsx"
    workbook = xlsxwriter.Workbook(path)
    sheet = workbook.add_worksheet()
    sheet.write_row(0, 0, ["midnight", "noon"])
    stamp = workbook.add_format({"num_format": "yyyy-mm-dd hh:mm:ss"})
    for row in range(1, 4):
        sheet.write_datetime(row, 0, dt.datetime(2024, 1, row), stamp)
        sheet.write_datetime(row, 1, dt.datetime(2024, 1, row, 12), stamp)
    workbook.close()

    loaded = load(path)

    assert loaded.schema["midnight"] == pl.Date
    assert loaded.schema["noon"] == pl.Datetime
