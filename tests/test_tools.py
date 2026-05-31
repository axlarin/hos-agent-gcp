"""Unit tests for individual tool functions."""
import io
import pytest
import pandas as pd

import tools.csv_tools as csv_tools
from tools.analysis_tools import _find_column, run_categorical_analysis, run_group_comparison


@pytest.fixture(autouse=True)
def _inject_fake_dataset():
    """Inject a minimal fake DataFrame so tests don't need real HOS files."""
    df = pd.DataFrame({
        "AGE": [65, 70, 75, 65, 80, 72, 68, 77],
        "SEX": [1, 2, 1, 2, 1, 2, 1, 2],
        "GENHEALT": [1, 2, 3, 2, 1, 3, 2, 4],
    })
    csv_tools._datasets["fake_ds"] = df
    yield
    csv_tools._datasets.pop("fake_ds", None)


def test_find_column_case_insensitive():
    df = pd.DataFrame({"MyCol": [1]})
    assert _find_column(df, "mycol") == "MyCol"


def test_find_column_missing_raises():
    df = pd.DataFrame({"A": [1]})
    with pytest.raises(KeyError):
        _find_column(df, "NOTEXIST")


def test_list_datasets_returns_fake():
    result = csv_tools.list_datasets()
    assert "fake_ds" in result


def test_run_categorical_analysis_single_column():
    result = run_categorical_analysis("fake_ds", "SEX")
    assert "Frequency" in result or "frequency" in result.lower()
    assert "1" in result and "2" in result


def test_run_group_comparison_two_groups():
    result = run_group_comparison("fake_ds", "GENHEALT", "SEX")
    assert "Mann-Whitney" in result or "mann-whitney" in result.lower()
