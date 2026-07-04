"""Unit tests for individual tool functions."""
import io
import numpy as np
import pytest
import pandas as pd

import tools.csv_tools as csv_tools
from tools.analysis_tools import _find_column, run_categorical_analysis, run_group_comparison
from tools.report_tools import (
    GateConfig,
    WORKFLOWS,
    _gate_needs_classification_target,
    _gate_needs_numeric_target,
    _gate_needs_sex_column,
    _gate_needs_sufficient_data,
    _run_workflow,
    generate_health_report,
)


@pytest.fixture(autouse=True)
def _inject_fake_dataset():
    """Inject a minimal fake DataFrame so tests don't need real HOS files."""
    df = pd.DataFrame({
        "AGE": [65, 70, 75, 65, 80, 72, 68, 77, 74, 69],
        "SEX": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "GENHEALT": [1, 2, 3, 2, 1, 3, 2, 4, 2, 3],
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


# ── Report workflow tests ─────────────────────────────────────────────────────

def test_workflow_definitions_are_non_empty():
    """WORKFLOWS is the single source of truth — every defined workflow must have steps."""
    assert WORKFLOWS, "WORKFLOWS must not be empty"
    for name, steps in WORKFLOWS.items():
        assert steps, f"Workflow '{name}' has no steps"


def test_run_workflow_happy_path():
    result = _run_workflow("fake_ds", "GENHEALT", "health_profile")
    assert result.resolved_column == "GENHEALT"
    assert result.summary["completed"] >= 1
    assert "HOS Health Profile Report" in result.report_markdown
    assert "Execution Summary" in result.report_markdown
    # to_dict covers every field including config
    d = result.to_dict()
    assert "config" in d
    assert "steps" in d
    assert d["summary"]["total"] == len(WORKFLOWS["health_profile"])


def test_gate_skips_sex_comparison_when_single_value():
    df_one_sex = pd.DataFrame({
        "AGE": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        "SEX": [1] * 10,
        "GENHEALT": [1, 2, 3, 2, 1, 3, 2, 4, 2, 3],
    })
    csv_tools._datasets["_one_sex"] = df_one_sex
    try:
        result = _run_workflow("_one_sex", "GENHEALT")
        sex_step = next(s for s in result.steps if s.name == "Group Comparison by Sex")
        assert sex_step.status == "skipped"
        assert "1 unique value" in sex_step.reason
    finally:
        csv_tools._datasets.pop("_one_sex", None)


def test_gate_skips_feature_importance_for_high_cardinality():
    cfg = GateConfig(max_unique_for_classification=5)
    df = pd.DataFrame({
        "AGE": list(range(10)),
        "SEX": [1, 2] * 5,
        "TARGET": list(range(10)),  # 10 unique values > threshold of 5
    })
    csv_tools._datasets["_high_card"] = df
    try:
        result = _run_workflow("_high_card", "TARGET", config=cfg)
        fi_step = next(s for s in result.steps if "Predictor" in s.name)
        assert fi_step.status == "skipped"
        assert "unique values" in fi_step.reason
    finally:
        csv_tools._datasets.pop("_high_card", None)


def test_configurable_missing_data_threshold():
    df = pd.DataFrame({
        "AGE": [1, 2] * 5,
        "SEX": [1, 2] * 5,
        "TARGET": [1.0, 2.0, np.nan, np.nan, 1.0, 2.0, np.nan, 1.0, 2.0, np.nan],  # 40% missing
    })
    csv_tools._datasets["_missing"] = df
    try:
        # Default threshold (0.5): 40% missing is fine — should proceed
        result_default = _run_workflow("_missing", "TARGET", config=GateConfig(max_missing_pct=0.5))
        assert "Aborted" not in result_default.report_markdown

        # Strict threshold (0.3): 40% missing exceeds it — should abort
        result_strict = _run_workflow("_missing", "TARGET", config=GateConfig(max_missing_pct=0.3))
        assert "Aborted" in result_strict.report_markdown
        assert result_strict.summary["total"] == 0  # no steps ran
    finally:
        csv_tools._datasets.pop("_missing", None)


def test_unknown_workflow_returns_error_message():
    result = _run_workflow("fake_ds", "GENHEALT", workflow="nonexistent")
    assert "Unknown workflow" in result.report_markdown
    assert result.steps == []


def test_step_results_carry_duration_and_status():
    result = _run_workflow("fake_ds", "GENHEALT")
    completed = [s for s in result.steps if s.status == "completed"]
    assert completed, "expected at least one completed step"
    for s in completed:
        assert s.duration_ms >= 0
        assert s.output  # non-empty tool output
