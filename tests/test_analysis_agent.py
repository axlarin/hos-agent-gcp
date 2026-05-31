"""Smoke tests for analysis_agent configuration."""
import pytest


def test_analysis_agent_has_all_tools():
    from agents.analysis_agent import analysis_agent
    tool_names = [t.__name__ if callable(t) else str(t) for t in analysis_agent.tools]
    expected = [
        "run_correlation_analysis",
        "run_feature_importance",
        "run_logistic_regression",
        "run_categorical_analysis",
        "run_group_comparison",
    ]
    for name in expected:
        assert name in tool_names, f"Missing tool: {name}"


def test_analysis_agent_description_mentions_statistics():
    from agents.analysis_agent import analysis_agent
    desc = analysis_agent.description.lower()
    assert "statistic" in desc or "regression" in desc or "correlation" in desc
