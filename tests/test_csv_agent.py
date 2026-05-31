"""Smoke tests for csv_agent routing and tool calls."""
import pytest


def test_csv_agent_has_correct_tools():
    from agents.csv_agent import csv_agent
    tool_names = [t.__name__ if callable(t) else str(t) for t in csv_agent.tools]
    assert "list_datasets" in tool_names
    assert "get_column_info" in tool_names


def test_csv_agent_description_contains_routing_keywords():
    from agents.csv_agent import csv_agent
    desc = csv_agent.description.lower()
    assert "dataset" in desc or "column" in desc or "schema" in desc
