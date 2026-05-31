"""Smoke tests for pdf_agent routing and tool calls."""
import pytest


def test_pdf_agent_has_correct_tools():
    from agents.pdf_agent import pdf_agent
    tool_names = [t.__name__ if callable(t) else str(t) for t in pdf_agent.tools]
    assert "search_pdf_guidance" in tool_names
    assert "get_column_info" in tool_names


def test_pdf_agent_description_contains_routing_keywords():
    from agents.pdf_agent import pdf_agent
    desc = pdf_agent.description.lower()
    assert "definition" in desc or "pdf" in desc or "documentation" in desc
