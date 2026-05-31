"""Smoke tests for orchestrator configuration."""
import pytest


def test_orchestrator_has_sub_agents():
    from agents.orchestrator import orchestrator
    assert len(orchestrator.sub_agents) == 2


def test_orchestrator_sub_agents_are_pdf_and_csv():
    from agents.orchestrator import orchestrator
    names = [a.name for a in orchestrator.sub_agents]
    assert "pdf_agent" in names
    assert "csv_agent" in names


def test_orchestrator_has_analysis_agent_tool():
    from agents.orchestrator import orchestrator
    from google.adk.tools import AgentTool
    agent_tools = [t for t in orchestrator.tools if isinstance(t, AgentTool)]
    assert len(agent_tools) == 1
    assert agent_tools[0].agent.name == "analysis_agent"
