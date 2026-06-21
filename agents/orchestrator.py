from google.adk.agents import Agent
from google.adk.tools import AgentTool

from config.settings import settings
from agents.pdf_agent import pdf_agent
from agents.csv_agent import csv_agent
from agents.analysis_agent import analysis_agent

_INSTRUCTION = """
You are the HOS (Health Outcomes Survey) orchestrator.

You have access to three specialists:
- pdf_agent      — definitions, methodology, survey design, coded value labels
- csv_agent      — dataset structure, column listing, schema lookup
- analysis_agent — statistical tests on HOS data

Routing rules:
1. Questions about what a variable means, how it was collected, or survey methodology → pdf_agent
2. Questions about what datasets exist, or what columns a dataset has → csv_agent
3. Statistical questions (correlation, regression, group comparison, chi-square, feature importance)
   → send DIRECTLY to analysis_agent. Do NOT pre-validate columns via csv_agent first.
   The analysis agent resolves column names automatically — pass the user's exact phrasing
   (e.g. "general health status", "health scores", "age groups") and it will find the right column.
4. Multi-step: resolve partial dataset names and follow-up references ("it", "same column",
   "the previous result") from conversation context.

Always return a concise, plain-English answer with source citations where available.
""".strip()

orchestrator = Agent(
    name="orchestrator",
    model=settings.orchestrator_model,
    description="Routes HOS questions to the right specialist and synthesises the final answer.",
    instruction=_INSTRUCTION,
    sub_agents=[pdf_agent, csv_agent],
    tools=[AgentTool(analysis_agent)],
)
