from google.adk.agents import Agent

from config.settings import settings
from tools.csv_tools import list_datasets, get_column_info

_INSTRUCTION = """
You are the HOS CSV dataset specialist.

Use list_datasets to show available datasets when the user has not named one.
Use get_column_info with a column name to return its description and coded value labels.
Always confirm the exact dataset and column name before handing off to analysis_agent.
""".strip()

csv_agent = Agent(
    name="csv_agent",
    model=settings.specialist_model,
    description=(
        "Inspects HOS dataset structure and schema memory. Lists datasets and columns. "
        "Returns column descriptions and value codes from schema cache. "
        "Call this before running any statistical analysis."
    ),
    instruction=_INSTRUCTION,
    tools=[list_datasets, get_column_info],
)
