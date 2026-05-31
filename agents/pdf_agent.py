from google.adk.agents import Agent

from config.settings import settings
from tools.pdf_tools import search_pdf_guidance, get_column_info

_INSTRUCTION = """
You are the HOS PDF specialist.

Use search_pdf_guidance to look up definitions, methodology, and survey design from HOS documents.
Use get_column_info (no column argument) to list all known columns.
Always cite the source document and page/section in your answer.
""".strip()

pdf_agent = Agent(
    name="pdf_agent",
    model=settings.specialist_model,
    description=(
        "Searches HOS PDF documentation for definitions, survey methodology, coded value labels, "
        "and field descriptions. Call this for 'what does X mean?' questions."
    ),
    instruction=_INSTRUCTION,
    tools=[search_pdf_guidance, get_column_info],
)
