# ── File references ───────────────────────────────────────────────────────────
# config/settings.py      — provides settings.specialist_model (gemini-2.5-flash)
# tools/pdf_tools.py      — search_pdf_guidance (RAG search over ChromaDB)
#                           get_column_info (schema lookup from schema_memory.json)
# agents/orchestrator.py  — registers pdf_agent as a sub_agent; the orchestrator
#                           delegates "what does X mean?" questions here automatically
# rag/vector_store.py     — ChromaDB store that search_pdf_guidance queries at runtime
# rag/schema_builder.py   — builds/loads the schema cache that get_column_info reads
# ─────────────────────────────────────────────────────────────────────────────

from google.adk.agents import Agent

from config.settings import settings
from tools.pdf_tools import search_pdf_guidance, get_column_info

# System instruction sent to Gemini with every conversation turn.
# Keeps the agent focused on PDF content only and enforces source citation.
_INSTRUCTION = """
You are the HOS PDF specialist.

Use search_pdf_guidance to look up definitions, methodology, and survey design from HOS documents.
Use get_column_info (no column argument) to list all known columns.
Always cite the source document and page/section in your answer.
""".strip()

# pdf_agent is a sub_agent of the orchestrator (see agents/orchestrator.py).
# ADK routes questions about HOS definitions, field descriptions, survey methodology,
# and coded value labels here. It does NOT touch CSV data — that goes to csv_agent.
pdf_agent = Agent(
    name="pdf_agent",
    # gemini-2.5-flash — set via SPECIALIST_MODEL in .env / config/settings.py
    model=settings.specialist_model,
    # The description is what the orchestrator reads to decide whether to delegate
    # a question to this agent; keep it precise so routing is accurate.
    description=(
        "Searches HOS PDF documentation for definitions, survey methodology, coded value labels, "
        "and field descriptions. Call this for 'what does X mean?' questions."
    ),
    instruction=_INSTRUCTION,
    # search_pdf_guidance  — semantic RAG search (tools/pdf_tools.py → rag/vector_store.py)
    # get_column_info      — schema lookup by column name (tools/pdf_tools.py → rag/schema_builder.py)
    tools=[search_pdf_guidance, get_column_info],
)
