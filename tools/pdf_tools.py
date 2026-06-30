from __future__ import annotations

import re
from typing import Optional

# VectorStore and SchemaBuilder are injected at startup via module-level singletons
# to avoid circular imports and to allow tools to be defined before the RAG layer loads.
_vector_store = None
_schema_builder = None

# Short acronym queries (e.g. "What does PCS mean?") embed poorly against the
# verbose PDF passages that define them. Expanding known acronyms before
# embedding the search query steers retrieval toward the right chunk.
_ACRONYM_EXPANSIONS = {
    "pcs": "Physical Component Summary (PCS)",
    "mcs": "Mental Component Summary (MCS)",
}


def _inject(vector_store, schema_builder) -> None:
    global _vector_store, _schema_builder
    _vector_store = vector_store
    _schema_builder = schema_builder


def _expand_acronyms(query: str) -> str:
    expanded = query
    for acronym, expansion in _ACRONYM_EXPANSIONS.items():
        if re.search(rf"\b{acronym}\b", query, re.IGNORECASE):
            expanded = f"{expanded} {expansion}"
    return expanded


def search_pdf_guidance(query: str) -> str:
    """Search HOS PDF documentation for definitions, methodology, and survey design.

    Args:
        query: The search query in plain English.

    Returns:
        Relevant passages from HOS documents with source citations (document name + section).
    """
    if _vector_store is None:
        raise RuntimeError("VectorStore not injected — call tools.pdf_tools._inject() at startup")
    results = _vector_store.search(_expand_acronyms(query), n_results=5)
    if not results:
        return "No relevant documentation found."
    parts = []
    for r in results:
        source = r.get("source", "unknown")
        text = r.get("text", "")
        parts.append(f"[{source}]\n{text}")
    return "\n\n---\n\n".join(parts)


def get_column_info(column: Optional[str] = None) -> str:
    """Return column information from schema memory.

    When called with no column argument, lists all known column names across all datasets.
    When called with a column name, returns the full description and coded value labels for
    that column.

    Args:
        column: Column name (case-insensitive). Omit to list all columns.

    Returns:
        Column description and value mappings, or a list of all column names.
    """
    if _schema_builder is None:
        raise RuntimeError("SchemaBuilder not injected — call tools.pdf_tools._inject() at startup")
    if column is None:
        return _schema_builder.list_all_columns()
    return _schema_builder.get_column_info(column)
