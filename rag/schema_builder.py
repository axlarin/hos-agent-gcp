from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class SchemaBuilder:
    """Builds and caches the column schema from the HOS PDF data dictionary.

    Pass 1 (authoritative): Parses the data dictionary PDF by field position.
    Pass 2 (fallback): Gemini decodes remaining admin/derived columns.

    Schema shape per column (stored in schema_memory.json):
        {
            "COLUMN_NAME": {
                "description": "Plain-English survey question or field label",
                "values": {"1": "Yes", "2": "No", ...},
                "source": "pdf" | "llm"
            },
            ...
        }
    """

    _cache: Dict[str, Any] = {}

    def __init__(self, settings, vector_store) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    async def build_or_load(self, force: bool = False) -> None:
        """Build schema from PDFs + Gemini, or load from cache.

        Args:
            force: If True, rebuilds even if cache exists.
        """
        cache_path = Path(self._settings.schema_cache_path)

        if not force and cache_path.exists():
            logger.info("Loading schema from cache: %s", cache_path)
            SchemaBuilder._cache = json.loads(cache_path.read_text())
            await self._load_csv_datasets()
            self._is_ready = True
            return

        logger.info("Building schema …")
        schema: Dict[str, Any] = {}

        pdf_dir = Path(self._settings.pdf_dir)
        if pdf_dir.exists():
            schema.update(await self._pass1_pdf_parsing(pdf_dir))

        await self._load_csv_datasets()
        from tools.csv_tools import _datasets
        remaining = self._find_undocumented_columns(_datasets, schema)
        if remaining:
            logger.info("Pass 2: Gemini fallback for %d undocumented columns", len(remaining))
            schema.update(await self._pass2_gemini_fallback(remaining))

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(schema, indent=2))
        SchemaBuilder._cache = schema

        # Inject schema into tool layer
        import tools.pdf_tools as pt
        pt._inject(self._vector_store, self)

        self._is_ready = True
        logger.info("Schema ready — %d columns documented", len(schema))

    def list_all_columns(self) -> str:
        """Return a formatted listing of all documented columns."""
        if not SchemaBuilder._cache:
            return "Schema not yet built."
        lines = [f"Documented columns ({len(SchemaBuilder._cache)} total):\n"]
        for col, entry in sorted(SchemaBuilder._cache.items()):
            desc = entry.get("description", "")
            src = entry.get("source", "?")
            lines.append(f"  {col} [{src}]: {desc}")
        return "\n".join(lines)

    def get_column_info(self, column: str) -> str:
        """Return description and value labels for a column."""
        key = column.upper()
        entry = SchemaBuilder._cache.get(key)
        if entry is None:
            return f"Column '{column}' not found in schema memory."
        desc = entry.get("description", "No description")
        values = entry.get("values", {})
        source = entry.get("source", "unknown")
        lines = [f"{column} [{source}]: {desc}"]
        if values:
            lines.append("Values:")
            for code, label in sorted(values.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0):
                lines.append(f"  {code} = {label}")
        return "\n".join(lines)

    @classmethod
    def get_cached_schema(cls) -> Dict[str, Any]:
        return cls._cache

    # ── Private ───────────────────────────────────────────────────────────────

    async def _pass1_pdf_parsing(self, pdf_dir: Path) -> Dict[str, Any]:
        """Parse HOS data dictionary PDF directly by field position."""
        raise NotImplementedError("PDF data dictionary parser — implement in rag/schema_builder.py")

    async def _pass2_gemini_fallback(self, columns: list[str]) -> Dict[str, Any]:
        """Use Gemini to decode remaining admin/derived columns."""
        raise NotImplementedError("Gemini fallback decoder — implement in rag/schema_builder.py")

    def _find_undocumented_columns(self, datasets: dict, schema: dict) -> list[str]:
        undocumented = set()
        for df in datasets.values():
            for col in df.columns:
                if col.upper() not in schema:
                    undocumented.add(col.upper())
        return sorted(undocumented)

    async def _load_csv_datasets(self) -> None:
        from tools.csv_tools import load_datasets
        load_datasets()
