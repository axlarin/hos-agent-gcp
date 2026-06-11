from __future__ import annotations

import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

SCHEMA_VERSION = "2.0"
# Fail-fast: raise if a PDF yields fewer than this % of expected fields.
MIN_COVERAGE_PCT = 70


# ── Format-specific parsers ───────────────────────────────────────────────────

class _FormatParser(ABC):
    """Base class for HOS data-dictionary table parsers.

    Each subclass handles one distinct table layout found across HOS codebooks.
    Adding support for a new layout means adding a subclass and updating
    _detect_parser() — no changes to SchemaBuilder required.
    """

    @abstractmethod
    def clean(self, body: str) -> str:
        """Strip repeating table headers and section sub-headers from extracted text."""

    @abstractmethod
    def anchor(self, n: int) -> str:
        """Regex that matches the start of field N's header line (not a value-code line)."""

    @abstractmethod
    def extract_description(self, block: str, field_num: int) -> str:
        """Extract a clean plain-English description from a field block."""


class _FormatAParser(_FormatParser):
    """c25a analytic PUF: 4-column table, 'Cohort N Baseline/Follow Up' field headers.

    Field headers ALWAYS start with 'Cohort N' or 'Unique Identifier', which lets
    us distinguish them from character-position lines whose numbers coincide with
    later field numbers (e.g. position line '79 In the past 12 months...' vs
    field-79 header '79 Cohort 25 Follow Up Survey: General Health...').
    """

    def clean(self, body: str) -> str:
        body = re.sub(
            r'Field # Field Description\s*\nField\s*\nPosition Valid Values and Notes\s*\n',
            '', body,
        )
        # Remove section sub-headers but NOT field-entry headers that contain
        # 'Follow Up Survey:' — section headers end with Questions/Administration,
        # field headers end with a colon after the question label.
        body = re.sub(
            r'(?:Cohort \d+ (?:Baseline|Follow Up) (?:Survey )?(?:Questions?|Administration)|'
            r'Identification and[^\n]+)\n',
            '', body, flags=re.IGNORECASE,
        )
        return body

    def anchor(self, n: int) -> str:
        return rf'\n{n} (?:Cohort \d+|Unique Identifier)'

    def extract_description(self, block: str, field_num: int) -> str:
        lines = block.split('\n')
        parts: List[str] = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if re.match(r'^\d+\s*=', s):
                continue
            if re.match(r'^(NOTE|Source|Example|Go to Q)\s*[:.]?', s, re.IGNORECASE):
                continue
            opening = re.match(rf'^{field_num}\s+(.*)', s)
            if opening:
                remainder = opening.group(1).strip()
                if remainder:
                    parts.append(remainder)
                continue
            if re.match(r'^\d+(-\d+)?\s*$', s):
                continue
            pos_with_text = re.match(r'^\d+(?:-\d+)?\s+(.+)', s)
            if pos_with_text:
                text_part = pos_with_text.group(1).strip()
                if not re.match(r'^\d+\s*=', text_part):
                    parts.append(text_part)
                continue
            parts.append(s)
        description = ' '.join(parts)
        description = re.sub(
            r'Cohort \d+ (?:Baseline|Follow Up) Survey:\s*', '', description, flags=re.IGNORECASE
        )
        description = re.sub(
            r'Cohort \d+ (?:Baseline|Follow Up):\s*', '', description, flags=re.IGNORECASE
        )
        description = re.sub(r'Cohort \d+ Analytic:\s*', '', description, flags=re.IGNORECASE)
        return description.strip()


class _FormatBParser(_FormatParser):
    """c26b/c27b baseline PUF: 5-column table, Num/Char type markers, no character positions.

    Safe to use the simpler '\\nN (?!=)' anchor since there are no character-position
    lines to cause false matches.
    """

    def clean(self, body: str) -> str:
        body = re.sub(
            r'Field # Field[/\s]*Name\s*\nDescription\s*\nField\s*\nType\s*\nField\s*\nLength Valid Values and Notes\s*\n',
            '', body,
        )
        body = re.sub(
            r'Field # Field Description\s*\nField\s*\nType\s*\nField\s*\nLength Valid Values and Notes\s*\n',
            '', body,
        )
        body = re.sub(r'(?:Identification and[^\n]+|.+? Table)\n', '', body)
        return body

    def anchor(self, n: int) -> str:
        return rf'\n{n} (?!=)'

    def extract_description(self, block: str, field_num: int) -> str:
        lines = block.split('\n')
        parts: List[str] = []
        for line in lines:
            s = line.strip()
            if not s:
                continue
            if re.match(r'^\d+\s*=', s):
                continue
            if re.match(r'^(NOTE|Source|Example|Go to Q)\s*[:.]?', s, re.IGNORECASE):
                continue
            opening = re.match(rf'^{field_num}\s+(.*)', s)
            if opening:
                remainder = opening.group(1).strip()
                # Strip 'Num 8' / 'Char 9' type-length marker that pypdf puts on the opening line.
                remainder = re.sub(r'\s+(?:Num|Char)\s+\d+\s*', ' ', remainder).strip()
                if remainder:
                    parts.append(remainder)
                continue
            if re.match(r'^(?:Num|Char)\s+\d+\s*$', s, re.IGNORECASE):
                continue
            num_char_with_text = re.match(r'^(?:Num|Char)\s+\d+\s+(.*)', s, re.IGNORECASE)
            if num_char_with_text:
                remainder = num_char_with_text.group(1).strip()
                if remainder and not re.match(r'^\d+\s*=', remainder):
                    parts.append(remainder)
                continue
            parts.append(s)
        return ' '.join(parts).strip()


def _detect_parser(body: str) -> _FormatParser:
    """Return the correct parser based on whether 'Cohort N' prefixes are present."""
    if re.search(r'\n\d+ Cohort \d+', body):
        return _FormatAParser()
    return _FormatBParser()


def _extract_values(block: str) -> Dict[str, str]:
    """Extract 'N = Label' coded-value lines from a field block (shared by all formats).

    Handles two positions where values appear:
    1. Own line:  '\\n1 = English'         (the common case)
    2. Embedded:  '\\n98 1 = English'       (first value on the position line)
    """
    values: Dict[str, str] = {}
    for m in re.finditer(r'\n(\d+)\s*=\s*(.+)', block):
        code, label = m.group(1), ' '.join(m.group(2).split())
        if label:
            values[code] = label
    for m in re.finditer(r'\n\d+\s+(\d+)\s*=\s*(.+)', block):
        code, label = m.group(1), ' '.join(m.group(2).split())
        if label and code not in values:
            values[code] = label
    return values


# ── SchemaBuilder ─────────────────────────────────────────────────────────────

class SchemaBuilder:
    """Builds and caches the HOS column schema from PDF data dictionaries.

    Two-pass pipeline:
      Pass 1 (authoritative): Each HOS codebook PDF is parsed by a format-specific
        parser. Coverage is validated against a MIN_COVERAGE_PCT threshold; a parse
        that falls below threshold raises immediately rather than silently returning
        a partial schema.
      Pass 2 (fallback): Gemini decodes any CSV columns not documented in the PDFs,
        marked confidence='medium' to distinguish them from PDF-derived entries.

    Schema shape per column (stored in schema_memory.json under 'columns'):
        {
            "COLUMN_NAME": {
                "description": "Plain-English survey question or field label",
                "value_labels": {"1": "Yes", "2": "No", ...},
                "source": "PDF data dictionary" | "LLM",
                "confidence": "high" | "medium"
            }
        }

    The file also carries a '_meta' block with:
        - schema_version: bumped when the output format changes
        - built_at: ISO-8601 UTC timestamp
        - coverage: per-cohort found/expected/pct metrics
        - pdf_hashes / csv_hashes: MD5s of input files; cache auto-invalidates
          when any input changes, so callers never need to pass force=True after
          updating data files.
    """

    _cache: Dict[str, Any] = {}   # columns dict; same shape as before
    _meta: Dict[str, Any] = {}    # build metadata (version, hashes, coverage)

    def __init__(self, settings, vector_store) -> None:
        self._settings = settings
        self._vector_store = vector_store
        self._is_ready = False

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    async def build_or_load(self, force: bool = False) -> None:
        """Build schema from PDFs + Gemini, or load from validated cache.

        Rebuilds automatically when PDF or CSV hashes change, even without force=True.
        """
        cache_path = Path(self._settings.schema_cache_path)
        await self._load_csv_datasets()

        manifest = self._build_file_manifest()

        if not force and self._cache_is_valid(cache_path, manifest):
            logger.info("Loading schema from cache: %s", cache_path)
            data = json.loads(cache_path.read_text())
            SchemaBuilder._cache = data["columns"]
            SchemaBuilder._meta = data.get("_meta", {})

            import tools.pdf_tools as pt
            pt._inject(self._vector_store, self)

            self._is_ready = True
            return

        if not force and cache_path.exists():
            logger.info("Cache stale (input hashes changed) — rebuilding …")
        else:
            logger.info("Building schema …")

        schema: Dict[str, Any] = {}
        coverage: Dict[str, Any] = {}

        pdf_dir = Path(self._settings.pdf_dir)
        if pdf_dir.exists():
            schema, coverage = await self._pass1_pdf_parsing(pdf_dir)

        from tools.csv_tools import _datasets
        remaining = self._find_undocumented_columns(_datasets, schema)
        if remaining:
            logger.info("Pass 2: Gemini fallback for %d undocumented columns", len(remaining))
            schema.update(await self._pass2_gemini_fallback(remaining))

        meta = {
            "schema_version": SCHEMA_VERSION,
            "built_at": datetime.now(timezone.utc).isoformat(),
            "coverage": coverage,
            **manifest,
        }
        data = {"_meta": meta, "columns": schema}

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(data, indent=2))
        SchemaBuilder._cache = schema
        SchemaBuilder._meta = meta

        import tools.pdf_tools as pt
        pt._inject(self._vector_store, self)

        self._is_ready = True
        n_pdf = sum(1 for e in schema.values() if e["source"] == "PDF data dictionary")
        n_llm = sum(1 for e in schema.values() if e["source"] == "LLM")
        logger.info("Schema ready — %d columns (%d PDF, %d LLM)", len(schema), n_pdf, n_llm)

    def list_all_columns(self) -> str:
        """Return a formatted listing of all documented columns with coverage summary."""
        if not SchemaBuilder._cache:
            return "Schema not yet built."
        lines = [f"Documented columns ({len(SchemaBuilder._cache)} total):\n"]
        for col, entry in sorted(SchemaBuilder._cache.items()):
            desc = entry.get("description", "")
            src = entry.get("source", "?")
            conf = entry.get("confidence", "")
            tag = f"{src}" + (f", {conf}" if conf else "")
            lines.append(f"  {col} [{tag}]: {desc}")
        if SchemaBuilder._meta.get("coverage"):
            lines.append("\nCoverage by cohort:")
            for cohort, cov in SchemaBuilder._meta["coverage"].items():
                lines.append(
                    f"  {cohort}: {cov['found']}/{cov['expected']} ({cov['pct']:.1f}%)"
                )
        return "\n".join(lines)

    def get_column_info(self, column: str) -> str:
        """Return description, value labels, source, and confidence for a column."""
        key = column.upper()
        entry = SchemaBuilder._cache.get(key)
        if entry is None:
            return f"Column '{column}' not found in schema memory."
        desc = entry.get("description", "No description")
        values = entry.get("value_labels", {})
        source = entry.get("source", "unknown")
        confidence = entry.get("confidence", "")
        tag = source + (f", confidence={confidence}" if confidence else "")
        lines = [f"{column} [{tag}]: {desc}"]
        if values:
            lines.append("Value labels:")
            for code, label in sorted(
                values.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0
            ):
                lines.append(f"  {code} = {label}")
        return "\n".join(lines)

    @classmethod
    def get_cached_schema(cls) -> Dict[str, Any]:
        return cls._cache

    @classmethod
    def get_meta(cls) -> Dict[str, Any]:
        """Return build metadata: version, timestamp, coverage, and file hashes."""
        return cls._meta

    # ── Private ───────────────────────────────────────────────────────────────

    def _build_file_manifest(self) -> Dict[str, Any]:
        """Compute MD5 hashes for all current PDFs and CSVs."""
        pdf_hashes = {
            p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(Path(self._settings.pdf_dir).glob("*.pdf"))
        }
        csv_hashes = {
            p.name: hashlib.md5(p.read_bytes()).hexdigest()
            for p in sorted(Path(self._settings.csv_dir).glob("*.csv"))
        }
        return {"pdf_hashes": pdf_hashes, "csv_hashes": csv_hashes}

    def _cache_is_valid(self, cache_path: Path, manifest: Dict[str, Any]) -> bool:
        """Return True only if cache exists, schema_version matches, and all file hashes match."""
        if not cache_path.exists():
            return False
        try:
            data = json.loads(cache_path.read_text())
            meta = data.get("_meta", {})
        except (json.JSONDecodeError, OSError):
            return False
        return (
            meta.get("schema_version") == SCHEMA_VERSION
            and meta.get("pdf_hashes") == manifest["pdf_hashes"]
            and meta.get("csv_hashes") == manifest["csv_hashes"]
        )

    async def _pass1_pdf_parsing(
        self, pdf_dir: Path
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Parse all HOS codebook PDFs; return (schema, coverage_by_cohort)."""
        import pypdf

        schema: Dict[str, Any] = {}
        coverage: Dict[str, Any] = {}

        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            cohort_id = self._cohort_id_from_filename(pdf_path.name)
            csv_columns = self._get_csv_columns_for_cohort(cohort_id)
            if not csv_columns:
                logger.warning(
                    "No CSV found for cohort '%s' (%s) — skipping PDF", cohort_id, pdf_path.name
                )
                continue

            reader = pypdf.PdfReader(str(pdf_path))
            full_text = "\n".join(page.extract_text() or "" for page in reader.pages)

            parsed, pct = self._parse_and_validate(full_text, csv_columns, pdf_path.name)
            coverage[cohort_id] = {
                "found": len(parsed),
                "expected": len(csv_columns),
                "pct": round(pct, 1),
            }

            # First PDF wins — later cohorts don't overwrite an already-documented column.
            for col, entry in parsed.items():
                if col not in schema:
                    schema[col] = entry

        return schema, coverage

    def _parse_and_validate(
        self, text: str, csv_columns: List[str], source_name: str
    ) -> Tuple[Dict[str, Any], float]:
        """Parse one PDF's field index table and fail-fast if coverage is below threshold.

        The coverage check catches the highest-risk assumption in Pass 1: that PDF
        field numbers align with CSV column order. A large mismatch (< MIN_COVERAGE_PCT)
        means either the PDF format changed or the wrong CSV was matched to this PDF.
        """
        parsed = self._parse_field_index_table(text, csv_columns, source_name)
        found = len(parsed)
        expected = len(csv_columns)
        pct = 100.0 * found / expected if expected else 0.0

        logger.info(
            "  %s: parsed %d / %d fields (%.1f%%)", source_name, found, expected, pct
        )

        if pct < MIN_COVERAGE_PCT:
            raise ValueError(
                f"{source_name}: only {pct:.1f}% of fields parsed ({found}/{expected}) — "
                f"below the {MIN_COVERAGE_PCT}% threshold. "
                f"Verify the PDF table format and that the correct CSV is matched to this cohort."
            )
        return parsed, pct

    def _cohort_id_from_filename(self, name: str) -> str:
        """Extract cohort token from filename, e.g. 'hos_dug_puf_c25a.pdf' → 'c25a'."""
        m = re.search(r'_(c\d+\w+)\.pdf$', name, re.IGNORECASE)
        return m.group(1).lower() if m else ""

    def _get_csv_columns_for_cohort(self, cohort_id: str) -> List[str]:
        """Return ordered column list from the CSV whose name contains cohort_id."""
        from tools.csv_tools import _datasets
        for dataset_name, df in _datasets.items():
            if cohort_id in dataset_name.lower():
                return list(df.columns)
        return []

    def _parse_field_index_table(
        self, text: str, csv_columns: List[str], source_name: str = ""
    ) -> Dict[str, Any]:
        """Locate the field index section, detect the table format, and extract all fields.

        Delegates format-specific cleaning, anchoring, and description extraction to the
        appropriate _FormatParser subclass. The field-block loop itself is format-agnostic.
        """
        schema: Dict[str, Any] = {}

        # Prefer the occurrence directly followed by the introductory paragraph
        # (the real section), not the TOC entry.
        section_m = re.search(
            r'Field Index with Field Descriptions\s*\nThe structure and fields', text
        )
        if not section_m:
            for section_m in re.finditer(r'Field Index with Field Descriptions', text):
                pass
        if not section_m:
            logger.warning(
                "'Field Index with Field Descriptions' not found in %s", source_name
            )
            return schema

        body = text[section_m.start():]
        body = re.sub(r'Medicare HOS [^\n]+\n', '', body)
        body = re.sub(r'Prepared by Health Services Advisory Group[^\n]+\n', '', body)

        parser = _detect_parser(body)
        body = parser.clean(body)

        # Sequential scan keeps search_from advancing so earlier text positions
        # can never match a later field number (critical for Format A).
        field_starts: Dict[int, int] = {}
        search_from = 0
        for n in range(1, len(csv_columns) + 1):
            m = re.search(parser.anchor(n), body[search_from:])
            if m:
                abs_pos = search_from + m.start()
                field_starts[n] = abs_pos
                search_from = abs_pos + 1

        sorted_ns = sorted(field_starts)
        for idx, n in enumerate(sorted_ns):
            start = field_starts[n]
            end = (
                field_starts[sorted_ns[idx + 1]] if idx + 1 < len(sorted_ns) else len(body)
            )
            block = body[start:end]
            col_name = csv_columns[n - 1].upper()
            values = _extract_values(block)
            description = parser.extract_description(block, n)
            if description or values:
                schema[col_name] = {
                    "description": description or col_name,
                    "value_labels": values,
                    "source": "PDF data dictionary",
                    "confidence": "high",
                }

        return schema

    async def _pass2_gemini_fallback(self, columns: List[str]) -> Dict[str, Any]:
        """Use Gemini to decode remaining admin/derived columns (confidence='medium').

        Batches 40 columns per API call with sample values drawn from loaded DataFrames.
        """
        import google.generativeai as genai
        from tools.csv_tools import _datasets

        genai.configure(api_key=self._settings.google_api_key)
        model = genai.GenerativeModel(self._settings.evaluator_model)

        col_samples: Dict[str, List[str]] = {}
        for col in columns:
            col_upper = col.upper()
            for df in _datasets.values():
                col_match = next((c for c in df.columns if c.upper() == col_upper), None)
                if col_match is not None:
                    unique_vals = df[col_match].dropna().unique()[:15]
                    col_samples[col] = sorted(str(v) for v in unique_vals)
                    break

        schema: Dict[str, Any] = {}
        for i in range(0, len(columns), 40):
            schema.update(await self._gemini_batch(model, columns[i : i + 40], col_samples))
        return schema

    async def _gemini_batch(
        self,
        model: Any,
        columns: List[str],
        col_samples: Dict[str, List[str]],
    ) -> Dict[str, Any]:
        """Send one batch of columns to Gemini and parse the JSON response."""
        col_list = "\n".join(
            f"- {col}: sample values = {col_samples.get(col, ['(no data)'])}"
            for col in columns
        )
        prompt = f"""You are a domain expert on the Medicare Health Outcomes Survey (HOS) Public Use File.

The column names follow the pattern: B{{cohort}}{{question_code}} for baseline fields,
F{{cohort}}{{question_code}} for follow-up fields, and short admin names for
administrative/derived fields (e.g. CASE_ID, COHORT, SFLAG, SAMPLED).

For each field below, provide:
1. A plain-English description (what this field measures or represents).
2. If the sample values look like a categorical code (small integer set), provide the
   value-to-label mapping. Otherwise return an empty object for "values".

Fields:
{col_list}

Return ONLY a valid JSON object. No markdown, no explanation. Use this exact structure:
{{
  "COLUMN_NAME": {{
    "description": "...",
    "values": {{"1": "Label1", "2": "Label2"}}
  }},
  ...
}}
"""
        try:
            response = await model.generate_content_async(prompt)
            raw = response.text
        except Exception as exc:
            logger.warning("Gemini Pass 2 API error: %s", exc)
            return {}

        json_m = re.search(r'\{[\s\S]+\}', raw)
        if not json_m:
            logger.warning(
                "Gemini Pass 2 returned no parseable JSON for batch starting at '%s'",
                columns[0],
            )
            return {}

        try:
            parsed = json.loads(json_m.group())
        except json.JSONDecodeError as exc:
            logger.warning("Gemini Pass 2 JSON parse error: %s", exc)
            return {}

        schema: Dict[str, Any] = {}
        col_upper_set = {c.upper() for c in columns}
        for col_key, entry in parsed.items():
            key = col_key.upper()
            if key not in col_upper_set:
                continue
            schema[key] = {
                "description": str(entry.get("description", key)),
                "value_labels": {
                    str(k): str(v) for k, v in entry.get("values", {}).items()
                },
                "source": "LLM",
                "confidence": "medium",
            }
        return schema

    def _find_undocumented_columns(self, datasets: dict, schema: dict) -> List[str]:
        undocumented: set = set()
        for df in datasets.values():
            for col in df.columns:
                if col.upper() not in schema:
                    undocumented.add(col.upper())
        return sorted(undocumented)

    async def _load_csv_datasets(self) -> None:
        from tools.csv_tools import load_datasets
        load_datasets()
