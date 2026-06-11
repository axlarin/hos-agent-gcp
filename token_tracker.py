"""
Token usage tracker for Gemini API calls made through the ADK event stream.

Each LLM response event carries usage_metadata with exact token counts.
Records are appended to outputs/token_usage.jsonl so costs accumulate
across sessions and can be reviewed via get_summary() or GET /usage.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_LOG_PATH = Path("./outputs/token_usage.jsonl")

# Gemini pricing per 1M tokens (USD) — https://ai.google.dev/pricing
# thinking_output applies to thoughts_token_count on 2.5 models
_PRICING: Dict[str, Dict[str, float]] = {
    "gemini-2.5-pro": {
        "input": 1.25,
        "output": 10.00,
        "thinking_output": 10.00,
    },
    "gemini-2.5-flash": {
        "input": 0.075,
        "output": 0.30,
        "thinking_output": 3.50,
    },
    "gemini-2.5-flash-lite": {
        "input": 0.015,
        "output": 0.06,
        "thinking_output": 0.06,
    },
}
_DEFAULT_PRICING = _PRICING["gemini-2.5-flash"]


def _pricing_for(model_version: str) -> Dict[str, float]:
    for key, pricing in _PRICING.items():
        if key in (model_version or ""):
            return pricing
    return _DEFAULT_PRICING


def _calc_cost(model_version: str, input_tokens: int, output_tokens: int, thinking_tokens: int) -> float:
    p = _pricing_for(model_version)
    return round(
        input_tokens    / 1_000_000 * p["input"]
        + output_tokens / 1_000_000 * p["output"]
        + thinking_tokens / 1_000_000 * p["thinking_output"],
        8,
    )


def record_event(event: Any, session_id: Optional[str] = None) -> None:
    """Extract usage_metadata from an ADK event and append to the log.

    Safe to call on every event — silently skips events with no usage data.
    """
    usage = getattr(event, "usage_metadata", None)
    if not usage:
        return

    input_tokens    = getattr(usage, "prompt_token_count", 0) or 0
    output_tokens   = getattr(usage, "candidates_token_count", 0) or 0
    thinking_tokens = getattr(usage, "thoughts_token_count", 0) or 0
    total_tokens    = getattr(usage, "total_token_count", 0) or (input_tokens + output_tokens + thinking_tokens)
    model_version   = getattr(event, "model_version", None) or "unknown"
    author          = getattr(event, "author", "unknown")

    cost = _calc_cost(model_version, input_tokens, output_tokens, thinking_tokens)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model_version,
        "agent": author,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "thinking_tokens": thinking_tokens,
        "total_tokens": total_tokens,
        "cost_usd": cost,
        "session_id": session_id,
    }

    _LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    logger.info(
        "[tokens] %s/%s — in=%d out=%d think=%d  $%.6f",
        model_version, author, input_tokens, output_tokens, thinking_tokens, cost,
    )


def get_summary() -> Dict[str, Any]:
    """Return cumulative usage statistics from the log file."""
    if not _LOG_PATH.exists():
        return {"operations": 0, "total_tokens": 0, "total_cost_usd": 0.0, "by_model": {}, "by_agent": {}}

    records: list[Dict[str, Any]] = []
    for line in _LOG_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            records.append(json.loads(line))

    by_model: Dict[str, Any] = {}
    by_agent: Dict[str, Any] = {}

    for r in records:
        for bucket, key in ((by_model, "model"), (by_agent, "agent")):
            k = r[key]
            if k not in bucket:
                bucket[k] = {"operations": 0, "input_tokens": 0, "output_tokens": 0,
                              "thinking_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
            b = bucket[k]
            b["operations"]      += 1
            b["input_tokens"]    += r["input_tokens"]
            b["output_tokens"]   += r["output_tokens"]
            b["thinking_tokens"] += r["thinking_tokens"]
            b["total_tokens"]    += r["total_tokens"]
            b["cost_usd"]         = round(b["cost_usd"] + r["cost_usd"], 8)

    return {
        "operations": len(records),
        "total_tokens": sum(r["total_tokens"] for r in records),
        "total_cost_usd": round(sum(r["cost_usd"] for r in records), 8),
        "by_model": by_model,
        "by_agent": by_agent,
    }
