from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd

from config.settings import settings

# Loaded datasets — populated at startup by SchemaBuilder
_datasets: dict[str, pd.DataFrame] = {}


def load_datasets() -> None:
    """Load all CSVs from the configured directory into memory."""
    csv_dir = Path(settings.csv_dir)
    if not csv_dir.exists():
        return
    for path in sorted(csv_dir.glob("*.csv")):
        name = path.stem
        _datasets[name] = pd.read_csv(path, low_memory=False)


def list_datasets() -> str:
    """List all available HOS datasets with their shape (rows × columns).

    Returns:
        A formatted list of dataset names and dimensions.
    """
    if not _datasets:
        return "No datasets loaded. Place CSV files in the configured CSV directory."
    lines = [f"Available datasets ({len(_datasets)} total):"]
    for name, df in sorted(_datasets.items()):
        lines.append(f"  {name}  —  {df.shape[0]:,} rows × {df.shape[1]} columns")
    return "\n".join(lines)


def get_column_info(column: Optional[str] = None) -> str:
    """Return schema information for a specific column.

    When called with a column name, returns the description and coded value labels
    from schema memory. When called without arguments, lists all columns.

    Args:
        column: Column name (case-insensitive). Omit to list all columns.

    Returns:
        Column description and value mappings, or full column listing.
    """
    # Delegates to pdf_tools.get_column_info — same underlying schema cache.
    from tools.pdf_tools import get_column_info as _get
    return _get(column)


def get_dataset(name: str) -> pd.DataFrame:
    """Retrieve a loaded dataset, resolving partial names (e.g. 'c25a' → 'c25a_puf').

    Args:
        name: Dataset name or partial name (case-insensitive).

    Returns:
        The matching DataFrame.

    Raises:
        KeyError: If no matching dataset is found.
    """
    name_lower = name.lower()
    # Exact match
    if name_lower in _datasets:
        return _datasets[name_lower]
    # Partial match
    matches = [k for k in _datasets if k.lower().startswith(name_lower)]
    if len(matches) == 1:
        return _datasets[matches[0]]
    if len(matches) > 1:
        raise KeyError(f"Ambiguous dataset name '{name}': matches {matches}")
    raise KeyError(f"Dataset '{name}' not found. Available: {list(_datasets)}")
