from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

from tools.csv_tools import get_dataset


def _decode_series(series: pd.Series, column: str, schema: dict) -> pd.Series:
    """Replace numeric codes with plain-English labels where available."""
    entry = schema.get(column.upper(), {})
    value_map = entry.get("values", {})
    if value_map:
        return series.map(lambda x: value_map.get(str(int(x)), x) if pd.notna(x) else x)
    return series


def _load_schema() -> dict:
    from rag.schema_builder import SchemaBuilder
    return SchemaBuilder.get_cached_schema()


def run_correlation_analysis(dataset: str, target_column: str, top_n: int = 10) -> str:
    """Run Pearson correlation between target_column and all numeric columns in the dataset.

    Args:
        dataset: Dataset name or partial name (e.g. 'c25a' or 'c25a_puf').
        target_column: Column to correlate against (case-insensitive).
        top_n: Number of top correlated columns to return (default 10).

    Returns:
        Ranked list of columns most correlated with target_column, with labels and r values.
    """
    df = get_dataset(dataset)
    schema = _load_schema()

    target = _find_column(df, target_column)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()

    correlations = {}
    for col in numeric_cols:
        if col == target:
            continue
        try:
            r, p = stats.pearsonr(df[target].dropna(), df[col].dropna().reindex(df[target].dropna().index))
            correlations[col] = (round(r, 4), round(p, 4))
        except Exception:
            continue

    sorted_cols = sorted(correlations, key=lambda c: abs(correlations[c][0]), reverse=True)[:top_n]

    lines = [f"Top {top_n} Pearson correlations with {target_column} in {dataset}:\n"]
    for col in sorted_cols:
        r, p = correlations[col]
        label = schema.get(col.upper(), {}).get("description", col)
        lines.append(f"  {col} ({label}): r={r}, p={p}")
    return "\n".join(lines)


def run_feature_importance(dataset: str, target_column: str, top_n: int = 10) -> str:
    """Run Random Forest feature importance to identify predictors of target_column.

    Args:
        dataset: Dataset name or partial name.
        target_column: Binary or categorical outcome column (case-insensitive).
        top_n: Number of top features to return (default 10).

    Returns:
        Ranked list of feature importances with plain-English labels.
    """
    df = get_dataset(dataset)
    schema = _load_schema()

    target = _find_column(df, target_column)
    feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target]

    sub = df[[target] + feature_cols].dropna()
    X = sub[feature_cols]
    y = sub[target]

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)

    importances = sorted(zip(feature_cols, clf.feature_importances_), key=lambda x: x[1], reverse=True)[:top_n]

    lines = [f"Top {top_n} predictors of {target_column} in {dataset} (Random Forest):\n"]
    for col, imp in importances:
        label = schema.get(col.upper(), {}).get("description", col)
        lines.append(f"  {col} ({label}): importance={round(imp, 4)}")
    return "\n".join(lines)


def run_logistic_regression(
    dataset: str,
    target_column: str,
    feature_columns: list[str],
    positive_values: Optional[list] = None,
) -> str:
    """Run logistic regression predicting target_column from feature_columns.

    Args:
        dataset: Dataset name or partial name.
        target_column: Binary outcome column. If multi-valued, provide positive_values to recode.
        feature_columns: List of predictor column names.
        positive_values: Values of target_column to treat as 1 (positive class).
                         Required when target has more than 2 unique values.

    Returns:
        Coefficients, odds ratios, p-values, and model accuracy.
    """
    df = get_dataset(dataset)
    schema = _load_schema()

    target = _find_column(df, target_column)
    features = [_find_column(df, c) for c in feature_columns]

    sub = df[[target] + features].dropna()
    y = sub[target]

    if positive_values is not None:
        y = y.isin(positive_values).astype(int)
    elif y.nunique() > 2:
        return (
            f"'{target_column}' has {y.nunique()} unique values. "
            "Please provide positive_values to recode it to binary (e.g. positive_values=[1,2])."
        )

    X = sub[features]
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X, y)
    accuracy = round(model.score(X, y), 4)

    lines = [f"Logistic regression — target: {target_column}, dataset: {dataset}\n",
             f"Accuracy: {accuracy}\n",
             "Coefficients:"]
    for col, coef in zip(features, model.coef_[0]):
        label = schema.get(col.upper(), {}).get("description", col)
        or_ = round(float(np.exp(coef)), 4)
        lines.append(f"  {col} ({label}): coef={round(coef, 4)}, OR={or_}")
    return "\n".join(lines)


def run_categorical_analysis(dataset: str, column1: str, column2: Optional[str] = None) -> str:
    """Run frequency table (1 column) or cross-tabulation + chi-square + Cramér's V (2 columns).

    Args:
        dataset: Dataset name or partial name.
        column1: First column — required.
        column2: Second column — if provided, runs crosstab + chi-square + Cramér's V.

    Returns:
        Frequency table or crosstab with chi-square statistics and effect size.
    """
    df = get_dataset(dataset)
    schema = _load_schema()

    col1 = _find_column(df, column1)

    if column2 is None:
        freq = df[col1].value_counts().sort_index()
        entry = schema.get(col1.upper(), {})
        value_map = entry.get("values", {})
        lines = [f"Frequency table for {col1} ({entry.get('description', col1)}) in {dataset}:\n"]
        for val, count in freq.items():
            label = value_map.get(str(int(val)), val) if value_map else val
            pct = round(100 * count / len(df), 1)
            lines.append(f"  {val} ({label}): {count:,} ({pct}%)")
        return "\n".join(lines)

    col2 = _find_column(df, column2)
    ct = pd.crosstab(df[col1], df[col2])
    chi2, p, dof, _ = stats.chi2_contingency(ct)
    n = ct.values.sum()
    cramers_v = round(float(np.sqrt(chi2 / (n * (min(ct.shape) - 1)))), 4)

    lines = [
        f"Cross-tabulation: {col1} × {col2} in {dataset}",
        f"Chi-square={round(chi2, 2)}, p={round(p, 6)}, dof={dof}, Cramér's V={cramers_v}\n",
        ct.to_string(),
    ]
    return "\n".join(lines)


def run_group_comparison(dataset: str, value_column: str, group_column: str) -> str:
    """Compare value_column across groups defined by group_column.

    Automatically selects Mann-Whitney U (2 groups) or Kruskal-Wallis (3+ groups).

    Args:
        dataset: Dataset name or partial name.
        value_column: Continuous or ordinal variable to compare (case-insensitive).
        group_column: Categorical variable that defines groups (case-insensitive).

    Returns:
        Test statistic, p-value, and group medians with decoded labels.
    """
    df = get_dataset(dataset)
    schema = _load_schema()

    val_col = _find_column(df, value_column)
    grp_col = _find_column(df, group_column)

    entry_grp = schema.get(grp_col.upper(), {})
    value_map = entry_grp.get("values", {})

    groups = {}
    for val, group_df in df.groupby(grp_col):
        label = value_map.get(str(int(val)), str(val)) if value_map else str(val)
        data = pd.to_numeric(group_df[val_col], errors="coerce").dropna().to_numpy(dtype=float)
        if len(data) >= 5:
            groups[label] = data

    if len(groups) < 2:
        return f"Not enough groups in '{group_column}' to compare (need ≥ 2 groups with ≥ 5 observations)."

    val_entry = schema.get(val_col.upper(), {})
    lines = [f"Group comparison: {val_col} ({val_entry.get('description', val_col)}) by {grp_col}\n"]
    for label, data in groups.items():
        lines.append(f"  {label}: n={len(data):,}, median={round(float(np.median(data)), 2)}")

    group_arrays = list(groups.values())
    if len(groups) == 2:
        stat, p = stats.mannwhitneyu(*group_arrays, alternative="two-sided")
        lines.append(f"\nMann-Whitney U={round(stat, 2)}, p={round(p, 6)}")
    else:
        stat, p = stats.kruskal(*group_arrays)
        lines.append(f"\nKruskal-Wallis H={round(stat, 2)}, p={round(p, 6)}")

    return "\n".join(lines)


# ── Helper ────────────────────────────────────────────────────────────────────

def _find_column(df: pd.DataFrame, name: str) -> str:
    """Resolve a column name with three-tier fallback:
    1. Exact case-insensitive match on column codes.
    2. Substring match of name within schema descriptions.
    3. Embedding cosine similarity against schema descriptions.
    """
    name_lower = name.lower()

    # Tier 1 — exact case-insensitive
    for col in df.columns:
        if col.lower() == name_lower:
            return col

    # Tiers 2+3 — schema-based resolution
    try:
        schema = _load_schema()
        df_upper = {col.upper(): col for col in df.columns}

        # Only consider columns that are actually in this dataset and have descriptions
        candidates = [
            (df_upper[code], entry["description"])
            for code, entry in schema.items()
            if df_upper.get(code) and entry.get("description")
        ]

        if candidates:
            # Tier 2 — name appears as a substring inside the description
            for actual, desc in candidates:
                if name_lower in desc.lower():
                    logger.info("_find_column: '%s' → '%s' (description substring)", name, actual)
                    return actual

            # Tier 3 — embedding cosine similarity (handles synonyms / paraphrases)
            from rag.embedder import embed
            vecs = embed([name] + [desc for _, desc in candidates])
            query_vec = vecs[0]
            best_sim, best_col = 0.0, None
            for i, (actual, _) in enumerate(candidates):
                # MiniLM vectors are L2-normalised so dot product == cosine similarity
                sim = sum(a * b for a, b in zip(query_vec, vecs[i + 1]))
                if sim > best_sim:
                    best_sim, best_col = sim, actual
            if best_sim >= 0.35 and best_col:
                logger.info(
                    "_find_column: '%s' → '%s' (embedding sim=%.3f)", name, best_col, best_sim
                )
                return best_col
    except Exception as exc:
        logger.debug("_find_column fallback error: %s", exc)

    # Build a hint list: column_code (short description) for up to 20 schema-known columns
    try:
        schema = _load_schema()
        hints = [
            f"{col} ({schema.get(col.upper(), {}).get('description', '')[:60]})"
            for col in df.columns
            if schema.get(col.upper(), {}).get("description")
        ][:20]
    except Exception:
        hints = list(df.columns[:20])
    raise KeyError(
        f"Column '{name}' not found. Pass the user's exact phrasing or use one of these "
        f"column codes: {hints}"
    )
