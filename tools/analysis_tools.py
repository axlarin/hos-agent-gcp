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


def _resolve_subset(
    df: pd.DataFrame,
    subset_column: Optional[str],
    subset_codes: Optional[list],
) -> tuple[pd.DataFrame, str]:
    """Filter df to rows where subset_column value is in subset_codes.

    Returns (filtered_df, filter_description). filter_description is an empty string
    when no filter is applied. Uses _find_column so natural-language column names work.
    subset_codes must be the exact numeric codes from the column's value_labels.
    """
    if not subset_column or not subset_codes:
        return df, ""

    col = _find_column(df, subset_column)
    codes = [int(c) for c in subset_codes]
    col_numeric = pd.to_numeric(df[col], errors="coerce")
    filtered = df[col_numeric.isin(codes)]

    if filtered.empty:
        raise ValueError(
            f"subset_column='{subset_column}' (resolved: '{col}') with subset_codes={codes} "
            "matched no rows. Check valid codes with run_categorical_analysis first."
        )

    desc = f"[Filtered: {col} in {codes} — {len(filtered):,} rows]"
    return filtered, desc


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


def run_correlation_analysis(
    dataset: str,
    target_column: str,
    top_n: int = 10,
    subset_column: Optional[str] = None,
    subset_codes: Optional[list] = None,
) -> str:
    """Run Pearson correlation between target_column and all numeric columns in the dataset.

    Args:
        dataset: Dataset name or partial name (e.g. 'c25a' or 'c25a_puf').
        target_column: Column to correlate against (case-insensitive).
        top_n: Number of top correlated columns to return (default 10).
        subset_column: Optional column to filter on (e.g. "age group" or "AGE").
        subset_codes: Exact numeric codes to keep (e.g. [2, 3] for AGE 65+).
                      Look up valid codes with run_categorical_analysis first.

    Returns:
        Ranked list of columns most correlated with target_column, with labels and r values.
    """
    df = get_dataset(dataset)
    df, filter_desc = _resolve_subset(df, subset_column, subset_codes)
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

    header = f"Top {top_n} Pearson correlations with {target_column} in {dataset}"
    if filter_desc:
        header += f" {filter_desc}"
    lines = [header + ":\n"]
    for col in sorted_cols:
        r, p = correlations[col]
        label = schema.get(col.upper(), {}).get("description", col)
        lines.append(f"  {col} ({label}): r={r}, p={p}")
    return "\n".join(lines)


def _detect_conditional_followups(
    df: pd.DataFrame,
    target_col: str,
    feature_cols: list[str],
    miss_diff_threshold: float = 0.5,
    corr_threshold: float = 0.6,
) -> tuple[list[str], list[str]]:
    """Identify feature columns that are conditional follow-up questions to target_col.

    Two detection strategies, applied in order:

    1. NaN-missingness asymmetry: feature's missing rate differs by more than
       miss_diff_threshold between target groups. Catches follow-ups stored as NaN
       for members who were not asked the question.

    2. High Pearson correlation: |r| > corr_threshold between feature and target.
       Catches follow-ups encoded as a numeric "not applicable" code (0, 8, 9, etc.)
       rather than NaN — the numeric code makes the column appear non-missing while
       still being perfectly aligned with the target. Legitimate predictors (SEX, AGE,
       education, clinical items) rarely exceed r≈0.4 with a binary HOS outcome.

    Returns (excluded_cols, remaining_cols).
    """
    target_groups = df[target_col].dropna().unique()
    if len(target_groups) < 2:
        return [], feature_cols

    target_numeric = pd.to_numeric(df[target_col], errors="coerce")
    excluded, remaining = [], []

    for col in feature_cols:
        flagged = False

        # Check 1: NaN missingness asymmetry across target groups
        miss_rates = [
            df.loc[df[target_col] == g, col].isna().mean()
            for g in target_groups
        ]
        if max(miss_rates) - min(miss_rates) > miss_diff_threshold:
            flagged = True

        # Check 2: very high Pearson correlation → likely a numeric-coded follow-up
        if not flagged:
            try:
                col_numeric = pd.to_numeric(df[col], errors="coerce")
                paired = pd.concat([target_numeric, col_numeric], axis=1).dropna()
                if len(paired) > 30:
                    r, _ = stats.pearsonr(paired.iloc[:, 0], paired.iloc[:, 1])
                    if abs(r) > corr_threshold:
                        flagged = True
            except Exception:
                pass

        if flagged:
            excluded.append(col)
        else:
            remaining.append(col)

    return excluded, remaining


def _dummy_encode_multilevel(
    df: pd.DataFrame, feature_cols: list[str], schema: dict
) -> tuple[pd.DataFrame, dict[str, str]]:
    """One-hot encode columns with 3+ response levels (ordinal/nominal multi-level responses).

    Decision logic per column:
    - Schema has 3+ value_labels → encode using schema labels (primary path)
    - Schema has 0-2 value_labels but column has 3-15 integer-valued unique codes
      → encode using data-driven codes (fallback for columns the PDF parser missed)
    - Otherwise (binary, continuous) → keep as numeric

    Returns the expanded DataFrame and a label_map translating dummy column names
    (e.g. 'B25VRDOWN_3') to plain-English descriptions ('Downhearted and Blue: A good bit of the time').
    """
    to_encode: list[tuple[str, dict]] = []  # (col, vl_dict)
    to_keep: list[str] = []

    for col in feature_cols:
        vl = schema.get(col.upper(), {}).get("value_labels", {})
        if len(vl) >= 3:
            to_encode.append((col, vl))
        elif len(vl) < 3:
            # Fallback: check if column looks like a coded categorical in the data
            col_vals = pd.to_numeric(df[col], errors="coerce").dropna()
            is_integer_like = len(col_vals) > 0 and (col_vals == col_vals.round()).all()
            n_uniq = int(col_vals.nunique())
            if is_integer_like and 3 <= n_uniq <= 15:
                # Build codes from data; labels will just be the code value (no schema text)
                data_vl = {str(int(v)): str(int(v)) for v in sorted(col_vals.unique())}
                to_encode.append((col, data_vl))
            else:
                to_keep.append(col)

    if not to_encode:
        return df[feature_cols], {}

    label_map: dict[str, str] = {}
    dummy_frames = []
    for col, vl in to_encode:
        col_int = pd.to_numeric(df[col], errors="coerce").round().astype("Int64")
        for code_str, level_label in sorted(vl.items(), key=lambda x: int(x[0])):
            code = int(code_str)
            dummy_name = f"{col}_{code}"
            dummy_frames.append(
                pd.Series((col_int == code).astype(int), name=dummy_name, index=df.index)
            )
            label_map[dummy_name] = f"{col}: {level_label}"

    expanded = pd.concat([df[to_keep]] + dummy_frames, axis=1)
    return expanded, label_map


def run_feature_importance(
    dataset: str,
    target_column: str,
    top_n: int = 10,
    subset_column: Optional[str] = None,
    subset_codes: Optional[list] = None,
    encode_multilevel: bool = True,
) -> str:
    """Run Random Forest feature importance to identify predictors of target_column.

    Multi-level survey responses (3+ coded answer levels, e.g. Likert scales) are
    one-hot encoded by default so the model learns independent weights per response
    level rather than treating the numeric codes as a continuous ordinal scale.
    This matches the CMS approach for HOS survey data.

    Args:
        dataset: Dataset name or partial name.
        target_column: Binary or categorical outcome column (case-insensitive).
        top_n: Number of top features to return (default 10).
        subset_column: Optional column to filter on (e.g. "age group" or "AGE").
        subset_codes: Exact numeric codes to keep (e.g. [2, 3] for AGE 65+).
                      Look up valid codes with run_categorical_analysis first.
        encode_multilevel: If True (default), one-hot encode columns with ≥3 coded
                           response levels before fitting. Binary and continuous
                           columns are always kept as numeric.

    Returns:
        Ranked list of feature importances with plain-English labels. When
        encode_multilevel=True, reports total importance per original variable
        (sum across its dummy columns) plus the single most important response level.
    """
    df = get_dataset(dataset)
    df, filter_desc = _resolve_subset(df, subset_column, subset_codes)
    schema = _load_schema()

    target = _find_column(df, target_column)
    feature_cols = [c for c in df.select_dtypes(include="number").columns if c != target]

    # Exclude conditional follow-up questions: survey items only answered by members
    # who already reported the condition (e.g. "talked to doctor about urine leakage"
    # is only asked if the member has urine leakage). Including them causes feature
    # leakage — they dominate importance and suppress genuine predictors like SEX/AGE.
    excluded_followups, feature_cols = _detect_conditional_followups(df, target, feature_cols)
    if excluded_followups:
        excluded_labels = [
            schema.get(c.upper(), {}).get("description", c)[:50] for c in excluded_followups
        ]
        logger.info(
            "Excluded %d conditional follow-up columns: %s",
            len(excluded_followups), excluded_labels,
        )

    sub = df[[target] + feature_cols].dropna()
    y = sub[target]

    if encode_multilevel:
        X_expanded, label_map = _dummy_encode_multilevel(sub, feature_cols, schema)
        X = X_expanded.fillna(0)
    else:
        X = sub[feature_cols]
        label_map = {}

    # n_estimators=50 and max_samples=0.3 keep results statistically solid
    # while cutting fit time ~3-4x on large HOS datasets (263k+ rows).
    clf = RandomForestClassifier(n_estimators=50, max_samples=0.3, random_state=42, n_jobs=-1)
    clf.fit(X, y)

    # SHAP TreeExplainer: more reliable than Gini importance — handles correlated
    # features and dummy-encoded groups correctly. Sample up to 5k rows for speed.
    try:
        import shap
        X_shap = X.sample(min(5000, len(X)), random_state=42)
        explainer = shap.TreeExplainer(clf)
        # Use the new callable API (SHAP 0.41+): returns an Explanation with a
        # consistent .values array — avoids ambiguous list-vs-array output of shap_values().
        shap_exp = explainer(X_shap)
        arr = np.abs(shap_exp.values)
        # arr shape:
        #   binary classification → (n_samples, n_features)
        #   multi-class           → (n_samples, n_features, n_classes)
        if arr.ndim == 3:
            mean_abs = arr.mean(axis=(0, 2))   # average over samples and classes → (n_features,)
        else:
            mean_abs = arr.mean(axis=0)        # average over samples → (n_features,)
        dummy_importance = dict(zip(X.columns, mean_abs.tolist()))
        importance_method = "SHAP"
    except Exception as exc:
        logger.warning("SHAP failed, falling back to RF Gini importance: %s", exc)
        dummy_importance = dict(zip(X.columns, clf.feature_importances_))
        importance_method = "RF Gini"

    if encode_multilevel:
        # Aggregate dummy importances back to original variable level
        var_importance: dict[str, float] = {}
        var_top_level: dict[str, tuple[str, float]] = {}  # col -> (level_label, imp)
        for dummy_col, imp in dummy_importance.items():
            # Recover original column name: everything before the last underscore+digits
            orig = dummy_col.rsplit("_", 1)[0] if dummy_col in label_map else dummy_col
            var_importance[orig] = var_importance.get(orig, 0.0) + imp
            if dummy_col in label_map:
                current = var_top_level.get(orig, ("", 0.0))
                if imp > current[1]:
                    var_top_level[orig] = (label_map[dummy_col], imp)
    else:
        var_importance = dummy_importance
        var_top_level = {}

    ranked_vars = sorted(var_importance.items(), key=lambda x: x[1], reverse=True)
    total_vars = len(ranked_vars)

    encoding_note = ", dummy-encoded multi-level responses" if encode_multilevel else ""
    header = f"Top {top_n} predictors of {target_column} in {dataset} (Random Forest + {importance_method}{encoding_note})"
    if filter_desc:
        header += f" {filter_desc}"
    lines = [header + ":\n"]
    if excluded_followups:
        excl_descs = [schema.get(c.upper(), {}).get("description", c)[:60] for c in excluded_followups]
        lines.append(
            f"  [Note: {len(excluded_followups)} conditional follow-up column(s) auto-excluded to prevent "
            f"feature leakage: {'; '.join(excl_descs)}]\n"
        )

    for rank_i, (col, total_imp) in enumerate(ranked_vars[:top_n], 1):
        entry = schema.get(col.upper(), {})
        label = entry.get("description", col)
        line = f"  #{rank_i} {col} ({label}): total_importance={round(total_imp, 4)}"
        if col in var_top_level:
            top_label, top_imp = var_top_level[col]
            # strip "col: " prefix already in label_map value
            level_only = top_label.split(": ", 1)[-1]
            line += f"  [strongest level: '{level_only}' imp={round(top_imp, 4)}]"
        lines.append(line)

    # Always report demographic variables separately so they are not silently
    # excluded when clinical items dominate the top_n ranking.
    _DEMOGRAPHIC_KEYWORDS = ["age", "sex", "race", "education", "marital", "educ"]
    demo_orig = [c for c in var_importance if any(kw in c.lower() for kw in _DEMOGRAPHIC_KEYWORDS)]
    if demo_orig:
        lines.append("\nDemographic variable importances (shown regardless of overall rank):")
        all_ranked = [c for c, _ in ranked_vars]
        for col in sorted(demo_orig, key=lambda c: var_importance[c], reverse=True):
            entry = schema.get(col.upper(), {})
            label = entry.get("description", col)
            rank = all_ranked.index(col) + 1
            line = (f"  {col} ({label}): total_importance={round(var_importance[col], 4)}"
                    f", rank #{rank} of {total_vars}")
            if col in var_top_level:
                top_label, top_imp = var_top_level[col]
                level_only = top_label.split(": ", 1)[-1]
                line += f"  [strongest level: '{level_only}' imp={round(top_imp, 4)}]"
            lines.append(line)

    return "\n".join(lines)


def run_logistic_regression(
    dataset: str,
    target_column: str,
    feature_columns: list[str],
    positive_values: Optional[list] = None,
    subset_column: Optional[str] = None,
    subset_codes: Optional[list] = None,
) -> str:
    """Run logistic regression predicting target_column from feature_columns.

    Args:
        dataset: Dataset name or partial name.
        target_column: Binary outcome column. If multi-valued, provide positive_values to recode.
        feature_columns: List of predictor column names.
        positive_values: Values of target_column to treat as 1 (positive class).
                         Required when target has more than 2 unique values.
        subset_column: Optional column to filter on (e.g. "age group" or "AGE").
        subset_codes: Exact numeric codes to keep (e.g. [2, 3] for AGE 65+).

    Returns:
        Coefficients, odds ratios, p-values, and model accuracy.
    """
    df = get_dataset(dataset)
    df, filter_desc = _resolve_subset(df, subset_column, subset_codes)
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

    header = f"Logistic regression — target: {target_column}, dataset: {dataset}"
    if filter_desc:
        header += f" {filter_desc}"
    lines = [header + "\n", f"Accuracy: {accuracy}\n", "Coefficients:"]
    for col, coef in zip(features, model.coef_[0]):
        label = schema.get(col.upper(), {}).get("description", col)
        or_ = round(float(np.exp(coef)), 4)
        lines.append(f"  {col} ({label}): coef={round(coef, 4)}, OR={or_}")
    return "\n".join(lines)


def run_categorical_analysis(
    dataset: str,
    column1: str,
    column2: Optional[str] = None,
    subset_column: Optional[str] = None,
    subset_codes: Optional[list] = None,
) -> str:
    """Run frequency table (1 column) or cross-tabulation + chi-square + Cramér's V (2 columns).

    Args:
        dataset: Dataset name or partial name.
        column1: First column — required.
        column2: Second column — if provided, runs crosstab + chi-square + Cramér's V.
        subset_column: Optional column to filter on (e.g. "age group" or "AGE").
        subset_codes: Exact numeric codes to keep (e.g. [2, 3] for AGE 65+).

    Returns:
        Frequency table or crosstab with chi-square statistics and effect size.
    """
    df = get_dataset(dataset)
    df, filter_desc = _resolve_subset(df, subset_column, subset_codes)
    schema = _load_schema()

    col1 = _find_column(df, column1)

    if column2 is None:
        freq = df[col1].value_counts().sort_index()
        entry = schema.get(col1.upper(), {})
        value_map = entry.get("values", {})
        header = f"Frequency table for {col1} ({entry.get('description', col1)}) in {dataset}"
        if filter_desc:
            header += f" {filter_desc}"
        lines = [header + ":\n"]
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

    header = f"Cross-tabulation: {col1} × {col2} in {dataset}"
    if filter_desc:
        header += f" {filter_desc}"
    lines = [
        header,
        f"Chi-square={round(chi2, 2)}, p={round(p, 6)}, dof={dof}, Cramér's V={cramers_v}\n",
        ct.to_string(),
    ]
    return "\n".join(lines)


def run_group_comparison(
    dataset: str,
    value_column: str,
    group_column: str,
    subset_column: Optional[str] = None,
    subset_codes: Optional[list] = None,
) -> str:
    """Compare value_column across groups defined by group_column.

    Automatically selects Mann-Whitney U (2 groups) or Kruskal-Wallis (3+ groups).

    Args:
        dataset: Dataset name or partial name.
        value_column: Continuous or ordinal variable to compare (case-insensitive).
        group_column: Categorical variable that defines groups (case-insensitive).
        subset_column: Optional column to filter on (e.g. "age group" or "AGE").
        subset_codes: Exact numeric codes to keep (e.g. [2, 3] for AGE 65+).

    Returns:
        Test statistic, p-value, and group medians with decoded labels.
    """
    df = get_dataset(dataset)
    df, filter_desc = _resolve_subset(df, subset_column, subset_codes)
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
    header = f"Group comparison: {val_col} ({val_entry.get('description', val_col)}) by {grp_col}"
    if filter_desc:
        header += f" {filter_desc}"
    lines = [header + "\n"]
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
