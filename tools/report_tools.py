"""Multi-step analytical workflow engine for HOS data.

Key design decisions:
  - WORKFLOWS is the single source of truth: adding a workflow only requires a
    new entry there; _run_workflow and generate_health_report are workflow-agnostic.
  - Gate thresholds are collected in GateConfig, passed down to every gate function,
    so callers (API, tests) can tune them without touching the workflow definitions.
  - _run_workflow returns a WorkflowResult with both the markdown and a structured
    execution trace (status/reason/duration per step). The ADK tool
    generate_health_report exposes only the markdown string (LLM-friendly); the full
    trace is written to outputs/ and returned by the /report endpoint.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Tuple

import pandas as pd

from tools.csv_tools import get_dataset

logger = logging.getLogger(__name__)


# ── Gate configuration ────────────────────────────────────────────────────────

@dataclass
class GateConfig:
    """Thresholds used by validation gate functions.

    All workflow steps share one GateConfig instance so callers can tune behaviour
    (e.g. tighter missing-data tolerance for research use) without touching
    workflow definitions or gate implementations.
    """
    max_missing_pct: float = 0.5
    max_unique_for_classification: int = 20
    min_obs_per_group: int = 5


# ── Validation gate functions ─────────────────────────────────────────────────
# Signature: (df, col, config) -> (should_skip: bool, reason: str)
# Uniform three-arg signature even when config is unused, so every gate is
# interchangeable in the WorkflowStep.skip_if slot without adapters.

def _gate_needs_multiple_values(
    df: pd.DataFrame, col: str, config: GateConfig
) -> Tuple[bool, str]:
    n = df[col].nunique(dropna=True)
    return (n < 2, f"column has only {n} unique value(s) — distribution not meaningful")


def _gate_needs_classification_target(
    df: pd.DataFrame, col: str, config: GateConfig
) -> Tuple[bool, str]:
    n = df[col].nunique(dropna=True)
    if n < 2:
        return True, f"target has only {n} unique value(s) — need ≥ 2 classes"
    if n > config.max_unique_for_classification:
        return True, (
            f"target has {n} unique values (threshold {config.max_unique_for_classification}) — "
            "likely a continuous variable; Pearson correlation is more appropriate"
        )
    return False, ""


def _gate_needs_sex_column(
    df: pd.DataFrame, col: str, config: GateConfig
) -> Tuple[bool, str]:
    sex_col = next((c for c in df.columns if c.upper() == "SEX"), None)
    if sex_col is None:
        return True, "SEX column not found in dataset"
    n = df[sex_col].nunique(dropna=True)
    if n < 2:
        return True, f"SEX column has only {n} unique value(s) — need ≥ 2 groups"
    return False, ""


def _gate_needs_age_column(
    df: pd.DataFrame, col: str, config: GateConfig
) -> Tuple[bool, str]:
    age_col = next((c for c in df.columns if c.upper() == "AGE"), None)
    if age_col is None:
        return True, "AGE column not found in dataset"
    n = df[age_col].nunique(dropna=True)
    if n < 2:
        return True, f"AGE column has only {n} unique value(s) — need ≥ 2 groups"
    return False, ""


def _gate_needs_numeric_target(
    df: pd.DataFrame, col: str, config: GateConfig
) -> Tuple[bool, str]:
    if not pd.api.types.is_numeric_dtype(df[col]):
        return True, (
            f"target column dtype is {df[col].dtype} — "
            "Pearson correlation requires a numeric column"
        )
    return False, ""


def _gate_needs_sufficient_data(
    df: pd.DataFrame, col: str, config: GateConfig
) -> Tuple[bool, str]:
    missing_pct = df[col].isna().mean()
    if missing_pct > config.max_missing_pct:
        return True, (
            f"{missing_pct:.0%} of target values are missing "
            f"(threshold {config.max_missing_pct:.0%}) — results would be unreliable"
        )
    return False, ""


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class WorkflowStep:
    name: str                                      # section header in the report
    analysis: str                                  # key into _STEP_DISPATCH
    params: dict = field(default_factory=dict)     # extra kwargs beyond dataset/col
    skip_if: Optional[Callable] = None             # (df, col, config) -> (bool, reason)


@dataclass
class StepResult:
    name: str
    analysis: str
    status: str            # "completed" | "skipped" | "failed"
    reason: str = ""       # gate reason if skipped, error message if failed
    output: str = ""       # full tool output if completed
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "analysis": self.analysis,
            "status": self.status,
            "reason": self.reason,
            "output_chars": len(self.output),
            "duration_ms": round(self.duration_ms, 1),
        }


@dataclass
class WorkflowResult:
    dataset: str
    target_column: str
    resolved_column: str
    workflow: str
    config: GateConfig
    steps: list[StepResult]
    report_markdown: str

    @property
    def summary(self) -> dict:
        by_status = {"completed": 0, "skipped": 0, "failed": 0}
        for s in self.steps:
            by_status[s.status] = by_status.get(s.status, 0) + 1
        return {**by_status, "total": len(self.steps)}

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "target_column": self.target_column,
            "resolved_column": self.resolved_column,
            "workflow": self.workflow,
            "config": asdict(self.config),
            "summary": self.summary,
            "steps": [s.to_dict() for s in self.steps],
            "report_markdown": self.report_markdown,
        }


# ── Workflow definitions ── single source of truth ───────────────────────────

WORKFLOWS: dict[str, list[WorkflowStep]] = {
    "health_profile": [
        WorkflowStep(
            name="Frequency Distribution",
            analysis="categorical",
            skip_if=_gate_needs_multiple_values,
        ),
        WorkflowStep(
            name="Top Predictors (Random Forest)",
            analysis="feature_importance",
            params={"top_n": 10},
            skip_if=_gate_needs_classification_target,
        ),
        WorkflowStep(
            name="Group Comparison by Sex",
            analysis="group_by_sex",
            skip_if=_gate_needs_sex_column,
        ),
        WorkflowStep(
            name="Group Comparison by Age",
            analysis="group_by_age",
            skip_if=_gate_needs_age_column,
        ),
        WorkflowStep(
            name="Top Correlates (Pearson)",
            analysis="correlation",
            params={"top_n": 10},
            skip_if=_gate_needs_numeric_target,
        ),
    ],
}


# ── Step dispatcher ───────────────────────────────────────────────────────────

def _dispatch_step(step: WorkflowStep, dataset: str, col: str) -> str:
    from tools.analysis_tools import (
        run_categorical_analysis,
        run_correlation_analysis,
        run_feature_importance,
        run_group_comparison,
    )
    dispatch = {
        "categorical":        lambda: run_categorical_analysis(dataset, col),
        "feature_importance": lambda: run_feature_importance(dataset, col, **step.params),
        "group_by_sex":       lambda: run_group_comparison(dataset, col, "SEX"),
        "group_by_age":       lambda: run_group_comparison(dataset, col, "AGE"),
        "correlation":        lambda: run_correlation_analysis(dataset, col, **step.params),
    }
    fn = dispatch.get(step.analysis)
    if fn is None:
        raise ValueError(f"Unknown analysis type: {step.analysis!r}")
    return fn()


# ── Core execution engine ─────────────────────────────────────────────────────

def _run_workflow(
    dataset: str,
    target_column: str,
    workflow: str = "health_profile",
    config: Optional[GateConfig] = None,
) -> WorkflowResult:
    """Execute a named workflow and return both the report and the execution trace.

    This is the internal execution engine. The ADK tool generate_health_report calls
    this and extracts report_markdown; the /report endpoint returns the full
    WorkflowResult.to_dict() for programmatic access and testing.
    """
    if config is None:
        config = GateConfig()

    if workflow not in WORKFLOWS:
        available = ", ".join(f'"{w}"' for w in WORKFLOWS)
        md = f'# HOS Report — Error\n\nUnknown workflow "{workflow}". Available: {available}'
        return WorkflowResult(
            dataset=dataset, target_column=target_column, resolved_column="",
            workflow=workflow, config=config, steps=[], report_markdown=md,
        )

    try:
        df = get_dataset(dataset)
    except KeyError as exc:
        md = f"# HOS Report — Error\n\nDataset error: {exc}"
        return WorkflowResult(
            dataset=dataset, target_column=target_column, resolved_column="",
            workflow=workflow, config=config, steps=[], report_markdown=md,
        )

    from tools.analysis_tools import _find_column
    try:
        col = _find_column(df, target_column)
    except KeyError as exc:
        md = f"# HOS Report — Error\n\nColumn error: {exc}"
        return WorkflowResult(
            dataset=dataset, target_column=target_column, resolved_column="",
            workflow=workflow, config=config, steps=[], report_markdown=md,
        )

    # Global data-quality gate — aborts before running any step
    skip, reason = _gate_needs_sufficient_data(df, col, config)
    if skip:
        md = (
            f"# HOS Health Profile Report — Aborted\n\n"
            f"**Dataset:** {dataset}  |  **Target:** {target_column} (`{col}`)\n\n"
            f"**Cannot proceed:** {reason}"
        )
        return WorkflowResult(
            dataset=dataset, target_column=target_column, resolved_column=col,
            workflow=workflow, config=config, steps=[], report_markdown=md,
        )

    step_results: list[StepResult] = []
    sections: list[str] = []

    for step in WORKFLOWS[workflow]:
        should_skip, skip_reason = False, ""
        if step.skip_if is not None:
            try:
                should_skip, skip_reason = step.skip_if(df, col, config)
            except Exception as exc:
                should_skip, skip_reason = True, f"validation error — {exc}"

        if should_skip:
            step_results.append(StepResult(
                name=step.name, analysis=step.analysis,
                status="skipped", reason=skip_reason,
            ))
            continue

        t0 = time.perf_counter()
        try:
            output = _dispatch_step(step, dataset, col)
            duration_ms = (time.perf_counter() - t0) * 1000
            step_results.append(StepResult(
                name=step.name, analysis=step.analysis,
                status="completed", output=output, duration_ms=duration_ms,
            ))
            sections.append(f"## {step.name}\n\n{output}")
        except Exception as exc:
            duration_ms = (time.perf_counter() - t0) * 1000
            step_results.append(StepResult(
                name=step.name, analysis=step.analysis,
                status="failed", reason=str(exc), duration_ms=duration_ms,
            ))
            logger.warning("Workflow step '%s' failed: %s", step.name, exc)

    # Build execution metadata block
    summary = {s.status: 0 for s in step_results}
    for s in step_results:
        summary[s.status] += 1
    meta_lines = [
        f"| Step | Status | Note |",
        f"|---|---|---|",
    ]
    for s in step_results:
        note = s.reason if s.status in ("skipped", "failed") else f"{s.duration_ms:.0f} ms"
        meta_lines.append(f"| {s.name} | {s.status} | {note} |")

    report_lines = [
        "# HOS Health Profile Report",
        f"**Dataset:** {dataset}  |  **Target:** {target_column} (`{col}`)",
        "---",
    ]
    report_lines.extend(sections)
    report_lines.append(
        "## Execution Summary\n\n"
        + f"Completed: {summary.get('completed', 0)}  |  "
        + f"Skipped: {summary.get('skipped', 0)}  |  "
        + f"Failed: {summary.get('failed', 0)}\n\n"
        + "\n".join(meta_lines)
    )

    return WorkflowResult(
        dataset=dataset,
        target_column=target_column,
        resolved_column=col,
        workflow=workflow,
        config=config,
        steps=step_results,
        report_markdown="\n\n".join(report_lines),
    )


# ── ADK tool ─────────────────────────────────────────────────────────────────

def generate_health_report(
    dataset: str,
    target_column: str,
    workflow: str = "health_profile",
) -> str:
    """Run a multi-step analytical workflow and return a structured markdown report.

    Each workflow is a named sequence of analysis steps defined in WORKFLOWS. Before
    each step, a validation gate checks whether required statistical assumptions hold
    (e.g. numeric target for Pearson correlation, ≤ 20 unique values for Random Forest
    classification, SEX/AGE column present for group comparisons). Steps that fail
    their gate are skipped with an explanation in the Execution Summary section.

    Use this tool instead of calling individual analysis tools when the user asks for a
    comprehensive or multi-part analysis — e.g.:
      "comprehensive analysis of X", "full profile of X", "report on X", "summarize X",
      "analyze general health across all dimensions", "give me a complete breakdown of X".

    Args:
        dataset: Dataset name or partial name (e.g. "c25a").
        target_column: Primary outcome variable. Accepts plain-English descriptions
            (e.g. "general health status") — resolved to the actual column code
            automatically.
        workflow: Named workflow to execute. Available:
            - "health_profile" — distribution + top predictors (Random Forest) + group
              comparisons by sex and age + top Pearson correlates. Designed for ordinal
              and categorical HOS outcome variables.

    Returns:
        Structured markdown report with one section per completed analysis, plus an
        Execution Summary table showing the status and timing of every step.
    """
    result = _run_workflow(dataset, target_column, workflow)

    # Persist the execution trace for observability and testing
    try:
        from config.settings import settings
        trace_path = Path(settings.report_trace_path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(json.dumps(result.to_dict(), indent=2))
        logger.info("Report trace written to %s", trace_path)
    except Exception as exc:
        logger.warning("Could not write report trace: %s", exc)

    return result.report_markdown
