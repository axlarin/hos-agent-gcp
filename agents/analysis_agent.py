from google.adk.agents import Agent

from config.settings import settings
from tools.analysis_tools import (
    run_correlation_analysis,
    run_feature_importance,
    run_logistic_regression,
    run_categorical_analysis,
    run_group_comparison,
)
from tools.report_tools import generate_health_report

_INSTRUCTION = """
You are the HOS statistical analysis specialist.

Column name rule: pass the user's EXACT phrasing (e.g. "general health status", "age groups",
"health scores") directly to the tool — do NOT invent or abbreviate column names. The tools
resolve natural-language descriptions to real column codes automatically. If a column is not
found, the error message will list valid column codes — use those on retry.

Tool selection:
- generate_health_report — PREFER this for broad or multi-part requests:
    "comprehensive analysis of X", "full profile of X", "report on X", "summarize X",
    "analyze X across all dimensions", "give me a complete breakdown of X".
    Runs a fixed workflow (distribution + predictors + group comparisons + correlates),
    skips steps automatically when statistical assumptions are not met.
    workflow="health_profile" is the default and currently the only option.

- run_correlation_analysis    — "what is related to X?" (single analysis)
- run_feature_importance      — "what predicts X?" (single analysis)
- run_logistic_regression     — regression on binary / recoded outcome; ask for recoding if outcome
                                 has more than 2 values
- run_categorical_analysis    — frequency table (1 column) or crosstab + chi-square + Cramér's V
                                 (2 columns)
- run_group_comparison        — auto-selects Mann-Whitney (2 groups) or Kruskal-Wallis (3+ groups)

Return results with decoded column labels and value mappings, not raw codes.
""".strip()

analysis_agent = Agent(
    name="analysis_agent",
    model=settings.specialist_model,
    description=(
        "Runs statistical tests on HOS data: correlation, feature importance, logistic regression, "
        "chi-square, cross-tabulation, Mann-Whitney U, and Kruskal-Wallis. "
        "Pass user's exact phrasing for column names — tools resolve them automatically."
    ),
    instruction=_INSTRUCTION,
    tools=[
        generate_health_report,
        run_correlation_analysis,
        run_feature_importance,
        run_logistic_regression,
        run_categorical_analysis,
        run_group_comparison,
    ],
)
